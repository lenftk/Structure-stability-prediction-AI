import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import timm

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Configs
# ---------------------------------------------------------
class CFG:
    seed = 42                 # 시드 고정
    epochs = 12               # 학습 에폭
    batch_size = 16           # 배치 사이즈 (메모리 부족 시 8로 감소)
    img_size = 224            # 입력 이미지 크기
    lr = 3e-4                 # 학습률
    min_lr = 1e-6             # 최소 학습률 (CosineAnnealing 패턴)
    weight_decay = 1e-2       # Weight Decay
    model_name = 'convnext_tiny' # Timm 라이브러리 지원 모델 (EfficientNet, Swin 등도 가능)
    num_classes = 2           # unstable, stable 2개
    n_fold = 5                # 교차 검증 K-Fold
    patience = 4              # Early Stopping 인내심
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# 2. Seed Fix
# ---------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)

# ---------------------------------------------------------
# 3. Data Processing
# ---------------------------------------------------------
BASE_PATH = './open'

train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
dev_df = pd.read_csv(f'{BASE_PATH}/dev.csv')
test_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

# Train과 Dev 합치기 (dev 데이터도 학습에 사용할 수 있다고 대회 규칙에 명시됨)
# 고정 카메라(train) 편향을 줄이기 위해 다양한 환경의 dev 데이터를 함께 학습합니다.
train_df['source'] = 'train'
dev_df['source'] = 'dev'
all_train_df = pd.concat([train_df, dev_df], axis=0).reset_index(drop=True)

# unstable: 0, stable: 1 로 라벨링
# sample_submission.csv 에 정의된 컬럼 순서가 (unstable_prob, stable_prob) 이므로 인덱스를 일치시킵니다.
all_train_df['target'] = all_train_df['label'].map({'unstable': 0, 'stable': 1})

# 이미지 경로 매핑
def get_img_paths(df, mode='train'):
    front_paths = []
    top_paths = []
    for i, row in df.iterrows():
        id_ = row['id']
        # source에 따라 하위 폴더가 다름
        folder = 'test' if mode == 'test' else row['source']
        f_path = os.path.join(BASE_PATH, folder, id_, 'front.png')
        t_path = os.path.join(BASE_PATH, folder, id_, 'top.png')
        front_paths.append(f_path)
        top_paths.append(t_path)
    
    df['front_path'] = front_paths
    df['top_path'] = top_paths
    return df

all_train_df = get_img_paths(all_train_df, mode='train')
# test 셋은 source가 없으므로 별도 처리
front_paths = []
top_paths = []
for i, row in test_df.iterrows():
    id_ = row['id']
    f_path = os.path.join(BASE_PATH, 'test', id_, 'front.png')
    t_path = os.path.join(BASE_PATH, 'test', id_, 'top.png')
    front_paths.append(f_path)
    top_paths.append(t_path)
test_df['front_path'] = front_paths
test_df['top_path'] = top_paths

# ---------------------------------------------------------
# 4. Dataset 정의
# ---------------------------------------------------------
class StructureDataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.df = df
        self.front_paths = df['front_path'].values
        self.top_paths = df['top_path'].values
        self.mode = mode
        if self.mode != 'test':
            self.targets = df['target'].values
            
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        front_path = self.front_paths[idx]
        top_path = self.top_paths[idx]
        
        # BGR -> RGB 로 읽고 변환
        front_img = cv2.imread(front_path)
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        
        top_img = cv2.imread(top_path)
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            # Front 이미지와 Top 이미지를 독립적으로 Augmentation
            # 혹은 환경 편향을 막기 위해 동일한 Augmentation을 적용할 수도 있으나,
            # 여기서는 각각 독립적으로 적용하여 다양성을 극대화합니다.
            front_res = self.transform(image=front_img)
            front_img = front_res['image']
            
            top_res = self.transform(image=top_img)
            top_img = top_res['image']
            
        if self.mode != 'test':
            target = self.targets[idx]
            return front_img, top_img, torch.tensor(target, dtype=torch.long)
        else:
            return front_img, top_img

# ---------------------------------------------------------
# 5. Transforms (데이터 증강)
# 고정 카메라(Train)로만 학습하면 이동 카메라(Dev/Test)에서 점수가 떨어집니다.
# ---------------------------------------------------------
def get_transforms(stage='train'):
    if stage == 'train':
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.HorizontalFlip(p=0.5), # 좌우 반전은 물리 법칙에서 안정성에 큰 영향을 미치지 않음
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5), # 카메라 흔들림 시뮬레이션
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), # 조명 변화
            A.Perspective(scale=(0.02, 0.08), p=0.3), # 카메라 각도 변화
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CFG.img_size, CFG.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ---------------------------------------------------------
# 6. Model (Multi-View CNN)
# ---------------------------------------------------------
class DualViewModel(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=True):
        super(DualViewModel, self).__init__()
        # Timm 라이브러리 사용. num_classes=0 은 최종 분류기 층을 오프하고 특징 벡터만 추출
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        in_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features * 2, 512), # Front와 Top 특징 결합
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, CFG.num_classes)
        )
        
    def forward(self, front, top):
        f_feat = self.backbone(front) # (B, F)
        t_feat = self.backbone(top)   # (B, F)
        
        combined = torch.cat([f_feat, t_feat], dim=1) # (B, F * 2)
        logits = self.classifier(combined)           # (B, 2)
        return logits

# ---------------------------------------------------------
# 7. Training & Evaluation Functions
# ---------------------------------------------------------
def train_fn(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for front, top, targets in tqdm(dataloader, desc='Training', leave=False):
        front, top, targets = front.to(device), top.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(front, top)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if scheduler is not None:
        scheduler.step()
        
    return total_loss / len(dataloader)

def valid_fn(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    
    with torch.no_grad():
        for front, top, targets in tqdm(dataloader, desc='Validating', leave=False):
            front, top, targets = front.to(device), top.to(device), targets.to(device)
            
            logits = model(front, top)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
            
    preds = np.vstack(preds)
    return total_loss / len(dataloader), preds

# ---------------------------------------------------------
# 8. K-Fold Training Loop
# ---------------------------------------------------------
folds = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
all_train_df['fold'] = -1
for fold_idx, (train_idx, val_idx) in enumerate(folds.split(all_train_df, all_train_df['target'])):
    all_train_df.loc[val_idx, 'fold'] = fold_idx

oof_preds = np.zeros((len(all_train_df), CFG.num_classes))
test_preds = np.zeros((len(test_df), CFG.num_classes))

# 런타임에 GPU 캐시 정리를 위한 로직 추가
for fold in range(CFG.n_fold):
    print(f"========== Fold: {fold} ==========")
    
    train_data = all_train_df[all_train_df['fold'] != fold].reset_index(drop=True)
    val_data = all_train_df[all_train_df['fold'] == fold].reset_index(drop=True)
    
    train_dataset = StructureDataset(train_data, transform=get_transforms('train'), mode='train')
    val_dataset = StructureDataset(val_data, transform=get_transforms('valid'), mode='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = DualViewModel(CFG.model_name, pretrained=True)
    model.to(CFG.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)
    
    best_loss = np.inf
    patience_cnt = 0
    
    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_fn(model, train_loader, criterion, optimizer, scheduler, CFG.device)
        val_loss, val_preds = valid_fn(model, val_loader, criterion, CFG.device)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            oof_preds[all_train_df['fold'] == fold] = val_preds
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            print("  => Saved Best Model!")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CFG.patience:
                print(f"Early Stopping at Epoch {epoch}")
                break
                
    # Free Memory
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # 9. Test Inference (Ensemble)
    # ---------------------------------------------------------
    print(f"Inferencing Fold {fold} Test Data...")
    test_dataset = StructureDataset(test_df, transform=get_transforms('valid'), mode='test')
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
    
    model = DualViewModel(CFG.model_name, pretrained=False)
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
    model.to(CFG.device)
    model.eval()
    
    fold_test_preds = []
    with torch.no_grad():
        for front, top in tqdm(test_loader, desc=f'Test Fold {fold}'):
            front, top = front.to(CFG.device), top.to(CFG.device)
            logits = model(front, top)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            fold_test_preds.append(probs)
            
    # 전체 Fold 예측 결과 평균
    test_preds += np.vstack(fold_test_preds) / CFG.n_fold
    
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

# ---------------------------------------------------------
# 10. Submission
# ---------------------------------------------------------
# OOF 앙상블 출력은 필요 시 분석용으로 사용.
# test_preds 배열 모양: (N, 2)
test_df['unstable_prob'] = test_preds[:, 0]
test_df['stable_prob'] = test_preds[:, 1]
# id, unstable_prob, stable_prob 만 저장
test_df[['id', 'unstable_prob', 'stable_prob']].to_csv('top3_submission.csv', index=False)
print("대회 예측 완료! top3_submission.csv 가 생성되었습니다.")
