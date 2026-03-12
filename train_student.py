import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import timm
from albumentations.pytorch import ToTensorV2
import albumentations as A

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

from main import CFG, StructureDataset, get_transforms, get_img_paths
from model_advanced import AdvancedDualViewModel

# ---------------------------------------------------------
# Seed Fix
# ---------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)

BASE_PATH = './open'

# ---------------------------------------------------------
# 1. 원본 Data 로드
# ---------------------------------------------------------
train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
train_df['source'] = 'train'

dev_df = pd.read_csv(f'{BASE_PATH}/dev.csv')
dev_df['source'] = 'dev'

# ---------------------------------------------------------
# 2. Pseudo Label Data 로드 (미리 생성해둔 파일)
# ---------------------------------------------------------
pseudo_path = 'pseudo_labels.csv'
if not os.path.exists(pseudo_path):
    raise FileNotFoundError(f"{pseudo_path} 가 없습니다. 먼저 generate_pseudo_labels.py 를 실행하세요.")

pseudo_df = pd.read_csv(pseudo_path)
# Test 폴더 하위의 데이터이므로 source는 'test'로 설정 (이미 csv에 설정되어 있음)
if 'source' not in pseudo_df.columns:
    pseudo_df['source'] = 'test'

print(f"Original Train: {len(train_df)} | Dev: {len(dev_df)} | Pseudo Labels: {len(pseudo_df)}")

# 전체 데이터 병합 (Train + Dev + Pseudo-labeled Test)
all_train_df = pd.concat([train_df, dev_df, pseudo_df], axis=0).reset_index(drop=True)
all_train_df['target'] = all_train_df['label'].map({'unstable': 0, 'stable': 1})
all_train_df = get_img_paths(all_train_df, mode='train')

# Test Data (제출용) 로드
test_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
test_df = get_img_paths(test_df, mode='test')

# ---------------------------------------------------------
# 3. Training / Validation / TTA Ensemble Logic 
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
    if scheduler is not None: scheduler.step()
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
    return total_loss / len(dataloader), np.vstack(preds)

def get_tta_transforms():
    """ TTA (Test-Time Augmentation)를 위한 변환 목록 생성 """
    return [
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]), # 원본
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]), # 좌우반전
        A.Compose([A.Resize(CFG.img_size, CFG.img_size), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]) # 밝기조절
    ]

if __name__ == '__main__':
    import gc
    
    folds = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    all_train_df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(all_train_df, all_train_df['target'])):
        all_train_df.loc[val_idx, 'fold'] = fold_idx

    oof_preds = np.zeros((len(all_train_df), CFG.num_classes))
    test_preds = np.zeros((len(test_df), CFG.num_classes))
    tta_transforms = get_tta_transforms()

    for fold in range(CFG.n_fold):
        print(f"========== Student Fold: {fold} ==========")
        train_data = all_train_df[all_train_df['fold'] != fold].reset_index(drop=True)
        val_data = all_train_df[all_train_df['fold'] == fold].reset_index(drop=True)
        
        train_dataset = StructureDataset(train_data, transform=get_transforms('train'), mode='train')
        val_dataset = StructureDataset(val_data, transform=get_transforms('valid'), mode='valid')
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Advanced Model 사용 (Swin Transformer 등 파라미터 변경 가능)
        model = AdvancedDualViewModel(model_name='convnext_tiny', pretrained=True).to(CFG.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr)
        
        best_loss = np.inf
        for epoch in range(1, CFG.epochs + 1):
            train_loss = train_fn(model, train_loader, criterion, optimizer, scheduler, CFG.device)
            val_loss, val_preds = valid_fn(model, val_loader, criterion, CFG.device)
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                oof_preds[all_train_df['fold'] == fold] = val_preds
                torch.save(model.state_dict(), f'student_best_fold{fold}.pth')
                print("  => Saved Best Student Model!")

        del model, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

        print(f"TTA Inferencing Student Fold {fold} Test Data...")
        model = AdvancedDualViewModel(model_name='convnext_tiny', pretrained=False).to(CFG.device)
        model.load_state_dict(torch.load(f'student_best_fold{fold}.pth'))
        model.eval()
        
        fold_test_preds = np.zeros((len(test_df), CFG.num_classes))
        
        for tta_tfm in tta_transforms:
            test_dataset = StructureDataset(test_df, transform=tta_tfm, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
            
            tta_preds = []
            with torch.no_grad():
                for front, top in tqdm(test_loader, desc=f'Test TTA'):
                    front, top = front.to(CFG.device), top.to(CFG.device)
                    logits = model(front, top)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    tta_preds.append(probs)
            
            fold_test_preds += np.vstack(tta_preds) / len(tta_transforms)
            
        test_preds += fold_test_preds / CFG.n_fold
        
        del model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    test_df['unstable_prob'] = test_preds[:, 0]
    test_df['stable_prob'] = test_preds[:, 1]
    test_df[['id', 'unstable_prob', 'stable_prob']].to_csv('final_top3_student_submission.csv', index=False)
    print("학생 모델 학습 및 최종 앙상블 완료! final_top3_student_submission.csv 가 생성되었습니다.")
