#pip install torch torchvision timm albumentations opencv-python scikit-learn pandas numpy tqdm

import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.utils import ModelEmaV2
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import warnings
import math

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 최고 성능을 위한 하이퍼파라미터 셋팅 (Grandmaster Settings)
# ==============================================================================
class CFG:
    DATA_DIR = './open'               # 데이터셋 경로 (train.csv 위치)
    
    # 💥 압도적인 성능을 내는 사전학습 모델 (ImageNet-22k에서 파인튜닝된 버전)
    # 메모리가 부족(OOM)하다면 'convnext_small_in22ft1k' 로 변경하세요.
    MODEL_NAME = 'convnext_base_in22ft1k' 
    
    IMG_SIZE = 224
    BATCH_SIZE = 8                # SAM 옵티마이저는 메모리를 2배 쓰므로 8로 낮춤 (안정성 확보)
    EPOCHS = 15                   # 15~20 에포크면 충분히 수렴합니다.
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    NUM_FOLDS = 5                 # 5-Fold Cross Validation
    MIXUP_ALPHA = 0.2             # Mixup 강도 (과적합 방지)
    EMA_DECAY = 0.999             # EMA 붕괴율 (Test 환경의 노이즈를 완벽 방어)
    SAM_RHO = 0.05                # SAM의 평탄화 반경 (Loss 공간을 평평하게 만듦)
    SEED = 2026
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(CFG.SEED)

# ==============================================================================
# 2. SAM Optimizer 정의 (도메인 일반화의 핵심 무기)
# ==============================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# ==============================================================================
# 3. 데이터셋 (Train + Dev 병합 & 강건한 증강)
# ==============================================================================
class StructureDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        folder = row['source'] if 'source' in row else 'test'
        
        front_path = os.path.join(CFG.DATA_DIR, folder, img_id, 'front.png')
        top_path = os.path.join(CFG.DATA_DIR, folder, img_id, 'top.png')
        
        front_img = cv2.cvtColor(cv2.imread(front_path), cv2.COLOR_BGR2RGB)
        top_img = cv2.cvtColor(cv2.imread(top_path), cv2.COLOR_BGR2RGB)
        
        if self.transform:
            front_img = self.transform(image=front_img)['image']
            top_img = self.transform(image=top_img)['image']
            
        if self.is_test:
            return front_img, top_img, img_id
        else:
            # stable을 1.0, unstable을 0.0으로 라벨링
            label = 1.0 if row['label'] == 'stable' else 0.0
            return front_img, top_img, torch.tensor(label, dtype=torch.float32)

# 극한의 일반화를 위한 어그멘테이션 (형태 왜곡은 최소화, 색상/노이즈 왜곡은 극대화)
train_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5), # 좌우반전은 물리 안정성에 영향 없음
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==============================================================================
# 4. 모델 설계 (Late Fusion - Cross View Network)
# ==============================================================================
class DualViewNet(nn.Module):
    def __init__(self, model_name=CFG.MODEL_NAME):
        super().__init__()
        # 1. Front View / Top View 각각의 특성을 뽑아냄
        self.backbone_front = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone_top = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
            feat_dim = self.backbone_front(dummy).shape[1]
            
        # 2. Classifier: 병합된 특징을 종합해 무너질지 판단
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1) # BCEWithLogitsLoss 용 1차원 출력
        )
        
    def forward(self, front, top):
        f_feat = self.backbone_front(front)
        t_feat = self.backbone_top(top)
        combined = torch.cat([f_feat, t_feat], dim=1) # 채널 축으로 결합
        return self.classifier(combined).squeeze(1)

# ==============================================================================
# 5. Mixup 함수 (과적합 억제)
# ==============================================================================
def mixup_data(front, top, y, alpha=CFG.MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = front.size()[0]
    index = torch.randperm(batch_size).to(CFG.DEVICE)
    
    mixed_front = lam * front + (1 - lam) * front[index, :]
    mixed_top = lam * top + (1 - lam) * top[index, :]
    y_a, y_b = y, y[index]
    return mixed_front, mixed_top, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==============================================================================
# 6. K-Fold 학습 및 검증 루프 (코어 엔진)
# ==============================================================================
def train_and_evaluate():
    print("Loading Data & Setting up K-Fold...")
    # Train과 Dev 통합 (Dev 환경을 모델에 주입하기 위함)
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'dev.csv'))
    
    train_df['source'] = 'train'
    dev_df['source'] = 'dev'
    all_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    all_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df['label'])):
        all_df.loc[val_idx, 'fold'] = fold

    for fold in range(CFG.NUM_FOLDS):
        print(f"\n======================================")
        print(f"      STARTING FOLD {fold+1}/{CFG.NUM_FOLDS}      ")
        print(f"======================================")
        
        trn_df = all_df[all_df['fold'] != fold].reset_index(drop=True)
        val_df = all_df[all_df['fold'] == fold].reset_index(drop=True)
        
        train_dataset = StructureDataset(trn_df, transform=train_transform)
        val_dataset = StructureDataset(val_df, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = DualViewNet().to(CFG.DEVICE)
        
        # EMA (지수이동평균) 장착: Test 평가 시 노이즈에 극도로 강해짐
        ema_model = ModelEmaV2(model, decay=CFG.EMA_DECAY)
        
        criterion = nn.BCEWithLogitsLoss()
        
        # SAM 옵티마이저 셋팅 (안정성을 위해 AMP-GradScaler 미사용)
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, rho=CFG.SAM_RHO)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)
        
        best_val_loss = float('inf')
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Train]')
            for front, top, labels in pbar:
                front, top, labels = front.to(CFG.DEVICE), top.to(CFG.DEVICE), labels.to(CFG.DEVICE)
                
                # 50% 확률로 믹스업 적용
                use_mixup = np.random.rand() > 0.5
                if use_mixup:
                    front, top, y_a, y_b, lam = mixup_data(front, top, labels)
                
                # ======= SAM 1단계 (Loss 계산 및 Local Maxima 찾기) =======
                outputs = model(front, top)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # ======= SAM 2단계 (가장 안 좋은 방향에서 최적화 수행) =======
                outputs_2 = model(front, top)
                if use_mixup:
                    loss_2 = mixup_criterion(criterion, outputs_2, y_a, y_b, lam)
                else:
                    loss_2 = criterion(outputs_2, labels)
                    
                loss_2.backward()
                optimizer.second_step(zero_grad=True)
                
                # EMA 업데이트
                ema_model.update(model)
                train_loss += loss_2.item()
                pbar.set_postfix({'loss': f'{loss_2.item():.4f}'})
                
            scheduler.step()
            
            # Validation (반드시 EMA 모델로 검증 수행)
            ema_model.module.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for front, top, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Val]'):
                    front, top, labels = front.to(CFG.DEVICE), top.to(CFG.DEVICE), labels.to(CFG.DEVICE)
                    
                    outputs = ema_model.module(front, top)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val (EMA) Loss: {avg_val_loss:.4f}")
            
            # 최고 성능 달성 시 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # 폴드별 최고의 EMA 모델 가중치 저장
                torch.save(ema_model.module.state_dict(), f'best_model_fold{fold}.pth')
                print(f"🔥 [NEW BEST] Model saved at fold {fold} with Val Loss: {best_val_loss:.4f}")
                
    print("\n[SUCCESS] Training Finished for all Folds!")

# ==============================================================================
# 7. Test 추론 (5-Fold 모델 평균 + TTA 적용)
# ==============================================================================
def inference_ensemble():
    print("\n======================================")
    print("   Starting Ultimate Ensemble Inference   ")
    print("======================================")
    
    test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    test_df['source'] = 'test'
    test_dataset = StructureDataset(test_df, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 5개의 Fold 모델이 예측한 확률을 누적할 배열
    ensemble_preds = np.zeros(len(test_df))
    
    for fold in range(CFG.NUM_FOLDS):
        model = DualViewNet().to(CFG.DEVICE)
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        model.eval()
        
        fold_preds =[]
        with torch.no_grad():
            for front, top, _ in tqdm(test_loader, desc=f'Predicting Fold {fold}'):
                front, top = front.to(CFG.DEVICE), top.to(CFG.DEVICE)
                
                # 1. 원본 이미지 예측
                out_orig = torch.sigmoid(model(front, top))
                
                # 2. TTA (좌우 반전 예측)
                front_flip = torch.flip(front, dims=[3])
                top_flip = torch.flip(top, dims=[3])
                out_flip = torch.sigmoid(model(front_flip, top_flip))
                
                # 원본과 TTA 예측값의 평균
                pred = (out_orig + out_flip) / 2.0
                fold_preds.extend(pred.cpu().numpy())
                
        # 5개 폴드 누적 합산 (1/5씩 가중치)
        ensemble_preds += np.array(fold_preds) / float(CFG.NUM_FOLDS)
        
    # 평가 산식(LogLoss) 에러 방지를 위한 확률값 클리핑 (매우 중요)
    eps = 1e-15
    stable_probs = np.clip(ensemble_preds, eps, 1 - eps)
    unstable_probs = 1.0 - stable_probs
    
    # 제출 양식에 맞게 입력 (제출 파일은 반드시 id, unstable_prob, stable_prob 순서)
    submission = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    submission['unstable_prob'] = unstable_probs
    submission['stable_prob'] = stable_probs
    submission.to_csv('submission_grandmaster.csv', index=False)
    
    print("\n🚀 [FINAL] Inference Complete! File saved as 'submission_grandmaster.csv'")
    print("제출 탭에 'submission_grandmaster.csv' 파일을 제출하세요. 좋은 결과 응원합니다!")

if __name__ == '__main__':
    # 1. 학습 실행
    train_and_evaluate()
    # 2. 추론 및 제출파일 생성 실행
    inference_ensemble()