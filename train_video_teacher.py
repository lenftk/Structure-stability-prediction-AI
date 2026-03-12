import os
import gc
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import mc3_18, MC3_18_Weights

from sklearn.model_selection import StratifiedKFold

from dataset_video import VideoStructureDataset

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

class VCFG:
    seed = 42
    epochs = 10
    batch_size = 4        # 3D CNN은 메모리를 많이 차지하므로 4 또는 8 사용
    img_size = 224
    num_frames = 16       # 프레임 수
    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 1e-2
    n_fold = 5
    patience = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(VCFG.seed)

BASE_PATH = './open'

train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
dev_df = pd.read_csv(f'{BASE_PATH}/dev.csv')
test_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

train_df['source'] = 'train'
dev_df['source'] = 'dev'
all_train_df = pd.concat([train_df, dev_df], axis=0).reset_index(drop=True)
all_train_df['target'] = all_train_df['label'].map({'unstable': 0, 'stable': 1})

# Model -> Torchvision의 MC3_18 (혼합 컨볼루션 구조로 효율적인 3D 학습)
class VideoModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = MC3_18_Weights.DEFAULT if pretrained else None
        self.backbone = mc3_18(weights=weights)
        
        # 마지막 레이어를 2진 분류기로 변경
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_feats, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_fn(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for videos, targets in tqdm(dataloader, desc='Train', leave=False):
        videos, targets = videos.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if scheduler: scheduler.step()
    return total_loss / len(dataloader)

def valid_fn(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    with torch.no_grad():
        for videos, targets in tqdm(dataloader, desc='Valid', leave=False):
            videos, targets = videos.to(device), targets.to(device)
            logits = model(videos)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
    return total_loss / len(dataloader), np.vstack(preds)


if __name__ == '__main__':
    folds = StratifiedKFold(n_splits=VCFG.n_fold, shuffle=True, random_state=VCFG.seed)
    all_train_df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(all_train_df, all_train_df['target'])):
        all_train_df.loc[val_idx, 'fold'] = fold_idx

    oof_preds = np.zeros((len(all_train_df), 2))
    test_preds = np.zeros((len(test_df), 2))

    for fold in range(VCFG.n_fold):
        print(f"\n[Video Training]========== Fold: {fold} ==========")
        
        trn_data = all_train_df[all_train_df['fold'] != fold].reset_index(drop=True)
        val_data = all_train_df[all_train_df['fold'] == fold].reset_index(drop=True)
        
        trn_dataset = VideoStructureDataset(df=trn_data, mode='train', num_frames=VCFG.num_frames, img_size=VCFG.img_size)
        val_dataset = VideoStructureDataset(df=val_data, mode='valid', num_frames=VCFG.num_frames, img_size=VCFG.img_size)
        
        trn_loader = DataLoader(trn_dataset, batch_size=VCFG.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=VCFG.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        model = VideoModel(pretrained=True).to(VCFG.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=VCFG.lr, weight_decay=VCFG.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VCFG.epochs, eta_min=VCFG.min_lr)
        
        best_loss = np.inf
        patience_cnt = 0
        
        for epoch in range(1, VCFG.epochs + 1):
            trn_loss = train_fn(model, trn_loader, criterion, optimizer, scheduler, VCFG.device)
            val_loss, val_preds = valid_fn(model, val_loader, criterion, VCFG.device)
            
            print(f"Epoch {epoch} | Train Loss {trn_loss:.4f} | Val Loss {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                oof_preds[all_train_df['fold'] == fold] = val_preds
                torch.save(model.state_dict(), f'video_best_fold{fold}.pth')
                patience_cnt = 0
                print(" => Saved Best Video Model!")
            else:
                patience_cnt += 1
                if patience_cnt >= VCFG.patience:
                    print(f"Early Stopping!")
                    break

        del model, trn_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Inferencing Fold {fold} Test Data...")
        test_dataset = VideoStructureDataset(df=test_df, mode='test', num_frames=VCFG.num_frames, img_size=VCFG.img_size)
        test_loader = DataLoader(test_dataset, batch_size=VCFG.batch_size, shuffle=False, num_workers=2)
        
        model = VideoModel(pretrained=False).to(VCFG.device)
        model.load_state_dict(torch.load(f'video_best_fold{fold}.pth'))
        model.eval()
        
        fold_test_preds = []
        with torch.no_grad():
            for videos in tqdm(test_loader, desc=f'Test Fold {fold}'):
                videos = videos.to(VCFG.device)
                logits = model(videos)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                fold_test_preds.append(probs)
                
        test_preds += np.vstack(fold_test_preds) / VCFG.n_fold
        
        del model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Inference 결과를 CSV로 먼저 저장 (나중에 앙상블 합산을 위함)
    test_df['unstable_prob'] = test_preds[:, 0]
    test_df['stable_prob'] = test_preds[:, 1]
    test_df[['id', 'unstable_prob', 'stable_prob']].to_csv('video_submission.csv', index=False)
    print("비디오 전용 학습/추론이 완료되어 video_submission.csv 에 저장되었습니다.")
