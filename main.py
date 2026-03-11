import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
class CFG:
    SEED = 42
    IMG_SIZE = 384               
    BATCH_SIZE = 16              
    EPOCHS = 15
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k_384' 
    NUM_WORKERS = 4              
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_FOLDS = 5                  
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG.SEED)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
train_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=int(CFG.IMG_SIZE*0.1), max_width=int(CFG.IMG_SIZE*0.1), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
class StructureDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_id = str(row['id'])
        img_dir = row.get('img_dir', 'test')
        front_path = os.path.join(img_dir, folder_id, 'front.png')
        top_path = os.path.join(img_dir, folder_id, 'top.png')
        front_img = cv2.cvtColor(cv2.imread(front_path), cv2.COLOR_BGR2RGB)
        top_img = cv2.cvtColor(cv2.imread(top_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            front_img = self.transform(image=front_img)['image']
            top_img = self.transform(image=top_img)['image']
        if self.is_test:
            return front_img, top_img, folder_id
        else:
            label = 1.0 if row['label'] == 'unstable' else 0.0
            return front_img, top_img, torch.tensor(label, dtype=torch.float32)
class DualViewNet(nn.Module):
    def __init__(self, model_name=CFG.MODEL_NAME, pretrained=True):
        super(DualViewNet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )
    def forward(self, front_img, top_img):
        front_feat = self.backbone(front_img)
        top_feat = self.backbone(top_img)
        fused_feat = torch.cat([front_feat, top_feat], dim=1)
        logits = self.classifier(fused_feat)
        return logits.squeeze(-1)
def train_one_fold(fold, train_loader, val_loader, device):
    print(f"\n========== Fold {fold} Training ==========")
    model = DualViewNet(model_name=CFG.MODEL_NAME, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss().to(device)
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        train_loss = []
        train_corrects = 0
        train_total = 0
        for front_img, top_img, labels in tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=False):
            front_img = front_img.to(device)
            top_img = top_img.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(front_img, top_img)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_corrects += (preds == labels).sum().item()
            train_total += labels.size(0)
        _train_loss = np.mean(train_loss)
        _train_acc = train_corrects / train_total
        model.eval()
        val_loss = []
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for front_img, top_img, labels in tqdm(val_loader, desc=f'Epoch {epoch} Valid', leave=False):
                front_img = front_img.to(device)
                top_img = top_img.to(device)
                labels = labels.to(device)
                logits = model(front_img, top_img)
                loss = criterion(logits, labels)
                val_loss.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
        _val_loss = np.mean(val_loss)
        _val_acc = val_corrects / val_total
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch[{epoch}/{CFG.EPOCHS}] Train Loss: {_train_loss:.4f} Acc: {_train_acc:.4f} | Val Loss: {_val_loss:.4f} Acc: {_val_acc:.4f}")
        if _val_loss < best_val_loss:
            best_val_loss = _val_loss
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, f'best_model_fold{fold}.pth')
            print(f"  --> ⭐ Best Model Saved! (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(best_model_weights)
    return model
def inference_ensemble(models, test_loader, device):
    for m in models:
        m.eval()
    ensemble_probs = []
    with torch.no_grad():
        for front_img, top_img, _ in tqdm(test_loader, desc='Ensemble Inference'):
            front_img = front_img.to(device)
            top_img = top_img.to(device)
            fold_probs = []
            for model in models:
                logits = model(front_img, top_img)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.append(probs)
            avg_probs = np.mean(fold_probs, axis=0)
            ensemble_probs.extend(avg_probs)
    return ensemble_probs
if __name__ == '__main__':
    print(f"Using Device: {CFG.DEVICE}")
    train_df = pd.read_csv('train.csv')  
    dev_df = pd.read_csv('dev.csv')
    test_df = pd.read_csv('test.csv')
    submit = pd.read_csv('sample_submission.csv')
    train_df['img_dir'] = 'train'
    dev_df['img_dir'] = 'dev'
    test_df['img_dir'] = 'test'
    merged_df = pd.concat([train_df, dev_df], axis=0).reset_index(drop=True)
    test_dataset = StructureDataset(test_df, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    trained_models = []
    print("🚀 Start K-Fold Training...")
    targets = (merged_df['label'] == 'unstable').astype(int)
    for fold, (train_idx, val_idx) in enumerate(skf.split(merged_df, targets), 1):
        fold_train_df = merged_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = merged_df.iloc[val_idx].reset_index(drop=True)
        train_dataset = StructureDataset(fold_train_df, transform=train_transform, is_test=False)
        val_dataset = StructureDataset(fold_val_df, transform=test_transform, is_test=False)
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        best_model = train_one_fold(fold, train_loader, val_loader, CFG.DEVICE)
        trained_models.append(best_model)
    print("Start Ensemble Inference...")
    preds_unstable = inference_ensemble(trained_models, test_loader, CFG.DEVICE)
    submit['unstable_prob'] = preds_unstable
    submit['stable_prob'] = 1.0 - np.array(preds_unstable)
    submit.to_csv('submission.csv', index=False)
    print("submission.csv Saved Successfully!")