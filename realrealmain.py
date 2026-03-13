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

warnings.filterwarnings('ignore')

class CFG:
    DATA_DIR = './open'
    MODEL_NAME = 'convnext_small_in22ft1k' #convnext_small_in22ft1k!, tf_efficientnetv2_s.in21k_ft_in1k!, convnext_base_in22ft1k??, swinv2_tiny_window16_256!, maxvit_tiny_tf_224.in1k
    IMG_SIZE = 288
    BATCH_SIZE = 32
    EPOCHS = 60
    MAX_LR = 2e-4
    WEIGHT_DECAY = 1e-2
    NUM_FOLDS = 5
    EMA_DECAY = 0.999
    SAM_RHO = 0.05
    SEED = 2026
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
seed_everything(CFG.SEED)

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
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)

front_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5), 
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.3), 
    A.CoarseDropout(max_holes=4, max_height=16, max_width=16, fill_value=0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

top_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class StructureDataset(Dataset):
    def __init__(self, df, is_train=True, is_test=False):
        self.df = df
        self.is_train = is_train
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        folder = row['source'] if 'source' in row else 'test'
        
        front_img = cv2.cvtColor(cv2.imread(os.path.join(CFG.DATA_DIR, folder, img_id, 'front.png')), cv2.COLOR_BGR2RGB)
        top_img = cv2.cvtColor(cv2.imread(os.path.join(CFG.DATA_DIR, folder, img_id, 'top.png')), cv2.COLOR_BGR2RGB)
        
        if self.is_train:
            front_img = front_transform(image=front_img)['image']
            top_img = top_transform(image=top_img)['image']
        else:
            front_img = val_transform(image=front_img)['image']
            top_img = val_transform(image=top_img)['image']
            
        if self.is_test: return front_img, top_img, img_id
        else:
            label = 1.0 if row['label'] == 'stable' else 0.0
            return front_img, top_img, torch.tensor(label, dtype=torch.float32)

class RobustFusionNet(nn.Module):
    def __init__(self, model_name=CFG.MODEL_NAME):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
            feat_dim = self.backbone(dummy).shape[1]
            
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 3, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
    def forward(self, front, top):
        f_feat = self.backbone(front)
        t_feat = self.backbone(top)
        
        interaction = f_feat * t_feat
        
        combined = torch.cat([f_feat, t_feat, interaction], dim=1)
        return self.classifier(combined).squeeze(1)

def train_and_evaluate():
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'train_pseudo.csv'))
    dev_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'dev.csv'))
    
    train_df['source'] = train_df['id'].apply(lambda x: str(x).split('_')[0].lower())
    dev_df['source'] = 'dev'
    all_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    all_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df['label'])):
        all_df.loc[val_idx, 'fold'] = fold

    scaler = torch.cuda.amp.GradScaler()

    for fold in range(CFG.NUM_FOLDS):
        print(f"\n========== STARTING FOLD {fold+1}/{CFG.NUM_FOLDS} ==========")
        
        trn_df = all_df[all_df['fold'] != fold].reset_index(drop=True)
        val_df = all_df[all_df['fold'] == fold].reset_index(drop=True)
        
        train_loader = DataLoader(StructureDataset(trn_df, is_train=True), batch_size=CFG.BATCH_SIZE, shuffle=True, 
                                  num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
        val_loader = DataLoader(StructureDataset(val_df, is_train=False), batch_size=CFG.BATCH_SIZE, shuffle=False, 
                                num_workers=2, pin_memory=True, persistent_workers=True)
        
        model = RobustFusionNet().to(CFG.DEVICE)
        ema_model = ModelEmaV2(model, decay=CFG.EMA_DECAY)
        
        criterion = nn.BCEWithLogitsLoss()
        
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=CFG.MAX_LR, weight_decay=CFG.WEIGHT_DECAY, rho=CFG.SAM_RHO)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer.base_optimizer, max_lr=CFG.MAX_LR, epochs=CFG.EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Train]')
            
            for front, top, labels in pbar:
                front, top, labels = front.to(CFG.DEVICE, non_blocking=True), top.to(CFG.DEVICE, non_blocking=True), labels.to(CFG.DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(front, top)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                with torch.cuda.amp.autocast():
                    outputs_2 = model(front, top)
                    loss_2 = criterion(outputs_2, labels)
                
                loss_2.backward()
                optimizer.second_step(zero_grad=True)
                
                scheduler.step()
                ema_model.update(model)
                train_loss += loss_2.item()
                pbar.set_postfix({'loss': f'{loss_2.item():.4f}'})
                
            ema_model.module.eval()
            val_loss = 0.0
            with torch.no_grad():
                for front, top, labels in val_loader:
                    front, top, labels = front.to(CFG.DEVICE, non_blocking=True), top.to(CFG.DEVICE, non_blocking=True), labels.to(CFG.DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        outputs = ema_model.module(front, top)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(ema_model.module.state_dict(), f'best_model_fold{fold}.pth')
                print(f"🔥 [NEW BEST] Fold {fold} Saved! Val Loss: {best_val_loss:.4f}")

def inference_ensemble():
    test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    test_df['source'] = 'test'
    test_dataset = StructureDataset(test_df, is_train=False, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, 
                             num_workers=2, pin_memory=True)
    
    ensemble_preds = np.zeros(len(test_df))
    
    for fold in range(CFG.NUM_FOLDS):
        model = RobustFusionNet().to(CFG.DEVICE)
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        model.eval()
        
        fold_preds =[]
        with torch.no_grad():
            for front, top, _ in tqdm(test_loader, desc=f'Predicting Fold {fold}'):
                front, top = front.to(CFG.DEVICE, non_blocking=True), top.to(CFG.DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    o1 = torch.sigmoid(model(front, top))
                    o2 = torch.sigmoid(model(torch.flip(front, [3]), torch.flip(top,[3])))
                    o3 = torch.sigmoid(model(front, torch.flip(top,[2])))
                    
                    front_s = F.interpolate(front, scale_factor=1.1, mode='bilinear')
                    top_s = F.interpolate(top, scale_factor=1.1, mode='bilinear')
                    start = int((CFG.IMG_SIZE * 1.1 - CFG.IMG_SIZE) / 2)
                    end = start + CFG.IMG_SIZE
                    o4 = torch.sigmoid(model(front_s[:,:,start:end,start:end], top_s[:,:,start:end,start:end]))
                    
                pred = (o1 + o2 + o3 + o4) / 4.0
                fold_preds.extend(pred.cpu().numpy())
                
        ensemble_preds += np.array(fold_preds) / float(CFG.NUM_FOLDS)
        
    stable_probs = np.clip(ensemble_preds, 1e-15, 1 - 1e-15)
    submission = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    submission['unstable_prob'] = 1.0 - stable_probs
    submission['stable_prob'] = stable_probs
    submission.to_csv('submission_v7_baseline.csv', index=False)
    print("\n🚀 [FINAL] Saved as 'submission_v7_baseline.csv'")

if __name__ == '__main__':
    train_and_evaluate()
    inference_ensemble()