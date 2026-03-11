import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights, swin3d_t, Swin3D_T_Weights
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
class CFG:
    SEED = 42
    NUM_FRAMES = 16              
    IMG_SIZE = 224               
    BATCH_SIZE = 8               
    EPOCHS = 10
    LR = 5e-5                    
    WEIGHT_DECAY = 1e-4
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
class VideoStructureDataset(Dataset):
    def __init__(self, df, img_dir='train'):
        self.df = df
        self.img_dir = img_dir
    def __len__(self):
        return len(self.df)
    def extract_frames(self, video_path, num_frames=CFG.NUM_FRAMES):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return np.zeros((num_frames, CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.float32)
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (CFG.IMG_SIZE, CFG.IMG_SIZE))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            frames.append(frame)
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.float32))
        frames = np.array(frames) 
        frames = frames.transpose(3, 0, 1, 2)
        return frames
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_id = str(row['id'])
        video_path = os.path.join(self.img_dir, folder_id, 'simulation.mp4')
        video_tensor = self.extract_frames(video_path)
        video_tensor = torch.tensor(video_tensor, dtype=torch.float32)
        label = 1.0 if row['label'] == 'unstable' else 0.0
        return video_tensor, torch.tensor(label, dtype=torch.float32)
class VideoTeacherNet(nn.Module):
    def __init__(self):
        super(VideoTeacherNet, self).__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        feat = self.backbone(x) 
        logits = self.classifier(feat)
        return logits.squeeze(-1)
def train_teacher_model(fold, train_loader, val_loader, device):
    print(f"\n========== Teacher Fold {fold} Training ==========")
    model = VideoTeacherNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss().to(device) 
    best_val_loss = float('inf')
    best_weights = None
    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        train_loss = []
        train_corrects = 0
        train_total = 0
        for video, labels in tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=False):
            video = video.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(video)
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
            for video, labels in tqdm(val_loader, desc=f'Epoch {epoch} Valid', leave=False):
                video = video.to(device)
                labels = labels.to(device)
                logits = model(video)
                loss = criterion(logits, labels)
                val_loss.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
        _val_loss = np.mean(val_loss)
        _val_acc = val_corrects / val_total
        print(f"Epoch[{epoch}/{CFG.EPOCHS}] Train Loss: {_train_loss:.4f} Acc: {_train_acc:.4f} | Val Loss: {_val_loss:.4f} Acc: {_val_acc:.4f}")
        if _val_loss < best_val_loss:
            best_val_loss = _val_loss
            best_weights = model.state_dict().copy()
            torch.save(best_weights, f'teacher_model_fold{fold}.pth')
            print(f"  --> Best Teacher Model Saved (Val Loss: {best_val_loss:.4f})")
    return best_weights
if __name__ == '__main__':
    train_df_all = pd.read_csv('open/train.csv')
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    targets = (train_df_all['label'] == 'unstable').astype(int)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df_all, targets), 1):
        fold_train = train_df_all.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df_all.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = VideoStructureDataset(fold_train, img_dir='open/train')
        val_dataset = VideoStructureDataset(fold_val, img_dir='open/train')
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        train_teacher_model(fold, train_loader, val_loader, CFG.DEVICE)
    print("Teacher 모델 K-Fold 학습이 완료되었습니다. (추후 Knowledge Distillation에 활용)")
