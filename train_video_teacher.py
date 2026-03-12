import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_video import VideoTeacherDataset
from generate_pseudo_labels import VideoTeacherModel
import torch.nn.functional as F
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="open")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="timesformer_base_patch16_224")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / np.sum(pred, axis=1).reshape(-1, 1)
    if len(true.shape) == 1:
        true_onehot = np.zeros_like(pred)
        true_onehot[np.arange(len(true)), true] = 1.0
        true = true_onehot
    loss = -np.sum(true * np.log(pred), axis=1)
    return np.mean(loss)

def validate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating Video Teacher"):
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            logits = model(video)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return logloss(all_targets, all_preds)

def train_teacher(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_img_dir = os.path.join(args.data_dir, "train")
    val_csv = os.path.join(args.data_dir, "dev.csv")
    val_img_dir = os.path.join(args.data_dir, "dev")
    
    train_dataset = VideoTeacherDataset(train_csv, train_img_dir, is_train=True)
    val_dataset = VideoTeacherDataset(val_csv, val_img_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = VideoTeacherModel(backbone_name=args.backbone, pretrained=True)
    model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, f"best_teacher_{args.backbone}.pth")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Teacher] Epoch {epoch}/{args.epochs}")
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training Teacher"):
            video = batch['video'].to(args.device)
            labels = batch['label'].to(args.device)
            
            optimizer.zero_grad()
            logits = model(video)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * video.size(0)
            
        train_loss = train_loss / len(train_dataset)
        val_score = validate(model, val_loader, args.device)
        scheduler.step()
        
        print(f"Teacher Train CE Loss: {train_loss:.4f} | Val LogLoss: {val_score:.4f}")
        
        if val_score < best_loss:
            best_loss = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved best Teacher model (LogLoss: {best_loss:.4f})")
            
    return best_model_path

if __name__ == "__main__":
    train_teacher(get_args())
