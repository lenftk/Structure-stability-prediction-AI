import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import StructureDataset
from model_advanced import AdvancedStructureModel
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Train Advanced Structure Model with KD")
    parser.add_argument("--data_dir", type=str, default="open", help="Path to open dataset directory")
    parser.add_argument("--teacher_preds", type=str, default="teacher_soft_labels.csv", help="Pseudo-labels from Teacher")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Timm backbone")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD Temperature")
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
        for batch in tqdm(val_loader, desc="Validating"):
            front = batch['front'].to(device)
            top = batch['top'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(front, top)
            probs = F.softmax(logits, dim=1)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    score = logloss(all_targets, all_preds)
    return score

# Custom Dataset wrapper to inject teacher soft labels
class DistillationDataset(StructureDataset):
    def __init__(self, csv_path, img_dir, teacher_csv, is_train=True):
        super().__init__(csv_path, img_dir, is_train)
        
        # Merge Teacher labels
        if is_train and os.path.exists(teacher_csv):
            teacher_df = pd.read_csv(teacher_csv)
            self.df = pd.merge(self.df, teacher_df, on='id', how='left')
            self.has_teacher = True
        else:
            self.has_teacher = False

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        if self.has_teacher:
            row = self.df.iloc[idx]
            # [unstable_prob, stable_prob]
            soft_label = np.array([row['teacher_unstable_prob'], row['teacher_stable_prob']], dtype=np.float32)
            sample['soft_label'] = torch.tensor(soft_label)
            
        return sample

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_img_dir = os.path.join(args.data_dir, "train")
    
    val_csv = os.path.join(args.data_dir, "dev.csv")
    val_img_dir = os.path.join(args.data_dir, "dev")
    
    train_dataset = DistillationDataset(train_csv, train_img_dir, args.teacher_preds, is_train=True)
    val_dataset = StructureDataset(val_csv, val_img_dir, is_train=False) # Validation doesn't need teacher labels
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = AdvancedStructureModel(backbone_name=args.backbone, pretrained=True)
    model.to(args.device)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean') # For distillation
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            front = batch['front'].to(args.device)
            top = batch['top'].to(args.device)
            labels = batch['label'].to(args.device)
            
            optimizer.zero_grad()
            logits = model(front, top)
            
            loss_hard = criterion_ce(logits, labels)
            
            if train_dataset.has_teacher:
                soft_labels = batch['soft_label'].to(args.device)
                # Compute KD loss (KL divergence between student log_softmax and teacher soft probabilities)
                # Teacher soft_labels should already be softmaxed probabilities
                log_p = F.log_softmax(logits / args.temperature, dim=1)
                # We need to target a distribution, so teacher labels should be the target
                # PyTorch KLDiv expects input to be Log-Probs and target to be Probs
                loss_soft = criterion_kl(log_p, soft_labels) * (args.temperature ** 2)
                
                loss = (1.0 - args.alpha) * loss_hard + args.alpha * loss_soft
            else:
                loss = loss_hard
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * front.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        val_score = validate(model, val_loader, args.device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val LogLoss: {val_score:.4f}")
        
        if val_score < best_loss:
            best_loss = val_score
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_student_{args.backbone}.pth"))
            print(f"--> Saved best Student model LogLoss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
