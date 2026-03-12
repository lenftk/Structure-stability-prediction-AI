import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StructureDataset
from model import StructureStabilityModel
import torch.nn.functional as F
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Train Structure Stability Model")
    parser.add_argument("--data_dir", type=str, default="open", help="Path to open dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Model backbone")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def logloss(true, pred, eps=1e-15):
    # Dacon's provided metric
    # true: (N,) indices or (N, 2) one-hot. Here it's expected to be one-hot for the formula.
    # pred: (N, 2) probabilities
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / np.sum(pred, axis=1).reshape(-1, 1)
    
    if len(true.shape) == 1:
        # Convert indices to one-hot structure for the formula
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

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    
    # Paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_img_dir = os.path.join(args.data_dir, "train")
    
    val_csv = os.path.join(args.data_dir, "dev.csv")
    val_img_dir = os.path.join(args.data_dir, "dev")
    
    # Datasets and Loaders
    train_dataset = StructureDataset(train_csv, train_img_dir, is_train=True)
    val_dataset = StructureDataset(val_csv, val_img_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = StructureStabilityModel(backbone_name=args.backbone, pretrained=True)
    model.to(args.device)
    
    # Loss and Optimizer
    # CrossEntropyLoss expects target class indices (0 or 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            front = batch['front'].to(args.device)
            top = batch['top'].to(args.device)
            labels = batch['label'].to(args.device)
            
            optimizer.zero_grad()
            logits = model(front, top)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * front.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        # Validate
        val_score = validate(model, val_loader, args.device)
        
        scheduler.step()
        
        print(f"Train CrossEntropyLoss: {train_loss:.4f} | Val LogLoss: {val_score:.4f}")
        
        # Save best model
        if val_score < best_loss:
            best_loss = val_score
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_{args.backbone}.pth"))
            print(f"--> Saved best model with LogLoss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
