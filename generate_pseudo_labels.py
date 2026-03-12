import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_video import VideoTeacherDataset
import timm
from tqdm import tqdm

class VideoTeacherModel(nn.Module):
    def __init__(self, backbone_name='timesformer_base_patch16_224', pretrained=True, num_classes=2):
        super().__init__()
        # Use TimeSformer or VideoMAE from timm for video backbones
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        # x expected shape usually (B, C, T, H, W) for somewhat standard video models
        # or (B, T, C, H, W) for others depending on timm wrapper
        # TimeSformer expects (B, T, C, H, W) mostly, let's assume it accepts our stacked frames
        return self.backbone(x)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="open")
    parser.add_argument("--weights_path", type=str, required=True, help="Teacher weights")
    parser.add_argument("--output_file", type=str, default="teacher_soft_labels.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = get_args()
    
    # We generate soft labels for the train set
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_img_dir = os.path.join(args.data_dir, "train")
    
    dataset = VideoTeacherDataset(train_csv, train_img_dir, num_frames=16, is_train=False) # is_train=False to disable augmentations for pure extraction
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = VideoTeacherModel()
    model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Teacher Soft Labels"):
            sample_ids = batch['id']
            video = batch['video'].to(args.device)
            
            logits = model(video)
            # Softmax with temperature scaling to soften probability distribution
            T = 2.0
            probs = F.softmax(logits / T, dim=1).cpu().numpy()
            
            for i in range(len(sample_ids)):
                results.append({
                    'id': sample_ids[i],
                    'teacher_unstable_prob': probs[i][0],
                    'teacher_stable_prob': probs[i][1]
                })
                
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Teacher soft labels saved to {args.output_file}")

if __name__ == "__main__":
    main()
