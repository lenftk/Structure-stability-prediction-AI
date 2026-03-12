import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class StructureDataset(Dataset):
    def __init__(self, csv_path, img_dir, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.is_train = is_train
        
        # Test sets don't have labels
        self.has_labels = 'label' in self.df.columns
        
        # Define transform lengths
        # In train data, camera/lighting is fixed. 
        # Dev and test environments have random lighting/cameras.
        # Strong augmentations are needed to generalize.
        if self.is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5), # We might not want to flip 'top' and 'front' identically unless we are careful, 
                                         # but generally structure stability is symmetric
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=224, width=224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        
        # Load front and top images
        front_img_path = os.path.join(self.img_dir, sample_id, 'front.png')
        top_img_path = os.path.join(self.img_dir, sample_id, 'top.png')
        
        front_img = cv2.imread(front_img_path)
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        
        top_img = cv2.imread(top_img_path)
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms independently or jointly?
        # A.Compose with default args applies independently. 
        # For structure, independent is probably fine or we can concatenate and apply.
        if self.transform:
            front_aug = self.transform(image=front_img)
            front_tensor = front_aug['image']
            
            top_aug = self.transform(image=top_img)
            top_tensor = top_aug['image']
        
        sample = {
            'id': sample_id,
            'front': front_tensor,
            'top': top_tensor
        }
        
        if self.has_labels:
            label_str = row['label']
            # Target output order based on rules: unstable_prob, stable_prob
            # Let's map Unstable -> [1.0, 0.0] and Stable -> [0.0, 1.0]
            # PyTorch CrossEntropyLoss expects class index. 
            # 0 -> unstable, 1 -> stable
            label_idx = 0 if label_str == 'unstable' else 1
            sample['label'] = torch.tensor(label_idx, dtype=torch.long)
            
        return sample
