import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VideoTeacherDataset(Dataset):
    """
    Dataset for training the Teacher model on `simulation.mp4`
    """
    def __init__(self, csv_path, img_dir, num_frames=16, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.num_frames = num_frames
        self.is_train = is_train
        
        # Define transform for individual frames
        if self.is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)
        
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniformly sample `num_frames` indices
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.zeros(self.num_frames, dtype=int)
            
        for d in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, d)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Fallback if frame read fails
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else: # Empty video completely?
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        # Final safety check to make sure we have exactly num_frames
        while len(frames) < self.num_frames:
             if len(frames) > 0:
                 frames.append(frames[-1].copy())
             else:
                 frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                 
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        
        video_path = os.path.join(self.img_dir, sample_id, 'simulation.mp4')
        frames = self._extract_frames(video_path)
        
        tensor_frames = []
        for frame in frames:
            aug = self.transform(image=frame)
            tensor_frames.append(aug['image'])
            
        # Stack frames: (num_frames, C, H, W)
        # Often video models expect (C, num_frames, H, W) -> we can permute later if needed
        video_tensor = torch.stack(tensor_frames)
        
        # For teacher model training, we ONLY care about train set labels
        label_str = row['label']
        label_idx = 0 if label_str == 'unstable' else 1
        label = torch.tensor(label_idx, dtype=torch.long)
        
        # For distillation we also need the ID to match against student models
        return {
            'id': sample_id,
            'video': video_tensor,  # (T, C, H, W)
            'label': label
        }
