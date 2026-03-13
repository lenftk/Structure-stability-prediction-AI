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
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

class CFG:
    DATA_DIR = './open'
    MODEL_NAME = 'convnext_base_in22ft1k' 
    IMG_SIZE = 288
    BATCH_SIZE = 32
    SEED = 2026
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 💡 가지고 계신 4개의 가중치 파일 이름을 정확히 입력해주세요.
    # 만약 fold 0~3까지 저장되었다면 아래 파일명이 맞을 것입니다.
    WEIGHT_FILES =[
        'best_sub_convnext_base_model_fold0.pth',
        'best_sub_convnext_base_model_fold1.pth',
        'best_sub_convnext_base_model_fold2.pth',
        'best_sub_convnext_base_model_fold3.pth',
        'best_sub_convnext_base_model_fold4.pth'
    ]

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
seed_everything(CFG.SEED)

val_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class StructureDataset(Dataset):
    def __init__(self, df, is_test=True):
        self.df = df
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        folder = 'test'
        
        front_img = cv2.cvtColor(cv2.imread(os.path.join(CFG.DATA_DIR, folder, img_id, 'front.png')), cv2.COLOR_BGR2RGB)
        top_img = cv2.cvtColor(cv2.imread(os.path.join(CFG.DATA_DIR, folder, img_id, 'top.png')), cv2.COLOR_BGR2RGB)
        
        front_img = val_transform(image=front_img)['image']
        top_img = val_transform(image=top_img)['image']
            
        return front_img, top_img, img_id

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

def inference_ensemble():
    test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    test_dataset = StructureDataset(test_df, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, 
                             num_workers=2, pin_memory=True)
    
    ensemble_preds = np.zeros(len(test_df))
    num_models = len(CFG.WEIGHT_FILES)
    
    print(f"총 {num_models}개의 모델로 앙상블 추론을 시작합니다...")
    
    for idx, weight_file in enumerate(CFG.WEIGHT_FILES):
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_file}")
            
        model = RobustFusionNet().to(CFG.DEVICE)
        model.load_state_dict(torch.load(weight_file, map_location=CFG.DEVICE))
        model.eval()
        
        fold_preds =[]
        with torch.no_grad():
            for front, top, _ in tqdm(test_loader, desc=f'Predicting Model {idx+1}/{num_models} ({weight_file})'):
                front, top = front.to(CFG.DEVICE, non_blocking=True), top.to(CFG.DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    # TTA (Test Time Augmentation)
                    o1 = torch.sigmoid(model(front, top))
                    o2 = torch.sigmoid(model(torch.flip(front, [3]), torch.flip(top, [3])))
                    o3 = torch.sigmoid(model(front, torch.flip(top, [2])))
                    
                    front_s = F.interpolate(front, scale_factor=1.1, mode='bilinear', align_corners=False)
                    top_s = F.interpolate(top, scale_factor=1.1, mode='bilinear', align_corners=False)
                    start = int((CFG.IMG_SIZE * 1.1 - CFG.IMG_SIZE) / 2)
                    end = start + CFG.IMG_SIZE
                    o4 = torch.sigmoid(model(front_s[:, :, start:end, start:end], top_s[:, :, start:end, start:end]))
                    
                pred = (o1 + o2 + o3 + o4) / 4.0
                fold_preds.extend(pred.cpu().numpy())
                
        # 4개의 모델이므로 전체 값에 대해 1/4씩 더함
        ensemble_preds += np.array(fold_preds) / float(num_models)
        
    stable_probs = np.clip(ensemble_preds, 1e-15, 1 - 1e-15)
    
    # CSV 저장
    submission = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    submission['unstable_prob'] = 1.0 - stable_probs
    submission['stable_prob'] = stable_probs
    
    csv_name = 'sub_convnext_base.csv'
    submission.to_csv(csv_name, index=False)
    print(f"\n🚀 [FINAL] 성공적으로 저장되었습니다: '{csv_name}'")

if __name__ == '__main__':
    inference_ensemble()