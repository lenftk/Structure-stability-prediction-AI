import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Video Teacher 파이프라인의 모듈/데이터셋 가져오기
from train_video_teacher import VideoTeacherNet, VideoStructureDataset, CFG

def extract_soft_labels():
    print("🚀 Teacher 모델로부터 Soft Label 추출을 시작합니다.")
    device = CFG.DEVICE
    
    # 추출 대상 데이터프레임 병합 (train + dev)
    train_df = pd.read_csv('open/train.csv')
    dev_df = pd.read_csv('open/dev.csv')
    
    # dev셋에는 비디오가 없지만, 'open/dev' 폴더에 영상이 주어졌는지 확인 필요
    # 제공 데이터스펙에 따르면 dev에는 `simulation.mp4`가 없는 경우가 많음.
    # 만약 dev.csv에도 비디오가 있다면 아래 주석을 풀고 함께 합치세요.
    # train_df['img_dir'] = 'open/train'
    # dev_df['img_dir'] = 'open/dev'
    # merged_df = pd.concat([train_df, dev_df], axis=0).reset_index(drop=True)
    
    # 안전하게 비디오가 확실히 존재하는 train셋만 사용하여 지식 증류(Soft Label) 우선 생성
    train_df['img_dir'] = 'open/train'
    target_df = train_df.copy()
    
    # K-Fold로 학습된 5개의 Teacher 모델 경로 확인
    teacher_models = []
    for fold in range(1, CFG.N_FOLDS + 1):
        weight_path = f'teacher_model_fold{fold}.pth'
        if not os.path.exists(weight_path):
            print(f"❌ 가중치 파일이 없습니다: {weight_path}")
            print("👉 먼저 'python train_video_teacher.py' 를 실행해 Teacher 모델을 모두 학습시켜주세요!")
            return
            
        model = VideoTeacherNet().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        teacher_models.append(model)
        
    # Dataset 로더 구성 (Batch Size는 VRAM에 맞춰 조절 가능)
    dataset = VideoStructureDataset(target_df, img_dir='open/train')
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
    
    print(f"✅ 총 {len(teacher_models)}개의 Teacher 모델 로드 완료. 앙상블 Soft Label 추론 중...")
    
    all_soft_labels = []
    
    with torch.no_grad():
        for video, _ in tqdm(loader, desc='Extracting Soft Labels'):
            video = video.to(device)
            fold_probs = []
            
            for model in teacher_models:
                logits = model(video)
                # Sigmoid를 통해 0~1 사이의 붕괴 확률 획득
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.append(probs)
                
            # 5개 모델의 평균을 Soft Label (정답)로 사용
            avg_probs = np.mean(fold_probs, axis=0)
            all_soft_labels.extend(avg_probs)
            
    # 원본 데이터프레임에 soft_unstable_prob 라는 이름으로 저장
    target_df['soft_unstable_prob'] = all_soft_labels
    
    # CSV 저장
    save_path = 'teacher_soft_labels.csv'
    target_df.to_csv(save_path, index=False)
    print(f"🎉 성공적으로 추출 완료되어 파일이 저장되었습니다: {save_path}")
    print("👉 이제 이 soft_labels.csv를 활용해 main.py(Student 모델)를 고도화할 차례입니다!")

if __name__ == '__main__':
    extract_soft_labels()
