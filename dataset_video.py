import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VideoStructureDataset(Dataset):
    """
    10초 분량의 MP4 비디오에서 N개의 프레임을 균등하게 추출하여 3D CNN 입력 포맷(C, T, H, W)으로 반환하는 데이터셋
    """
    def __init__(self, df, base_path='./open', num_frames=16, img_size=224, mode='train'):
        super().__init__()
        self.df = df
        self.base_path = base_path
        self.num_frames = num_frames
        self.img_size = img_size
        self.mode = mode
        
        # 이미지 증강 대신 간단한 Spatial 증강 사용 (시간적 일관성 유지가 필요하므로 주의)
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        if total_frames > 0:
            # 전체 영상 길이에서 num_frames 개수만큼 균등하게 인덱스 추출
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # 읽기 실패 시 검은 화면 추가
                    frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        else:
            # 비디오가 깨진 경우 예외 처리
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) for _ in range(self.num_frames)]
            
        cap.release()
        
        # 부족한 프레임은 마지막 프레임으로 복제하여 채움
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            
        # 초과된 프레임이 있다면 자름
        frames = frames[:self.num_frames]
        
        # 각 프레임에 동일한 transform 적용 (Spatial 차원 전처리)
        # Augmentation 시 seed 고정이나 동일 파라미터 적용이 중요할 수 있으나,
        # 단순 Resize / Normalize / Flip 은 파이프라인에서 직관적으로 반복 적용해도 무방합니다.
        # 주의: RandomFlip 같은 통계적 증강은 프레임별로 다르게 먹히면 안 되므로, 
        # Albumentations의 ReplayCompose를 쓰거나 고정된 변환만 사용합니다.
        
        # 시간적 일관성을 위해 파이썬 Random 등을 고정하는 트릭
        # 여기서는 가장 기초적인 차원에서 정적 transform만 적용 (HorizFlip 같은 건 일관성 없어도 물리 판단에 큰 영향을 주지 않도록 p=0 처리해도 됩니다. 본 예시에서는 p=0.5로 두었으나 실 시드는 프레임 단위로 다를 수 있습니다.)
        # 더 엄격하게 하려면 albumentations Additional Targets 방식을 씁니다.
        
        # 엄격한 일관성을 위한 추가 타겟 세팅 (Albumentation 문서 참고)
        transform_dict = {f'image{i}': 'image' for i in range(1, self.num_frames)}
        consistent_transform = A.Compose(self.transform.transforms, additional_targets=transform_dict)
        
        inputs = {'image': frames[0]}
        for i in range(1, self.num_frames):
            inputs[f'image{i}'] = frames[i]
            
        res = consistent_transform(**inputs)
        
        proc_frames = [res['image']]
        for i in range(1, self.num_frames):
            proc_frames.append(res[f'image{i}'])
            
        # (T, C, H, W) -> 3D CNN 용 (C, T, H, W)
        video_tensor = torch.stack(proc_frames) # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3) # (C, T, H, W)
        
        return video_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_id = row['id']
        
        if self.mode == 'test':
            vid_path = os.path.join(self.base_path, 'test', vid_id, 'simulation.mp4')
            video_tensor = self.extract_frames(vid_path)
            return video_tensor
        else:
            source = row.get('source', 'train') # 기본값 train (dev 병합본은 source 컬럼 존재)
            vid_path = os.path.join(self.base_path, source, vid_id, 'simulation.mp4')
            video_tensor = self.extract_frames(vid_path)
            
            target = row['target'] # {unstable: 0, stable: 1} 등 매핑된 정수
            return video_tensor, torch.tensor(target, dtype=torch.long)
