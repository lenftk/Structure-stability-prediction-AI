import pandas as pd
import numpy as np

# 임계값 설정 (이 확률 이상일 때만 확실한 정답으로 간주)
THRESHOLD = 0.98

def main():
    # 앞서 학습한 모델들의 앙상블 결과 (가정: ensemble_submission.csv 에 저장됨)
    # 실제로는 Image Model, Video Model, TTA 결괏값을 평균 낸 가장 높은 점수의 CSV를 입력으로 씁니다.
    sub_path = 'top3_submission.csv' # 예시로 top3_submission.csv 파일 경로 사용
    
    try:
        sub_df = pd.read_csv(sub_path)
    except FileNotFoundError:
        print(f"[{sub_path}] 파일이 없습니다. 먼저 모델들을 학습하고 예측 CSV를 생성해야 합니다.")
        return
        
    print(f"Total Test Samples: {len(sub_df)}")
    
    # Pseudo-Label링 기준 통과 샘플 추출
    # unstable 확률이 높거나 stable 확률이 매우 높은 경우
    is_unstable = sub_df['unstable_prob'] > THRESHOLD
    is_stable = sub_df['stable_prob'] > THRESHOLD
    
    confident_samples = sub_df[is_unstable | is_stable].copy()
    print(f"Confident Samples (Threshold: {THRESHOLD}): {len(confident_samples)}")
    
    if len(confident_samples) == 0:
        print("확신을 가진 샘플이 없습니다. 임계값을 조절해 보세요.")
        return
        
    # 새로운 Label 컬럼 생성
    def get_label(row):
        return 'unstable' if row['unstable_prob'] > THRESHOLD else 'stable'
        
    confident_samples['label'] = confident_samples.apply(get_label, axis=1)
    
    # Train 데이터셋과 동일한 포맷으로 맞춤 (source 컬럼 추가하여 test 데이터임을 명시)
    pseudo_df = confident_samples[['id', 'label']].copy()
    pseudo_df['source'] = 'test'
    
    # csv 저장
    pseudo_df.to_csv('pseudo_labels.csv', index=False)
    print("pseudo_labels.csv 가 생성되었습니다. 이 파일을 기존 train 데이터와 합쳐서 모델을 재학습(Student Model) 시키세요!")

if __name__ == '__main__':
    main()
