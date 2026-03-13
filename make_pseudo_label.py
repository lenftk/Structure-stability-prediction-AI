import pandas as pd

# 1. 방금 만든 1차 앙상블 파일 불러오기
best_sub = pd.read_csv('./submission_ensemble_best.csv') 

# 2. 100% 확신하는 조건(Threshold) 설정
UPPER_BOUND = 0.98  # 안정 확률 98% 이상
LOWER_BOUND = 0.02  # 안정 확률 2% 이하 (즉, 불안정 98% 이상)

# 3. 데이터 필터링 (확실한 놈들만 쏙 빼오기)
confident_stable = best_sub[best_sub['stable_prob'] >= UPPER_BOUND].copy()
confident_unstable = best_sub[best_sub['stable_prob'] <= LOWER_BOUND].copy()

# 라벨 달아주기
confident_stable['label'] = 'stable'
confident_unstable['label'] = 'unstable'

# 합치기
pseudo_df = pd.concat([confident_stable, confident_unstable])
pseudo_df = pseudo_df[['id', 'label']]
pseudo_df['source'] = 'test' # 이 이미지들은 test 폴더에 있으니까 경로 표시!

print(f"🔥 오리지널 데이터 1,000개에 추가될 확실한 Test 샘플 개수: {len(pseudo_df)} 개")

# 4. 기존 train.csv 랑 합체!!!
train_df = pd.read_csv('./open/train.csv')
train_df['source'] = 'train'
new_train_df = pd.concat([train_df, pseudo_df], ignore_index=True)

# 5. 새로운 train_pseudo.csv 로 저장
new_train_df.to_csv('./open/train_pseudo.csv', index=False)
print("✅[open/train_pseudo.csv] 생성 완료!!")