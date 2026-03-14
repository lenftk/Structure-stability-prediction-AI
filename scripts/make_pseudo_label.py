import pandas as pd

# 🚨 0.032를 기록한 완벽한 단일 모델 파일을 불러옵니다!
best_sub = pd.read_csv('subs/sub_convnext_small.csv') 

UPPER_BOUND = 0.98  # 98% 이상 확신하는 정답만
LOWER_BOUND = 0.02  # 2% 이하 확신하는 정답만

confident_stable = best_sub[best_sub['stable_prob'] >= UPPER_BOUND].copy()
confident_unstable = best_sub[best_sub['stable_prob'] <= LOWER_BOUND].copy()

confident_stable['label'] = 'stable'
confident_unstable['label'] = 'unstable'

pseudo_df = pd.concat([confident_stable, confident_unstable])
pseudo_df = pseudo_df[['id', 'label']]
pseudo_df['source'] = 'test'

print(f"🔥 추출된 확실한 Test 샘플 개수: {len(pseudo_df)} 개")

train_df = pd.read_csv('./open/train.csv')
train_df['source'] = 'train'
new_train_df = pd.concat([train_df, pseudo_df], ignore_index=True)

# 빵빵해진 데이터 저장!
new_train_df.to_csv('./open/train_pseudo.csv', index=False)
print("✅ [train_pseudo.csv] 생성 완료!!")