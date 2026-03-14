import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ==========================================
# 1. 비교할 두 파일 경로 설정
# ==========================================
# FILE_A: 내가 가진 최고의 점수 (기준점 / 0.032 모델)
FILE_A = 'subs/mixed.csv' 

# FILE_B: 테스트해보고 싶은 새로운 모델이나 앙상블 파일
FILE_B = 'sub_swinv2.csv'

# ==========================================
# 2. 데이터 불러오기 및 병합
# ==========================================
df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

# ID 기준으로 병합 (순서가 달라도 안전하게 매칭)
merged = pd.merge(df_a, df_b, on='id', suffixes=('_A', '_B'))

prob_a = merged['stable_prob_A'].values
prob_b = merged['stable_prob_B'].values

# ==========================================
# 3. 분석 지표 계산
# ==========================================
# ① Mean Absolute Difference (평균 확률 오차)
# 두 모델이 예측한 확률이 평균적으로 몇 %나 차이나는가?
mad = np.mean(np.abs(prob_a - prob_b))

# ② Correlation (상관계수)
# 두 모델의 예측 경향성이 얼마나 비슷한가? (1.0에 가까울수록 비슷함)
corr, _ = pearsonr(prob_a, prob_b)

# ③ Hard Label Agreement (정답 일치율)
# 0.5를 기준으로 '안정/불안정' 판정을 내렸을 때, 1000문제 중 몇 문제를 똑같이 찍었는가?
pred_a = (prob_a > 0.5).astype(int)
pred_b = (prob_b > 0.5).astype(int)
agreement = np.mean(pred_a == pred_b) * 100
disagree_count = np.sum(pred_a != pred_b)

print("="*50)
print(f"기준 모델 (A): {FILE_A}")
print(f"비교 모델 (B): {FILE_B}")
print("-" * 50)
print(f"🔹 평균 확률 차이 (MAD): {mad:.4f} ({mad*100:.2f}%)")
print(f"🔹 예측 상관계수 (Corr): {corr:.4f}")
print(f"🔹 정답 일치율 (Agree) : {agreement:.2f}% (총 {len(merged)}개 중 {disagree_count}개 의견 불일치)")
print("=" * 50)

# ==========================================
# 4. 심층 분석: "완전히 의견이 엇갈린 치명적 문제들"
# ==========================================
# A는 80% 이상 확신했는데, B는 20% 이하로 확신하는 등 완전히 정반대로 예측한 샘플 추출
extreme_diff = merged[np.abs(merged['stable_prob_A'] - merged['stable_prob_B']) > 0.5]

if len(extreme_diff) > 0:
    print(f"⚠️ [경고] 두 모델이 완전히 정반대로 예측한 데이터가 {len(extreme_diff)}개 있습니다!")
    print(extreme_diff[['id', 'stable_prob_A', 'stable_prob_B']].head(10))
else:
    print("✅ 두 모델이 정반대로 크게 엇갈린 데이터는 없습니다. (안정적인 변화)")