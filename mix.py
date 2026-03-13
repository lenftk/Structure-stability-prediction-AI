import pandas as pd
import numpy as np

from scipy.special import logit, expit

# 1. 파일 불러오기
sub_small = pd.read_csv('subs/sub_convnext_small.csv') # 0.032 찍은 Small
# sub_base  = pd.read_csv('subs/sub_convnext_base.csv')      # 0.058 찍은 Base (성능 하락의 원인으로 판단되어 제외)
sub_swinv2 = pd.read_csv('subs/sub_swinv2.csv')

# --- 새로 적용된 [로짓 앙상블 (Logit Ensemble)] ---
# 확률값을 클리핑하여 무한대 발산 방지
eps = 1e-15
p_small = np.clip(sub_small['stable_prob'].values, eps, 1 - eps)
p_swinv2= np.clip(sub_swinv2['stable_prob'].values, eps, 1 - eps)

# 1) 확률(Probability) -> 로짓(Logit) 스케일 변환
logit_small = logit(p_small)
logit_swinv2= logit(p_swinv2)

# 2) 로짓 공간에서의 가중 평균 (압도적으로 좋은 Small 모델 위주)
w_small = 0.95
w_swinv2 = 0.05

blended_logit = (logit_small * w_small) + (logit_swinv2 * w_swinv2)

# 3) 다시 확률(Probability)로 복원 (Sigmoid 통과)
final_stable = expit(blended_logit)

# 4. 제출 파일 만들기 
# 🚨 데이콘 규정: 반드시 id -> unstable_prob -> stable_prob 순서여야 함!
final_sub = pd.DataFrame()
final_sub['id'] = sub_small['id']

# 🚨 unstable_prob를 먼저 넣습니다!
final_sub['unstable_prob'] = 1.0 - np.clip(final_stable, 1e-15, 1 - 1e-15)
# 그 다음 stable_prob를 넣습니다!
final_sub['stable_prob'] = np.clip(final_stable, 1e-15, 1 - 1e-15)

final_sub.to_csv('subs/mix.csv', index=False)
print("🚀[mix.csv] 컬럼 에러 수정 완료!! 당장 다시 제출해보세요!")  