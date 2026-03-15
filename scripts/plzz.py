import pandas as pd
import numpy as np

# 1. 우리의 최고 에이스 파일 (0.032 원본) 불러오기
ace_sub = pd.read_csv('subs/sub_convnext_small.csv') 

# 2. 온도 조절 함수 (Softening: temperature를 1.0보다 '크게' 줍니다!)
def soften_probabilities(p, temperature=1.15):
    # 온도가 1.15면, 0.999가 0.98 정도로 살짝 부드러워집니다. (치명적 감점 방어)
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
    odds = (p_clipped / (1 - p_clipped)) ** (1.0 / temperature)
    return odds / (1 + odds)

# 3. Softening 적용!
final_stable = soften_probabilities(ace_sub['stable_prob'], temperature=1.15)

# 4. 제출 파일 만들기 (컬럼 순서 엄수)
final_sub = pd.DataFrame()
final_sub['id'] = ace_sub['id']
final_sub['unstable_prob'] = 1.0 - np.clip(final_stable, 1e-15, 1 - 1e-15)
final_sub['stable_prob']   = np.clip(final_stable, 1e-15, 1 - 1e-15)

final_sub.to_csv('ACE_SOFTENED_SUBMISSION.csv', index=False)
print("🚀[ACE_SOFTENED_SUBMISSION.csv] 에이스 방어형(Softening) 파일 완성!")