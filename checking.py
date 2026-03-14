import pandas as pd
import numpy as np

# 모델 로드
main = pd.read_csv("subs/mixed.csv")      # 메인 모델
alt  = pd.read_csv("subs/swinv2.csv")     # 보정 모델

pA = main["stable_prob"].values
pB = alt["stable_prob"].values

diff = np.abs(pA - pB)

# threshold
t1 = 0.35
t2 = 0.60

# 가중치
wA1, wB1 = 0.85, 0.15   # stage1
wA2, wB2 = 0.30, 0.70   # stage2

p_final = pA.copy()

mask1 = (diff >= t1) & (diff < t2)
mask2 = diff >= t2

# Stage 1 (약한 앙상블)
p_final[mask1] = (wA1*pA[mask1] + wB1*pB[mask1]) / (wA1 + wB1)

# Stage 2 (강한 보정)
p_final[mask2] = (wA2*pA[mask2] + wB2*pB[mask2]) / (wA2 + wB2)

print("Stage0:", np.sum(diff < t1))
print("Stage1:", np.sum(mask1))
print("Stage2:", np.sum(mask2))

# ⭐ 확률 안정화 (LogLoss 방어)
p_final = np.clip(p_final, 1e-5, 1 - 1e-5)

import numpy as np


print("min : ", np.min(p_final))
print("1% : ", np.percentile(p_final, 1))
print("5% : ", np.percentile(p_final, 5))
print("50% : ", np.percentile(p_final, 50))
print("95% : ", np.percentile(p_final, 95))
print("99% : ", np.percentile(p_final, 99))
print("max : ", np.max(p_final))

# 제출 파일 생성
sub = pd.DataFrame()
sub["id"] = main["id"]
sub["stable_prob"] = p_final
sub["unstable_prob"] = 1 - p_final

sub = sub[["id","unstable_prob","stable_prob"]]

sub.to_csv("subs/two_stage_clip.csv", index=False)

print("two_stage_clip.csv 생성 완료")