import pandas as pd
import glob

# subs 폴더 안의 3개 csv 파일 불러오기
sub_files = glob.glob('./subs/*.csv')
print(f"총 {len(sub_files)}개의 파일을 앙상블합니다: {sub_files}")

# 앙상블 틀 잡기
ensemble_df = pd.read_csv(sub_files[0])
stable_probs = ensemble_df['stable_prob'].values / len(sub_files)
unstable_probs = ensemble_df['unstable_prob'].values / len(sub_files)

# 나머지 파일 확률 누적 합산
for file in sub_files[1:]:
    df = pd.read_csv(file)
    stable_probs += df['stable_prob'].values / len(sub_files)
    unstable_probs += df['unstable_prob'].values / len(sub_files)

# 결과 저장
ensemble_df['stable_prob'] = stable_probs
ensemble_df['unstable_prob'] = unstable_probs

# LogLoss 에러 방지용 안전장치
ensemble_df['stable_prob'] = ensemble_df['stable_prob'].clip(1e-15, 1 - 1e-15)
ensemble_df['unstable_prob'] = 1.0 - ensemble_df['stable_prob']

# 🌟 다음 단계(Pseudo Label)를 위해 이름을 이렇게 저장합니다!
ensemble_df.to_csv('submission_ensemble_best.csv', index=False)
print("🏆 1차 앙상블 완료![submission_ensemble_best.csv] 파일이 생성되었습니다!")