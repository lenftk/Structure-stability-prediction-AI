# 구조물 안정성 예측 AI 경진대회 - Top3 솔루션 파이프라인

본 저장소 코드는 데이콘 [구조물 안정성 물리 추론 AI 경진대회]에서 상위권 달성을 위해 고안된 **Advanced 딥러닝 예측 모델**입니다.
이 코드는 주어진 정적 이미지(Front, Top)뿐만 아니라, `simulation.mp4` 비디오에서 **물리적 동역학 정보(지식)**를 추출하여 이미지 모델에 주입하는 **Knowledge Distillation (지식 증류)** 기법과 **Cross-Attention** 기술이 모두 적용되어 있습니다.

---

## 🚀 파이프라인 실행 방법 가이드 (GPU 서버 환경 기준)

### 0. 데이터 및 파일 준비 
1. `open.zip` 압축을 해제하여 이 폴더 안에 `open/` 이라는 이름으로 둡니다. (내부에 `train/`, `dev/`, `test/`, `train.csv` 등이 존재해야 합니다)
2. 터미널(혹은 아나콘다 프롬프트)을 열고 현재 이 폴더로 진입합니다.

### 1. 관련 라이브러리 설치
작성하신 `requirements.txt`를 통해 필수 딥러닝 패키지를 서버에 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. 한 번의 명령어로 전체 파이프라인 학습 및 예측 완료하기!
복잡한 스크립트를 순서대로 칠 필요 없이, 제공된 **`main.py` 통합 실행 스크립트**를 사용하여 전체 과정을 자동화했습니다!

터미널에 아래 명령어를 단 한 줄만 입력하세요.
```bash
python main.py --data_dir ./open --batch_size 16 --epochs 30 --output final_submission.csv
```

**자동으로 수행되는 작업:**
1. **[Step 1] Video Teacher 모델 학습:** 비디오 데이터(`simulation.mp4`) 기반 붕괴 동역학 지식 사전학습
2. **[Step 2] Pseudo-Label 생성:** Teacher 모델이 `train` 데이터의 "세밀한 확률표(Soft Level)" 정답 추출
3. **[Step 3] 고급 Student 모델 강력 학습:** 2D 이미지만 보는 타겟 모델을 Cross-Attention과 지식 증류(Distillation)로 학습
4. **[Step 4] 시범 추론 및 제출파일 생성:** 성공적으로 학습된 최고 성능 가중치를 이용해 1,000건의 최종 답안을 자동 생성

모든 과정이 다 끝나면 **현재 폴더에 `final_submission.csv` 파일이 툭! 떨어집니다.** 이를 데이콘에 제출하시면 됩니다.

---

💡 **Top 1 달성 추가 미세조정 팁**
- `python main.py` 명령어 실행 시 `--student_backbone` 옵션을 바꿔가며(예: `swinv2_tiny_window16_256`, `tf_efficientnetv2_s` 등) 서로 다른 2~3종류의 모델을 추가로 훈련시키세요.
- 이후 다르게 나온 `submission.csv` 결과 파일들의 확률들을 단순 평균(Soft-Voting 앙상블)하여 제출물로 만들면 점수가 드라마틱하게 상승합니다.
