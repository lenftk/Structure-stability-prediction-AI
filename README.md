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

### 2. Teacher 모델 (Video) 소프트 라벨 추출하기
대회 `train` 폴더에만 동영상(`simulation.mp4`)이 들어있습니다. 이 동영상을 본 똑똑한 Teacher 모델의 통찰력(지식)을 추출하는 과정입니다.
(만약 별도의 Teacher 모델 가중치가 없다면 본인의 비디오 학습 모델 결과를 사용하거나, 현재는 데모용 임시 Teacher 추출을 해볼 수 있습니다.)

```bash
# 기본적으로 포함된 오픈소스 비디오 백본을 사용해 Train셋에 대한 '가짜 라벨(Soft Label)'을 찍어냅니다.
# --weights_path 에는 서버에서 사전에 학습시킨 비디오 모델의 가중치 경로를 주거나 임시 체크포인트를 사용하세요.
python generate_pseudo_labels.py --data_dir ./open --weights_path 체크포인트경로.pth
```
- **결과물:** `teacher_soft_labels.csv` 파일이 생성되며, 내부에는 1과 0이 아닌 `[0.83, 0.17]` 등 세밀한 정답 확률이 담기게 됩니다.

### 3. 최종 Student 모델 (제출용 2D 이미지 모델) 강력하게 학습하기
이제 테스트 환경과 동일하게 정적 이미지(Front, Top) 2장만을 보는 2D 타겟 모델(Student)을 학습시킵니다.
하지만 이때 `CrossEntropyLoss`(딱딱한 정답)뿐만 아니라, 앞서 생성한 Teacher의 소프트 라벨러(세밀한 정답)를 맞춰가며 모델 체급을 극한으로 높입니다 (Knowledge Distillation).

```bash
# 배치 사이즈가 서버의 그래픽카드 메모리에 안맞으면 숫자를(예: 16 -> 8) 줄여주세요.
python train_student.py --data_dir ./open --teacher_preds teacher_soft_labels.csv --batch_size 16 --epochs 30 --backbone convnext_tiny
```
- **결과물:** 훈련이 성공적으로 종료되면, `checkpoints/` 폴더 내부에 Validation LogLoss 점수가 가장 좋았던 `best_student_convnext_tiny.pth` (최적 가중치 파일)가 저장됩니다.

*(단순 베이스라인 점수가 궁금하시다면 `python train.py`를 실행하여 앙상블 비교군을 만드실 수 있습니다.)*

### 4. 최종 시범 추론(Inference) 및 제출 파일 생성
방금 서버에서 학습된 가장 똑똑한 모델의 가중치로, `test` 데이터 1,000건의 정답을 예측합니다.

먼저, `inference.py`를 열어 `AdvancedStructureModel`을 임포트하도록만 스위치 해줍니다.
(현재 `inference.py`는 기본 `StructureStabilityModel`를 불러오게 되어 있습니다.)
```python
# inference.py 코드의 상단에서 아래로 수정
from model_advanced import AdvancedStructureModel as StructureStabilityModel
```

이후 추론 코드를 실행합니다.
```bash
python inference.py --data_dir ./open --weights_path checkpoints/best_student_convnext_tiny.pth --backbone convnext_tiny --output_file my_top_submission.csv
```
- **최종 산출물:** `my_top_submission.csv` 파일이 생성되며, 이를 데이콘 홈페이지에 제출하시면 됩니다!

---

💡 **Top 1 달성 추가 미세조정 팁**
- `train_student.py` 스크립트 실행 시 `--backbone` 옵션을 `tf_efficientnetv2_s` 나 `swinv2_tiny_window16_256` 로 변경하여 2~3종류의 모델을 각각 끝까지 훈련시키세요.
- 이후 나온 3가지 형태의 `submission.csv` 결과 파일들의 확률을 평균 내어(Soft-Voting 앙상블) 1개의 제출물로 만들면 점수가 드라마틱하게 상승합니다.
