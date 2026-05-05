# 안면 피부 질환 분류기

AI Hub 피부 데이터셋으로 학습한 6종 안면 피부 질환 분류 모델이다.
피부톤 불균형을 보정하는 tone-balanced augmentation을 적용했다.

**분류 클래스:** 건선(psoriasis), 아토피(atopy), 여드름(acne), 정상(normal), 주사(rosacea), 지루성피부염(seborrheic)

---

## 모델 성능 (테스트셋, n=225)

| 모델 | F1 | Accuracy |
|---|---:|---:|
| efficientnet_swin (앙상블) | **0.909** | **0.911** |
| resnet_swin (앙상블) | 0.906 | 0.907 |
| densenet_swin (앙상블) | 0.876 | 0.876 |
| densenet | 0.898 | 0.902 |
| swin | 0.885 | 0.880 |
| efficientnet | 0.878 | 0.880 |
| resnet | 0.846 | 0.853 |

학습 기간: 2026-05-03 ~ 2026-05-04. 전체 지표는 [training-results-20260504/](training-results-20260504/)에서 확인할 수 있다.

> 학습된 모델 가중치(`.pth`)는 용량 문제로 포함하지 않는다. 필요하면 저자에게 문의한다.

---

## 레포지토리 구조

```
inference-code/
  models/
    base_model.py               # NUM_CLASSES, CLASS_NAMES 정의
    resnet.py                   # ResNet-50, 입력 256×256
    densenet.py                 # DenseNet-121, 입력 256×256
    efficientnet_v2.py          # EfficientNetV2-S, 입력 384×384
    swin_transformer.py         # Swin-T, 입력 256×256
    ensemble.py                 # 소프트 보팅 앙상블 (4개 모델 중 임의 조합 가능)

data-preprocessing/
  face-segmentation/
    segment_faces.py            # SegFormer + SAM2 얼굴 분할 파이프라인
    config.json                 # 경로, 모델명, 분할 파라미터
    requirements.txt
  color-correction/
    skin_tone_level_report.py         # ITA 기반 3단계 피부톤 분석 (1단계)
    skin_tone_ita6_report.py          # ITA 6구간 세분화 (2단계)
    build_tone_balanced_split.py      # 피부톤 균형 분할 + 증강 (3단계)
    colorfit_tone_balanced_normal.py  # 정상 클래스 LAB 색보정 (4단계)
    requirements.txt

training-results-20260504/
  run_summary.json              # 전체 실행 메타데이터 (7개 모델, 종료 코드)
  train.log                     # 전체 학습 로그
  checkpoints/
    {model}/summary.json        # 모델별 F1, Accuracy, 클래스별 지표
```

---

## 추론

각 모델 파일은 독립적으로 동작한다. `inference-code/`를 Python 경로에 추가한 뒤 import해서 사용한다.

### 앙상블 모델

```python
import sys
sys.path.insert(0, "inference-code")

from models.ensemble import EnsembleModel
from models.base_model import CLASS_NAMES
import torch
from PIL import Image
import torchvision.transforms.functional as TF

model = EnsembleModel(pretrained=False, weights_dir="/path/to/weights")
model.eval()

img = TF.to_tensor(Image.open("face.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    log_probs = model(img)
probs = torch.exp(log_probs)
pred = CLASS_NAMES[probs.argmax().item()]
```

입력 이미지는 256px 이상이면 되고, 내부에서 모델별로 자동 리사이즈된다. 출력은 log-softmax이므로 `torch.exp()`로 확률로 변환한다.

### 단일 모델

```python
import sys
sys.path.insert(0, "inference-code")

from models.efficientnet_v2 import build_efficientnetv2
import torch

model = build_efficientnetv2(pretrained=False)
model.load_state_dict(torch.load("efficientnet_best.pth", map_location="cpu", weights_only=True))
model.eval()
```

### 앙상블 가중치 파일명

| 키 | 파일명 |
|---|---|
| `efficientnet` | `efficientnet_best.pth` |
| `resnet` | `resnet_best.pth` |
| `densenet` | `densenet_best.pth` |
| `swin` | `swin_best.pth` |

일부 모델만 조합할 경우 `components`에 원하는 키를 지정한다.

```python
model = EnsembleModel(pretrained=False, components=("efficientnet", "swin"), weights_dir="/path/to/weights")
```

---

## 데이터 전처리 파이프라인

아래 순서대로 실행한다. 각 스크립트는 프로젝트 루트 기준으로 `datasets/`를 참조하며, 결과 리포트는 `reports/`에 저장된다.

### 1단계. 얼굴 분할

SegFormer(face-parsing)로 머리카락·눈썹·눈 영역을 마스킹하고, SAM2로 얼굴 전경을 분리한다.

```bash
cd data-preprocessing/face-segmentation
pip install -r requirements.txt
python segment_faces.py --src /path/to/images --dst /path/to/output --device 0
```

모델명과 SAM2 파라미터는 `config.json`에서 변경한다. `--config /path/to/config.json`으로 설정 파일을 오버라이드할 수 있다.

### 2–4단계. 피부톤 균형 증강

```bash
cd data-preprocessing/color-correction
pip install -r requirements.txt

python skin_tone_level_report.py
python skin_tone_ita6_report.py
python build_tone_balanced_split.py --out-root /path/to/output
python colorfit_tone_balanced_normal.py --src /path/to/dataset
```

1단계에서 ITA 기반 3단계 피부톤 분석을 수행하고, 2단계에서 6구간으로 세분화한다. 3단계에서 피부톤 균형을 맞춘 train/val 분할과 증강을 적용하며, 4단계에서 정상 클래스의 밝은 피부톤 샘플에 LAB 색보정을 적용해 분포를 일반화한다.

---

## 의존성

- Python 3.10+
- PyTorch 2.x + torchvision
- 얼굴 분할: `transformers>=4.46.0`, `Pillow`, `numpy`, `huggingface_hub`, `safetensors`, `accelerate`
- 색보정: `opencv-python-headless`, `numpy`, `Pillow`
