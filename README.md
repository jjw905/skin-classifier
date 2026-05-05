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

각 모델 파일은 독립적으로 동작한다.

### 1단계. 의존성 설치

```bash
pip install torch torchvision pillow
```

### 2단계. 가중치 파일 배치

4개 파일을 동일한 디렉터리(예: `weights/`)에 넣는다.

| 키 | 파일명 | 크기 |
|---|---|---|
| `efficientnet` | `efficientnet_best.pth` | ~82 MB |
| `resnet` | `resnet_best.pth` | ~94 MB |
| `densenet` | `densenet_best.pth` | ~28 MB |
| `swin` | `swin_best.pth` | ~110 MB |

### 3단계. 앙상블 추론

`inference-code/`를 Python 경로에 추가한 뒤 4개 모델의 출력 확률을 평균 내는 소프트 보팅 방식으로 최종 클래스를 결정한다.

```python
import sys
sys.path.insert(0, "inference-code")

from models.ensemble import EnsembleModel
from models.base_model import CLASS_NAMES
import torch
from PIL import Image
import torchvision.transforms.functional as TF

model = EnsembleModel(pretrained=False, weights_dir="weights/")
model.eval()

img = TF.to_tensor(Image.open("face.jpg").convert("RGB")).unsqueeze(0)

with torch.no_grad():
    log_probs = model(img)

probs = torch.exp(log_probs)
pred = CLASS_NAMES[probs.argmax().item()]
print(pred)

for cls, p in zip(CLASS_NAMES, probs[0].tolist()):
    print(f"{cls}: {p:.3f}")
```

입력 이미지는 256px 이상이면 된다. 내부에서 모델별로 자동 리사이즈되며, 출력은 log-softmax이므로 `torch.exp()`로 확률로 변환한다.

### 단일 모델 추론

특정 모델 하나만 사용할 경우이다. 아래는 EfficientNetV2-S 예시이며, 나머지 모델도 동일한 방식으로 사용한다.

```python
import sys
sys.path.insert(0, "inference-code")

from models.efficientnet_v2 import build_efficientnetv2
from models.base_model import CLASS_NAMES
import torch
from PIL import Image
import torchvision.transforms.functional as TF

model = build_efficientnetv2(pretrained=False)
model.load_state_dict(torch.load("efficientnet_best.pth", map_location="cpu", weights_only=True))
model.eval()

img = TF.to_tensor(Image.open("face.jpg").convert("RGB")).unsqueeze(0)

with torch.no_grad():
    log_probs = model(img)

probs = torch.exp(log_probs)
pred = CLASS_NAMES[probs.argmax().item()]
print(pred)
```

다른 모델의 build 함수는 아래와 같다.

| 모델 | import |
|---|---|
| ResNet-50 | `from models.resnet import build_resnet` |
| DenseNet-121 | `from models.densenet import build_densenet` |
| EfficientNetV2-S | `from models.efficientnet_v2 import build_efficientnetv2` |
| Swin-T | `from models.swin_transformer import build_swin` |

### 일부 모델만 조합하는 앙상블

`components`에 사용할 키를 튜플로 지정한다. 지정하지 않으면 4개 전체를 사용한다.

```python
model = EnsembleModel(
    pretrained=False,
    components=("efficientnet", "swin"),
    weights_dir="weights/",
)
```

---

## 데이터 전처리 파이프라인

아래 순서대로 실행한다. 각 스크립트는 프로젝트 루트 기준으로 `datasets/`를 참조하며, 결과 리포트는 `reports/`에 저장된다.

### 1단계. 얼굴 분할 (`face-segmentation/segment_faces.py`)

SegFormer(face-parsing)로 머리카락·눈썹·눈 영역을 회색으로 마스킹하고, SAM2로 얼굴 전경을 분리해 배경을 회색으로 대체한다. 처음 실행 시 HuggingFace에서 모델이 자동 다운로드된다.

```bash
cd data-preprocessing/face-segmentation
pip install -r requirements.txt

python segment_faces.py \
  --src /path/to/원본이미지 \
  --dst /path/to/분할결과 \
  --device 0
```

주요 옵션은 아래와 같다.

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--src` | 원본 이미지 루트 (클래스 디렉터리 포함) | config.json 참조 |
| `--dst` | 분할 결과 저장 경로 | config.json 참조 |
| `--device` | GPU 인덱스 (`-1`이면 CPU) | `0` |
| `--skip-existing` | 이미 처리된 파일 건너뜀 | 기본 활성화 |
| `--no-skip` | 이미 처리된 파일도 재처리 | - |
| `--limit N` | 처음 N개만 처리 (테스트용) | 전체 처리 |

모델명과 SAM2 파라미터는 `config.json`에서 변경한다. 커맨드라인 인자가 config보다 우선 적용된다.

```bash
python segment_faces.py --config /path/to/config.json
```

### 2단계. ITA 피부톤 3단계 분석 (`skin_tone_level_report.py`)

CIELAB 색공간의 ITA(Individual Typology Angle) 값을 기준으로 각 이미지를 밝음·중간·어두움 3단계로 분류한다. 결과 JSON은 이후 단계에서 입력으로 사용된다.

```bash
cd data-preprocessing/color-correction
pip install -r requirements.txt

python skin_tone_level_report.py
```

`PROJECT_ROOT/datasets/`의 이미지를 읽고 `PROJECT_ROOT/reports/`에 결과를 저장한다.

### 3단계. ITA 6구간 세분화 (`skin_tone_ita6_report.py`)

2단계 결과를 바탕으로 ITA 값을 6구간(dark / brown / tan / intermediate / light / very light)으로 세분화한다. 구간별 샘플 수와 분포를 확인하는 데 사용한다.

```bash
python skin_tone_ita6_report.py
```

### 4단계. 피부톤 균형 분할 + 증강 (`build_tone_balanced_split.py`)

3단계 구간별 분포를 기반으로 train/val 데이터셋을 구성한다. 샘플 수가 부족한 피부톤 구간은 이미지 증강(회전·플립·색상 변환 등)으로 보완하며, 증강 비율은 최대 3배로 제한한다.

```bash
python build_tone_balanced_split.py --out-root /path/to/출력
```

`--out-root`에 균형 잡힌 train/val 데이터셋이 생성된다.

### 5단계. 정상 클래스 LAB 색보정 (`colorfit_tone_balanced_normal.py`)

정상(normal) 클래스에서 밝은 피부톤 샘플이 과다 대표되는 경우, LAB 색공간에서 중앙값 기반 색 이동을 적용해 다른 질환 클래스의 피부톤 분포에 맞춘다.

```bash
python colorfit_tone_balanced_normal.py --src /path/to/dataset
```

`--src`에는 4단계에서 생성한 tone-balanced 데이터셋 루트를 지정한다.

---

## 의존성

- Python 3.10+
- PyTorch 2.x + torchvision
- 얼굴 분할: `transformers>=4.46.0`, `Pillow`, `numpy`, `huggingface_hub`, `safetensors`, `accelerate`
- 색보정: `opencv-python-headless`, `numpy`, `Pillow`
