# Skin Disease Classifier

6-class facial skin disease classifier trained on the AI Hub dataset with tone-balanced augmentation.

**Classes:** psoriasis, atopy, acne, normal, rosacea, seborrheic

## Model Performance (Test Set, n=225)

| Model | F1 | Accuracy |
|---|---:|---:|
| efficientnet_swin (ensemble) | **0.909** | **0.911** |
| resnet_swin (ensemble) | 0.906 | 0.907 |
| densenet_swin (ensemble) | 0.876 | 0.876 |
| densenet | 0.898 | 0.902 |
| swin | 0.885 | 0.880 |
| efficientnet | 0.878 | 0.880 |
| resnet | 0.846 | 0.853 |

Training run: 2026-05-03 to 2026-05-04. Full metrics in [training-results-20260504/](training-results-20260504/).

> Trained model weights (`.pth`) are not included due to file size limits. Contact the author to obtain them.

## Repository Structure

```
inference-code/
  models/
    base_model.py          # NUM_CLASSES, CLASS_NAMES
    resnet.py              # ResNet-50, 256x256
    densenet.py            # DenseNet-121, 256x256
    efficientnet_v2.py     # EfficientNetV2-S, 384x384
    swin_transformer.py    # Swin-T, 256x256
    ensemble.py            # Soft-voting ensemble (any subset of the 4)

data-preprocessing/
  face-segmentation/
    segment_faces.py       # SegFormer + SAM2 face segmentation pipeline
    config.json            # Runtime paths, model names, segmentation params
    requirements.txt
  color-correction/
    skin_tone_level_report.py      # ITA-based 3-level tone analysis (step 1)
    skin_tone_ita6_report.py       # ITA 6-category breakdown (step 2)
    build_tone_balanced_split.py   # Tone-balanced split with augmentation (step 3)
    colorfit_tone_balanced_normal.py  # LAB color shift on normal class (step 4)
    requirements.txt

training-results-20260504/
  run_summary.json          # Overall run metadata (7 models, exit codes)
  train.log                 # Full training log
  checkpoints/
    {model}/summary.json    # Per-model F1, accuracy, class metrics
```

## Inference

Each model file is self-contained. Add `inference-code/` to your Python path, then import directly:

```python
import sys
sys.path.insert(0, "inference-code")

from models.ensemble import EnsembleModel
from models.base_model import CLASS_NAMES

# Load ensemble with saved weights
model = EnsembleModel(pretrained=False, weights_dir="/path/to/weights")
model.eval()

import torch
from PIL import Image
import torchvision.transforms.functional as TF

img = TF.to_tensor(Image.open("face.jpg").convert("RGB")).unsqueeze(0)  # [1,3,H,W]
with torch.no_grad():
    log_probs = model(img)   # input any size >= 256; internally resized per sub-model
probs = torch.exp(log_probs)
pred = CLASS_NAMES[probs.argmax().item()]
```

To use a single model instead:

```python
from models.efficientnet_v2 import build_efficientnetv2, INPUT_SIZE
import torch

model = build_efficientnetv2(pretrained=False)
model.load_state_dict(torch.load("efficientnet_best.pth", map_location="cpu", weights_only=True))
model.eval()
```

### Ensemble weight filenames

| Key | File |
|---|---|
| `efficientnet` | `efficientnet_best.pth` |
| `resnet` | `resnet_best.pth` |
| `densenet` | `densenet_best.pth` |
| `swin` | `swin_best.pth` |

Partial ensembles (e.g. efficientnet + swin only):

```python
model = EnsembleModel(pretrained=False, components=("efficientnet", "swin"), weights_dir="/path/to/weights")
```

## Data Preprocessing Pipeline

Run in order on your own dataset:

### 1. Face segmentation

Uses SegFormer (face-parsing) to mask hair/eyebrows/eyes, then SAM2 to isolate foreground.

```bash
cd data-preprocessing/face-segmentation
pip install -r requirements.txt
python segment_faces.py --src /path/to/images --dst /path/to/output --device 0
```

Edit `config.json` to change model names or SAM2 parameters. Pass `--config /path/to/config.json` to override defaults.

### 2–4. Tone-balanced augmentation

```bash
cd data-preprocessing/color-correction
pip install -r requirements.txt

# Step 1: compute ITA skin tone levels (3-level)
python skin_tone_level_report.py

# Step 2: compute ITA 6-category breakdown (uses output from step 1)
python skin_tone_ita6_report.py

# Step 3: build tone-balanced split with augmentation
python build_tone_balanced_split.py --out-root /path/to/output

# Step 4: LAB color shift on normal class (lighter skin → normal distribution)
python colorfit_tone_balanced_normal.py --src /path/to/dataset
```

Each script expects `datasets/` relative to the project root (two directories up from the script).
Reports are written to `reports/` under the project root.

## Dependencies

- Python 3.10+
- PyTorch 2.x + torchvision
- For segmentation: `transformers>=4.46.0`, `Pillow`, `numpy`, `huggingface_hub`, `safetensors`, `accelerate`
- For color-correction: `opencv-python-headless`, `numpy`, `Pillow`
