"""
전체 모델 임계값 통합 비교 스크립트 (Kaggle 실행용)

Single/Ensemble 모델 7개를 순서대로 평가하여
ROC·PR 임계값과 AUC를 표와 히트맵으로 출력한다.

경로 (Kaggle 기준):
    BASE(루트 경로)          Kaggle 입력 데이터셋 루트
    DATA_ROOT(데이터 경로)   테스트셋 최상위 폴더 (test/ 하위 폴더 포함)
    W_SINGLE(단일 가중치)    단일 모델 .pth 파일 폴더
    W_ENSEMBLE(앙상블 가중치) 앙상블 모델 .pth 파일 폴더
    OUT_DIR(출력 경로)       결과 그래프 저장 폴더

출력 파일:
    01_heatmap_roc_threshold.png  ROC 기준 임계값 히트맵
    02_heatmap_pr_threshold.png   PR 기준 임계값 히트맵
    03_heatmap_auc.png            클래스별 AUC 히트맵
    04_mean_auc_bar.png           모델별 평균 AUC 막대 그래프
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model       import CLASS_NAMES
from models.resnet           import build_resnet,           INPUT_SIZE as RES_SIZE
from models.densenet         import build_densenet,         INPUT_SIZE as DEN_SIZE
from models.efficientnet_v2  import build_efficientnetv2,   INPUT_SIZE as EFF_SIZE
from models.swin_transformer import build_swin_transformer, INPUT_SIZE as SWN_SIZE
from models.ensemble         import EnsembleModel

BASE       = "/kaggle/input/datasets/jjw905"
DATA_ROOT  = f"{BASE}/skin-classifier-kagglev4/dataset"
W_SINGLE   = f"{BASE}/skin-classifier-weightsv3/skin-classifier-weights/Single models"
W_ENSEMBLE = f"{BASE}/skin-classifier-weightsv3/skin-classifier-weights/Ensemble models"
OUT_DIR    = "/kaggle/working/threshold_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("resnet",            "Single",   f"{W_SINGLE}/resnet_best.pth",              (224, 224), False),
    ("densenet",          "Single",   f"{W_SINGLE}/densenet_best.pth",            (224, 224), False),
    ("efficientnet",      "Single",   f"{W_SINGLE}/efficientnet_best.pth",        (300, 300), False),
    ("swin",              "Single",   f"{W_SINGLE}/swin_best.pth",                (224, 224), False),
    ("resnet_swin",       "Ensemble", f"{W_ENSEMBLE}/resnet_swin_best.pth",       (512, 512), True),
    ("densenet_swin",     "Ensemble", f"{W_ENSEMBLE}/densenet_swin_best.pth",     (512, 512), True),
    ("efficientnet_swin", "Ensemble", f"{W_ENSEMBLE}/efficientnet_swin_best.pth", (512, 512), True),
]

PARTIAL_ENSEMBLE = {
    "densenet_swin":     ("densenet",     "swin"),
    "efficientnet_swin": ("efficientnet", "swin"),
    "resnet_swin":       ("resnet",       "swin"),
}

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}\n")


def build_model(name, weights_path, input_size, is_log):
    if name in PARTIAL_ENSEMBLE:
        model = EnsembleModel(pretrained=False, components=PARTIAL_ENSEMBLE[name])
    elif name == "resnet":
        model = build_resnet(pretrained=False)
    elif name == "densenet":
        model = build_densenet(pretrained=False)
    elif name == "efficientnet":
        model = build_efficientnetv2(pretrained=False)
    elif name == "swin":
        model = build_swin_transformer(pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def get_loader(input_size):
    tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=tf)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)


def collect(model, loader, is_log):
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            p = torch.exp(out) if is_log else F.softmax(out, dim=1)
            probs_list.append(p.cpu())
            labels_list.append(lbls)
    return torch.cat(probs_list).numpy(), torch.cat(labels_list).numpy()


def compute_thresholds(all_probs, all_labels):
    roc_thr, pr_thr, aucs = {}, {}, {}
    for i, cls in enumerate(CLASS_NAMES):
        y = (all_labels == i).astype(int)
        s = all_probs[:, i]
        fpr, tpr, th = roc_curve(y, s)
        j = tpr - fpr
        best = np.argmax(j)
        roc_thr[cls] = float(th[best])
        aucs[cls]    = float(auc(fpr, tpr))
        prec, rec, th2 = precision_recall_curve(y, s)
        mask = rec[:-1] >= 0.95
        if mask.any():
            idx = np.argmax(prec[:-1][mask])
            pr_thr[cls] = float(th2[mask][idx])
        else:
            pr_thr[cls] = float(th2[np.argmax(rec[:-1])])
    return roc_thr, pr_thr, aucs


all_results = {}

for name, kind, wpath, isize, is_log in MODELS:
    print(f"[{kind}] {name} ...")
    model  = build_model(name, wpath, isize, is_log)
    loader = get_loader(isize)
    probs, labels = collect(model, loader, is_log)
    roc_thr, pr_thr, aucs = compute_thresholds(probs, labels)
    all_results[name] = {"roc": roc_thr, "pr": pr_thr, "auc": aucs}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  done. mean AUC: {np.mean(list(aucs.values())):.4f}")

print("\n분석 완료. 그래프 생성 중...\n")

model_names = [m[0] for m in MODELS]
print("=" * 100)
print(f"{'Model':<20} {'Type':<10}", end="")
for cls in CLASS_NAMES:
    print(f"  {cls[:6]:>8}", end="")
print("  │  mean_AUC")
print("-" * 100)

for (name, kind, *_) in MODELS:
    r = all_results[name]
    mean_auc = np.mean(list(r["auc"].values()))
    print(f"\n{'[ROC] '+name:<20} {kind:<10}", end="")
    for cls in CLASS_NAMES:
        print(f"  {r['roc'][cls]:>8.4f}", end="")
    print(f"  │  {mean_auc:.4f}")
    print(f"{'[PR]  '+name:<20} {kind:<10}", end="")
    for cls in CLASS_NAMES:
        print(f"  {r['pr'][cls]:>8.4f}", end="")
    print()

print("=" * 100)


def draw_heatmap(results, models_info, key, title, fname):
    data = np.array([
        [results[m[0]][key][cls] for cls in CLASS_NAMES]
        for m in models_info
    ])
    names = [m[0] for m in models_info]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(names)));      ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i, j] < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


draw_heatmap(all_results, MODELS, "roc", "ROC Threshold (Youden's J) — All Models",
             "01_heatmap_roc_threshold.png")
draw_heatmap(all_results, MODELS, "pr",  "PR Threshold (recall>=0.95) — All Models",
             "02_heatmap_pr_threshold.png")
draw_heatmap(all_results, MODELS, "auc", "AUC per Class — All Models",
             "03_heatmap_auc.png")

fig, ax = plt.subplots(figsize=(10, 5))
names     = [m[0] for m in MODELS]
kinds     = [m[1] for m in MODELS]
mean_aucs = [np.mean(list(all_results[n]["auc"].values())) for n in names]
colors    = ["steelblue" if k == "Single" else "tomato" for k in kinds]
bars = ax.bar(names, mean_aucs, color=colors)
ax.set_ylim([0.95, 1.001])
ax.set_ylabel("Mean AUC")
ax.set_title("Mean AUC by Model")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, mean_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.legend(handles=[Patch(color="steelblue", label="Single"),
                   Patch(color="tomato",    label="Ensemble")])
fig.tight_layout()
path = os.path.join(OUT_DIR, "04_mean_auc_bar.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  saved: {path}")

print(f"\n완료. 결과: {OUT_DIR}/")
