"""
전체 모델 혼동행렬 + 클래스별 지표 그래프 생성 스크립트 (Kaggle 실행용)

Single/Ensemble 모델 7개의 혼동행렬, 클래스별 Precision/Recall/F1,
모델 간 비교 히트맵 및 정확도 막대 그래프를 저장한다.

경로 (Kaggle 기준):
    BASE(루트 경로)           Kaggle 입력 데이터셋 루트
    DATA_ROOT(데이터 경로)    테스트셋 최상위 폴더 (test/ 하위 폴더 포함)
    W_SINGLE(단일 가중치)     단일 모델 .pth 파일 폴더
    W_ENSEMBLE(앙상블 가중치) 앙상블 모델 .pth 파일 폴더
    OUT_DIR(출력 경로)        결과 그래프 저장 폴더

출력 파일 (모델별):
    cm_{모델명}.png       혼동행렬 (수치 + 비율 정규화 2종)
    metrics_{모델명}.png  클래스별 Precision / Recall / F1 막대 그래프

출력 파일 (통합):
    comparison_heatmap.png  모델 간 클래스별 지표 비교 히트맵
    accuracy_bar.png        모델별 전체 정확도 막대 그래프
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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
OUT_DIR    = "/kaggle/working/confusion_metrics"
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


def build_model(name, weights_path):
    if name in PARTIAL_ENSEMBLE:
        m = EnsembleModel(pretrained=False, components=PARTIAL_ENSEMBLE[name])
    elif name == "resnet":       m = build_resnet(pretrained=False)
    elif name == "densenet":     m = build_densenet(pretrained=False)
    elif name == "efficientnet": m = build_efficientnetv2(pretrained=False)
    elif name == "swin":         m = build_swin_transformer(pretrained=False)
    m.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return m.to(device)


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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            p = torch.exp(out) if is_log else F.softmax(out, dim=1)
            all_preds.extend(p.argmax(dim=1).cpu().tolist())
            all_labels.extend(lbls.tolist())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion(preds, labels, model_name, kind):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Count", "Normalized (Recall per class)"]
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(1 if fmt == ".2f" else None))
        ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)
        thresh = data.max() / 2
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = f"{data[i, j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        fontsize=9, color="white" if data[i, j] > thresh else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Confusion Matrix — {model_name} ({kind})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"cm_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_metrics(preds, labels, model_name, kind):
    prec = precision_score(labels, preds, average=None, zero_division=0)
    rec  = recall_score(labels, preds, average=None, zero_division=0)
    f1   = f1_score(labels, preds, average=None, zero_division=0)
    acc  = (preds == labels).mean()

    x = np.arange(len(CLASS_NAMES))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, prec, w, label="Precision", color="steelblue")
    ax.bar(x,     rec,  w, label="Recall",    color="tomato")
    ax.bar(x + w, f1,   w, label="F1",        color="seagreen")

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i,     r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class Metrics — {model_name} ({kind})  |  Accuracy: {acc:.4f}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"metrics_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def plot_comparison_heatmap(all_metrics):
    metrics = ["precision", "recall", "f1"]
    model_names = [m[0] for m in MODELS]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, metric in zip(axes, metrics):
        data = np.array([all_metrics[n][metric] for n in model_names])
        im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(len(model_names))); ax.set_yticklabels(model_names)
        for i in range(len(model_names)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if data[i, j] < 0.75 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(metric.capitalize())
    fig.suptitle("Model Comparison — Per-class Metrics", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "comparison_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_accuracy_bar(all_metrics):
    model_names = [m[0] for m in MODELS]
    kinds       = [m[1] for m in MODELS]
    accs        = [all_metrics[n]["accuracy"] for n in model_names]
    colors      = ["steelblue" if k == "Single" else "tomato" for k in kinds]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, accs, color=colors)
    ax.set_ylim([min(accs) - 0.05, 1.01])
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy — All Models")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[Patch(color="steelblue", label="Single"),
                       Patch(color="tomato",    label="Ensemble")])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


all_metrics = {}

for name, kind, wpath, isize, is_log in MODELS:
    print(f"[{kind}] {name} ...")
    model  = build_model(name, wpath)
    loader = get_loader(isize)
    preds, labels = collect(model, loader, is_log)
    plot_confusion(preds, labels, name, kind)
    all_metrics[name] = plot_metrics(preds, labels, name, kind)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n통합 비교 그래프 생성 중...")
plot_comparison_heatmap(all_metrics)
plot_accuracy_bar(all_metrics)

print(f"\n완료. 결과: {OUT_DIR}/")
print("\n파일 목록:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
