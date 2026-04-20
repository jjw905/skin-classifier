"""
단일/앙상블 모델 임계값 분석 스크립트

테스트셋에서 신뢰도 분포, ROC 곡선, PR 곡선을 생성하고
클래스별 최적 임계값을 콘솔에 출력한다.

사용법 (단일 모델):
    python scripts/04_threshold_analysis.py \
        --model resnet \
        --weights "skin-classifier-weights/Single models/resnet_best.pth" \
        --data_root /path/to/dataset

사용법 (앙상블):
    python scripts/04_threshold_analysis.py \
        --model ensemble \
        --weights "skin-classifier-weights/Ensemble models/ensemble_best.pth" \
        --data_root /path/to/dataset

인자:
    --model(모델)              분석할 모델 이름 (기본값: resnet)
    --weights(가중치 경로)     불러올 .pth 가중치 파일 경로
    --data_root(데이터 경로)   테스트셋 폴더 경로 (test/ 하위 폴더 포함)
    --batch_size(배치 크기)    한 번에 처리하는 이미지 수 (기본값: 64)
    --min_recall(최소 재현율)  PR 곡선에서 보장할 최소 재현율(recall) (기본값: 0.95)
    --out_dir(출력 경로)       결과 그래프를 저장할 폴더 (기본값: threshold_analysis)

출력 파일:
    01_confidence_distribution.png  정답/오답별 신뢰도 분포 히스토그램
    02_roc_curves.png               클래스별 ROC 곡선 및 최적 임계값
    03_pr_curves.png                클래스별 PR 곡선 및 최적 임계값
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model         import CLASS_NAMES, NUM_CLASSES
from models.resnet             import build_resnet,             INPUT_SIZE as RES_SIZE
from models.densenet           import build_densenet,           INPUT_SIZE as DEN_SIZE
from models.efficientnet_v2    import build_efficientnetv2,     INPUT_SIZE as EFF_SIZE
from models.swin_transformer   import build_swin_transformer,   INPUT_SIZE as SWN_SIZE
from models.ensemble           import EnsembleModel, ALL_COMPONENTS

MODEL_REGISTRY = {
    "resnet":       (build_resnet,           RES_SIZE),
    "densenet":     (build_densenet,         DEN_SIZE),
    "efficientnet": (build_efficientnetv2,   EFF_SIZE),
    "swin":         (build_swin_transformer, SWN_SIZE),
}

PARTIAL_ENSEMBLE = {
    "densenet_swin":     ("densenet",     "swin"),
    "efficientnet_swin": ("efficientnet", "swin"),
    "resnet_swin":       ("resnet",       "swin"),
}


def build_model(model_name, device, weights=None):
    is_log = False
    if model_name == "ensemble":
        model = EnsembleModel(pretrained=False)
        input_size = (512, 512)
        is_log = True
    elif model_name in PARTIAL_ENSEMBLE:
        comps = PARTIAL_ENSEMBLE[model_name]
        model = EnsembleModel(pretrained=False, components=comps)
        input_size = (512, 512)
        is_log = True
    else:
        build_fn, input_size = MODEL_REGISTRY[model_name]
        model = build_fn(pretrained=False)

    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  loaded: {weights}")
    return model.to(device), input_size, is_log


def get_test_loader(data_root, input_size, batch_size=64):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def collect_probs(model, loader, device, is_log):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)
            probs = torch.exp(out) if is_log else F.softmax(out, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def plot_confidence_distribution(all_probs, all_labels, out_dir):
    preds = all_probs.argmax(axis=1)
    correct_mask = (preds == all_labels)
    max_conf = all_probs.max(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(max_conf[correct_mask],  bins=bins, alpha=0.6, color="steelblue", label="Correct")
    ax.hist(max_conf[~correct_mask], bins=bins, alpha=0.6, color="tomato",    label="Incorrect")

    h_correct, edges = np.histogram(max_conf[correct_mask],  bins=bins, density=True)
    h_wrong,   _     = np.histogram(max_conf[~correct_mask], bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    diff = h_correct - h_wrong
    cross_idx = np.where(np.diff(np.sign(diff)))[0]
    cross_val = None
    if len(cross_idx):
        cross_val = centers[cross_idx[0]]
        ax.axvline(cross_val, color="black", linestyle="--", linewidth=1.5,
                   label=f"교차점: {cross_val:.3f}")

    ax.set_xlabel("Max Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "01_confidence_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")
    return cross_val


def plot_roc_curves(all_probs, all_labels, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    optimal = {}

    for i, cls in enumerate(CLASS_NAMES):
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        j = tpr - fpr
        best_idx = np.argmax(j)
        best_thr = thresholds[best_idx]
        optimal[cls] = {"roc_threshold": round(float(best_thr), 4),
                        "youden_j":      round(float(j[best_idx]), 4),
                        "auc":           round(roc_auc, 4),
                        "sensitivity":   round(float(tpr[best_idx]), 4),
                        "specificity":   round(float(1 - fpr[best_idx]), 4)}

        ax = axes[i]
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.scatter(fpr[best_idx], tpr[best_idx], color="red", zorder=5,
                   label=f"Youden thr={best_thr:.3f}")
        ax.set_title(f"{cls}")
        ax.set_xlabel("FPR (1-Specificity)")
        ax.set_ylabel("TPR (Sensitivity)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves — Youden's J Optimal Threshold", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "02_roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")
    return optimal


def plot_pr_curves(all_probs, all_labels, out_dir, min_recall=0.95):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    optimal = {}

    for i, cls in enumerate(CLASS_NAMES):
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        mask = recall[:-1] >= min_recall
        if mask.any():
            best_idx  = np.argmax(precision[:-1][mask])
            cand_thr  = thresholds[mask][best_idx]
            cand_prec = precision[:-1][mask][best_idx]
            cand_rec  = recall[:-1][mask][best_idx]
        else:
            cand_thr  = thresholds[np.argmax(recall[:-1])]
            cand_prec = precision[:-1][np.argmax(recall[:-1])]
            cand_rec  = recall[:-1][np.argmax(recall[:-1])]

        optimal[cls] = {"pr_threshold": round(float(cand_thr), 4),
                        "precision":    round(float(cand_prec), 4),
                        "recall":       round(float(cand_rec), 4)}

        ax = axes[i]
        ax.plot(recall, precision, color="steelblue", lw=2)
        ax.axvline(min_recall, color="gray", linestyle=":", lw=1, label=f"recall={min_recall}")
        ax.scatter(cand_rec, cand_prec, color="red", zorder=5, label=f"thr={cand_thr:.3f}")
        ax.set_title(f"{cls}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Precision-Recall Curves — max Precision @ recall≥{min_recall}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "03_pr_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")
    return optimal


def print_summary(cross_val, roc_opt, pr_opt):
    print("\n" + "=" * 72)
    print(f"{'Class':<14} {'ROC thr':>8} {'Youden J':>9} {'AUC':>6} │ {'PR thr':>8} {'Recall':>7} {'Prec':>7}")
    print("-" * 72)
    for cls in CLASS_NAMES:
        r = roc_opt[cls]
        p = pr_opt[cls]
        print(
            f"{cls:<14} {r['roc_threshold']:>8.4f} {r['youden_j']:>9.4f} "
            f"{r['auc']:>6.4f} │ {p['pr_threshold']:>8.4f} "
            f"{p['recall']:>7.4f} {p['precision']:>7.4f}"
        )
    print("=" * 72)
    if cross_val is not None:
        print(f"\n신뢰도 분포 교차점: {cross_val:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str,   required=True,
                   choices=list(MODEL_REGISTRY) + ["ensemble"] + list(PARTIAL_ENSEMBLE))
    p.add_argument("--weights",    type=str,   default=None)
    p.add_argument("--data_root",  type=str,   required=True)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--min_recall", type=float, default=0.95)
    p.add_argument("--out_dir",    type=str,   default="threshold_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device} | model: {args.model}")
    print(f"dataset: {args.data_root}")

    model, input_size, is_log = build_model(args.model, device, args.weights)
    loader = get_test_loader(args.data_root, input_size, args.batch_size)
    print(f"test samples: {len(loader.dataset)}")

    print("collecting probabilities...")
    all_probs, all_labels = collect_probs(model, loader, device, is_log)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n그래프 저장 중 → {args.out_dir}/")

    cross_val = plot_confidence_distribution(all_probs, all_labels, args.out_dir)
    roc_opt   = plot_roc_curves(all_probs, all_labels, args.out_dir)
    pr_opt    = plot_pr_curves(all_probs, all_labels, args.out_dir, args.min_recall)

    print_summary(cross_val, roc_opt, pr_opt)


if __name__ == "__main__":
    main()
