"""

사용법:
    python train.py --model resnet
    python train.py --model ensemble --weights_dir /path/to/pth

인자:
    --model(모델)             학습할 모델 이름 (기본값: resnet)
    --epochs(에폭)            전체 학습 반복 횟수 (기본값: 25)
    --batch_size(배치 크기)   한 번에 처리하는 이미지 수 (기본값: 32)
    --lr(학습률)              가중치를 얼마나 빠르게 갱신할지 (기본값: 1e-4)
    --num_workers(병렬 수)    데이터 로딩 병렬 처리 수 — Kaggle: 4, 로컬: 0 (기본값: 0)
    --save_dir(저장 경로)     학습된 모델 저장 폴더 (기본값: checkpoints)
    --weights_dir(가중치 경로) 앙상블 전용 — 단일 모델 .pth 파일이 있는 폴더 경로
"""

import argparse
import os
import torch
import torch.nn as nn

from data.dataset import get_dataloaders
from models.resnet           import build_resnet,           INPUT_SIZE as RES_SIZE
from models.densenet         import build_densenet,         INPUT_SIZE as DEN_SIZE
from models.efficientnet_v2  import build_efficientnetv2,   INPUT_SIZE as EFF_SIZE
from models.swin_transformer import build_swin_transformer, INPUT_SIZE as SWN_SIZE
from models.ensemble         import EnsembleModel, ALL_COMPONENTS
from utils.metrics           import evaluate

MODEL_REGISTRY = {
    "resnet":       (build_resnet,           RES_SIZE),
    "densenet":     (build_densenet,         DEN_SIZE),
    "efficientnet": (build_efficientnetv2,   EFF_SIZE),
    "swin":         (build_swin_transformer, SWN_SIZE),
}

# 앙상블은 내부에서 각 모델 크기로 리사이즈하므로 512×512 원본 입력
ENSEMBLE_INPUT_SIZE = (512, 512)

PARTIAL_ENSEMBLE_COMPONENTS = {
    "densenet_swin":     ("densenet",     "swin"),
    "efficientnet_swin": ("efficientnet", "swin"),
    "resnet_swin":       ("resnet",       "swin"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="resnet",
                        choices=["resnet", "densenet", "efficientnet", "swin",
                                 "ensemble",
                                 "densenet_swin", "efficientnet_swin", "resnet_swin"])
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--save_dir",    type=str,   default="checkpoints")
    parser.add_argument("--weights_dir", type=str,   default=None)
    return parser.parse_args()


def build_model(model_name: str, device: torch.device, weights_dir: str = None) -> tuple:
    if model_name == "ensemble":
        model = EnsembleModel(pretrained=True, weights_dir=weights_dir)
        input_size = ENSEMBLE_INPUT_SIZE
    elif model_name in PARTIAL_ENSEMBLE_COMPONENTS:
        components = PARTIAL_ENSEMBLE_COMPONENTS[model_name]
        model = EnsembleModel(pretrained=True, components=components, weights_dir=weights_dir)
        input_size = ENSEMBLE_INPUT_SIZE
    else:
        build_fn, input_size = MODEL_REGISTRY[model_name]
        model = build_fn(pretrained=True)
    return model.to(device), input_size


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total = len(loader)

    for i, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"\r  batch {i}/{total}  loss: {loss.item():.4f}", end="", flush=True)

    print()
    return total_loss / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                          if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} | model: {args.model}")

    model, input_size = build_model(args.model, device, args.weights_dir)
    loaders = get_dataloaders(
        input_size=input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    is_ensemble = args.model in PARTIAL_ENSEMBLE_COMPONENTS or args.model == "ensemble"
    # 앙상블은 로그 확률(log prob) 출력 → NLLLoss, 단일 모델은 교차 엔트로피(CrossEntropy) + 레이블 스무딩(label smoothing) 0.1
    criterion = nn.NLLLoss() if is_ensemble else nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW 옵티마이저: 가중치 감쇠(weight decay) 1e-4로 과적합 억제
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 코사인 스케줄러(CosineAnnealingLR): 학습률을 에폭마다 서서히 낮춰 최솟값 1e-6까지 감소
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], device)
        scheduler.step()

        m = val_metrics
        print(
            f"[{epoch:03d}/{args.epochs}]  lr: {scheduler.get_last_lr()[0]:.2e}\n"
            f"  loss:       {train_loss:.4f}\n"
            f"  accuracy:   {m['accuracy']:.4f}\n"
            f"  precision:  {m['precision']:.4f}\n"
            f"  recall:     {m['recall']:.4f}\n"
            f"  f1:         {m['f1_score']:.4f}\n"
            f"  confidence: {m['confidence']:.4f}\n"
            f"  ── 클래스별 ──────────────────────────────"
        )
        for cls, scores in m["per_class"].items():
            print(f"  {cls:<12} P: {scores['precision']:.4f}  R: {scores['recall']:.4f}  F1: {scores['f1']:.4f}")
        print()

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), save_path)
            print(f"  → best model saved (f1: {best_f1:.4f})\n")

    print("\n── Test Evaluation ──")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, loaders["test"], device)
    print(
        f"test_acc: {test_metrics['accuracy']:.4f} | "
        f"test_f1: {test_metrics['f1_score']:.4f} | "
        f"test_recall: {test_metrics['recall']:.4f} | "
        f"confidence: {test_metrics['confidence']:.4f}"
    )


if __name__ == "__main__":
    main()
