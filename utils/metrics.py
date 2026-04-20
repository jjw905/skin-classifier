"""
모델 평가 함수 — 전체 및 클래스별 성능 지표 계산

반환값 (evaluate):
    accuracy(정확도)          전체 정확도
    precision(정밀도)         예측이 맞을 확률 (클래스 평균)
    recall(재현율)            실제 질환을 빠뜨리지 않는 비율 (클래스 평균)
    f1_score(F1 점수)         정밀도와 재현율의 조화 평균 (클래스 평균)
    confidence(신뢰도)        모델이 예측에 확신하는 평균 점수
    per_class(클래스별 지표)  위 지표를 각 질환 클래스별로 세분화한 딕셔너리
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from models.base_model import CLASS_NAMES


def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    is_log_probs = getattr(model, "output_log_probs", False)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs  = torch.exp(logits) if is_log_probs else F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.max(dim=1).values.cpu().tolist())

    result = {
        "accuracy":   round(accuracy_score(all_labels, all_preds), 4),
        "precision":  round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "recall":     round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "f1_score":   round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "confidence": round(sum(all_probs) / len(all_probs), 4),
    }

    per_class_p = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f = f1_score(all_labels, all_preds, average=None, zero_division=0)

    result["per_class"] = {
        name: {
            "precision": round(float(per_class_p[i]), 4),
            "recall":    round(float(per_class_r[i]), 4),
            "f1":        round(float(per_class_f[i]), 4),
        }
        for i, name in enumerate(CLASS_NAMES)
    }

    return result
