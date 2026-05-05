"""
EfficientNetV2-S 분류 모델 (torchvision 제공 사전학습 모델)

인자:
    num_classes(클래스 수)  분류할 질환 수 (기본값: 6)
    pretrained(사전학습)    ImageNet으로 미리 학습된 가중치 사용 여부 (기본값: True)

구조:
    INPUT_SIZE(입력 크기)  입력 이미지 크기 384×384 픽셀
    Dropout(드롭아웃)      마지막 분류층 직전, 30% 뉴런을 무작위로 꺼서 과적합 방지
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from models.base_model import NUM_CLASSES

INPUT_SIZE = (384, 384)


def build_efficientnetv2(num_classes: int = NUM_CLASSES,
                         pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
