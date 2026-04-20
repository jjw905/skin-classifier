"""
Swin Transformer-Tiny 분류 모델 (torchvision 제공 사전학습 모델)

인자:
    num_classes(클래스 수)  분류할 질환 수 (기본값: 6)
    pretrained(사전학습)    ImageNet으로 미리 학습된 가중치 사용 여부 (기본값: True)

구조:
    INPUT_SIZE(입력 크기)  입력 이미지 크기 256×256 픽셀
                           (윈도우 크기(window_size)=7, 패치 크기(patch_size)=4 조건상 32의 배수여야 함)
    Dropout(드롭아웃)      마지막 분류층 직전, 30% 뉴런을 무작위로 꺼서 과적합 방지
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import Swin_T_Weights
from models.base_model import NUM_CLASSES

INPUT_SIZE = (256, 256)


def build_swin_transformer(num_classes: int = NUM_CLASSES,
                            pretrained: bool = True) -> nn.Module:
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
