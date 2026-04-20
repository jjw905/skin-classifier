"""
소프트 보팅 앙상블 모델 — 여러 모델의 예측 확률을 평균하여 최종 분류

인자:
    components(구성 모델)    앙상블에 포함할 모델 이름 목록 (기본값: 4개 전부)
    pretrained(사전학습)     각 모델의 ImageNet 사전학습 가중치 사용 여부 (기본값: True)
    weights_dir(가중치 경로) 개별 학습된 .pth 파일이 있는 폴더 경로
                             (지정 시 ImageNet 가중치 대신 해당 파일 로드)

입출력:
    입력   512×512 이미지 — 내부에서 각 모델 크기로 자동 리사이즈
    출력   로그 확률값(log prob) — NLLLoss 사용 / torch.exp() 로 일반 확률 변환 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.efficientnet_v2   import build_efficientnetv2,    INPUT_SIZE as EFF_SIZE
from models.resnet             import build_resnet,             INPUT_SIZE as RES_SIZE
from models.densenet           import build_densenet,           INPUT_SIZE as DEN_SIZE
from models.swin_transformer   import build_swin_transformer,   INPUT_SIZE as SWN_SIZE
from models.base_model         import NUM_CLASSES

_BUILDERS = {
    "efficientnet": (build_efficientnetv2,   EFF_SIZE),
    "resnet":       (build_resnet,            RES_SIZE),
    "densenet":     (build_densenet,          DEN_SIZE),
    "swin":         (build_swin_transformer,  SWN_SIZE),
}

ALL_COMPONENTS = ("efficientnet", "resnet", "densenet", "swin")

WEIGHT_FILENAMES = {
    "efficientnet": "efficientnet_best.pth",
    "resnet":       "resnet_best.pth",
    "densenet":     "densenet_best.pth",
    "swin":         "swin_best.pth",
}


class EnsembleModel(nn.Module):
    output_log_probs: bool = True  # 로그 확률(log prob) 출력 — NLLLoss 사용을 위해 필요

    def __init__(self, pretrained: bool = True,
                 components: tuple = ALL_COMPONENTS,
                 weights_dir: str = None):
        super().__init__()
        self._components = components
        self._sizes = {}
        for name in components:
            build_fn, size = _BUILDERS[name]
            model = build_fn(pretrained=pretrained)
            if weights_dir:
                import os
                path = os.path.join(weights_dir, WEIGHT_FILENAMES[name])
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                print(f"  loaded {path}")
            setattr(self, name, model)
            self._sizes[name] = size

    def _resize(self, x: torch.Tensor, size: tuple) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [
            F.softmax(getattr(self, name)(self._resize(x, self._sizes[name])), dim=1)
            for name in self._components
        ]
        avg = torch.stack(outs, dim=0).mean(dim=0)
        return torch.log(avg.clamp(min=1e-8))
