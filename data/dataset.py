"""
데이터셋 로더 및 전처리 (Kaggle 경로 기준)

상수:
    DATASET_ROOT(데이터셋 경로)  이미지 폴더 최상위 경로 (train / val / test 하위 폴더 포함)
    MEAN / STD(평균·표준편차)    ImageNet 기준 픽셀 정규화 값 (채널별 평균·표준편차)

get_transforms(split, input_size):
    split(분할)       "train" | "val" | "test"
    input_size(입력 크기)  모델에 입력할 이미지 크기 (픽셀 단위, 기본값: 224×224)

get_dataloaders(input_size, batch_size, num_workers):
    input_size(입력 크기)    모델에 입력할 이미지 크기
    batch_size(배치 크기)    한 번에 처리하는 이미지 수 (기본값: 32)
    num_workers(병렬 수)     데이터 로딩 병렬 처리 수 — Kaggle: 4, 로컬: 0 (기본값: 4)
"""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_ROOT = "/kaggle/input/skin-classifier/dataset"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, input_size: tuple = (224, 224)) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((input_size[0] + 32, input_size[1] + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            # hue(색조) 변화는 최소화 — 피부색이 질환 판단 기준이 되므로
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            # 머리카락·옷 등으로 병변 일부가 가려지는 상황을 시뮬레이션
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.0)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


def get_dataloaders(input_size: tuple = (224, 224),
                    batch_size: int = 32,
                    num_workers: int = 4) -> dict:
    import torch
    root = Path(DATASET_ROOT)
    pin = torch.cuda.is_available()  # 핀 메모리(pin_memory): GPU 사용 시 데이터 전송 속도 향상
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(root / split, transform=get_transforms(split, input_size))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin,
        )
    return loaders
