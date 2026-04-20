"""
데이터셋 Train / Val / Test 분리 스크립트

Training + Validation 원천데이터를 클래스별로 모아
8:1:1 비율로 나눠 ImageFolder 형식으로 저장한다.

경로:
    BASE_DATA(원본 경로)   원본 이미지가 있는 최상위 폴더 (Training/, Validation/ 하위 폴더 포함)
    OUTPUT_DIR(출력 경로)  분리된 데이터를 저장할 폴더

설정:
    RANDOM_SEED(시드값)    데이터 분리 결과를 재현하기 위한 고정 시드값 (기본값: 42)
    CLASS_MAP(클래스 매핑) 한국어 질환명 → 영문 폴더명 변환표
    IMAGE_EXTS(확장자)     처리할 이미지 확장자 목록

출력 구조:
    dataset/
      train/{클래스명}/
      val/{클래스명}/
      test/{클래스명}/
"""

import os
import shutil
import random
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DATA  = "/Users/905jjw/4 Project/Original_data/data/3.개방데이터/1.데이터"
OUTPUT_DIR = "/Users/905jjw/4 Project/skin_classifier/dataset"

CLASS_MAP = {
    "건선": "psoriasis",
    "아토피": "atopy",
    "여드름": "acne",
    "정상": "normal",
    "주사": "rosacea",
    "지루": "seborrheic",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def collect_images_by_class() -> dict:
    class_images: dict[str, list] = {k: [] for k in CLASS_MAP}

    for split in ["Training", "Validation"]:
        src_dir = Path(BASE_DATA) / split / "01.원천데이터"
        if not src_dir.exists():
            print(f"경고: {src_dir} 없음")
            continue

        for folder in src_dir.iterdir():
            if not folder.is_dir():
                continue
            parts = folder.name.split("_")
            if len(parts) < 2:
                continue
            class_name = parts[1]
            if class_name not in class_images:
                continue
            for img_file in folder.iterdir():
                if img_file.suffix.lower() in IMAGE_EXTS:
                    class_images[class_name].append(img_file)

    return class_images


def split_and_copy(class_images: dict):
    stats = {}

    for class_kr, images in class_images.items():
        class_en = CLASS_MAP[class_kr]
        random.shuffle(images)

        total   = len(images)
        n_train = int(total * 0.8)
        n_val   = int(total * 0.1)
        n_test  = total - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dest = Path(OUTPUT_DIR) / split_name / class_en
            dest.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, dest / src.name)

        stats[class_kr] = {"total": total, "train": n_train, "val": n_val, "test": n_test}
        print(f"  [{class_kr:6s}] 전체 {total:4d}  →  train {n_train}  val {n_val}  test {n_test}")

    return stats


def main():
    print("=== 이미지 수집 ===")
    class_images = collect_images_by_class()
    for k, v in class_images.items():
        print(f"  {k}: {len(v)}장")

    print(f"\n=== 8:1:1 분리 → {OUTPUT_DIR} ===")
    stats = split_and_copy(class_images)

    total_train = sum(s["train"] for s in stats.values())
    total_val   = sum(s["val"]   for s in stats.values())
    total_test  = sum(s["test"]  for s in stats.values())
    print(f"\n총계: train {total_train} / val {total_val} / test {total_test}")


if __name__ == "__main__":
    main()
