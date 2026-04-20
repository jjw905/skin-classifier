"""
이미지 + 라벨 JSON 통합 정리 스크립트

02_split_dataset.py 로 분리된 이미지와 원본 라벨 JSON을
질환별·split별로 매칭하여 dataset_labeled/ 에 저장한다.

경로:
    DATASET_IMG  02번 스크립트 출력 폴더 (split/클래스 구조)
    LABEL_BASE   원본 라벨 JSON 파일들이 있는 최상위 폴더
    OUTPUT       이미지+라벨 통합 결과를 저장할 폴더

출력 구조:
    dataset_labeled/
      {train|val|test}/
        {한국어}_{영문}/
          images/
          labels/
"""

import os
import shutil
from pathlib import Path

DATASET_IMG = "/Users/905jjw/4 Project/skin_classifier/dataset"
LABEL_BASE  = "/Users/905jjw/4 Project/Original_data/data/3.개방데이터/1.데이터"
OUTPUT      = "/Users/905jjw/4 Project/skin_classifier/dataset_labeled"

CLASS_MAP = {
    "psoriasis":  "건선",
    "atopy":      "아토피",
    "acne":       "여드름",
    "normal":     "정상",
    "rosacea":    "주사",
    "seborrheic": "지루",
}


def build_label_index() -> dict:
    index = {}
    for root, _, files in os.walk(LABEL_BASE):
        for f in files:
            if f.endswith(".json"):
                stem = Path(f).stem
                index[stem] = Path(root) / f
    return index


def reorganize():
    label_index = build_label_index()
    print(f"라벨 인덱스 구축 완료: {len(label_index)}개\n")

    for split in ("train", "val", "test"):
        split_dir = Path(DATASET_IMG) / split
        for cls_en, cls_kr in CLASS_MAP.items():
            src_img_dir = split_dir / cls_en
            if not src_img_dir.exists():
                continue

            folder_name = f"{cls_kr}_{cls_en}"
            out_img = Path(OUTPUT) / split / folder_name / "images"
            out_lbl = Path(OUTPUT) / split / folder_name / "labels"
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            matched, missing = 0, 0
            for img_file in src_img_dir.iterdir():
                if img_file.suffix.lower() != ".png":
                    continue
                shutil.copy2(img_file, out_img / img_file.name)
                stem = img_file.stem
                if stem in label_index:
                    shutil.copy2(label_index[stem], out_lbl / f"{stem}.json")
                    matched += 1
                else:
                    missing += 1

            print(f"[{split:5s}] {folder_name:20s} | 이미지 {matched+missing} | 라벨 매칭 {matched} | 미매칭 {missing}")

    print("\n완료")


if __name__ == "__main__":
    reorganize()
