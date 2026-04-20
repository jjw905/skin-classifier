"""
원본 데이터 ZIP 압축 해제 및 폴더 교체 스크립트

BASE 경로 하위의 모든 .zip 파일을 같은 위치에 압축 해제하고
기존 폴더가 있으면 덮어쓴 뒤 zip 파일을 삭제한다.

경로:
    BASE  원본 zip 파일들이 있는 최상위 폴더 경로
"""

import zipfile
import os
import shutil

BASE = "/Users/905jjw/4 Project/Original_data/data/3.개방데이터/1.데이터"


def unzip_and_replace(zip_path: str):
    parent_dir = os.path.dirname(zip_path)
    folder_name = os.path.splitext(os.path.basename(zip_path))[0]
    target_dir = os.path.join(parent_dir, folder_name)

    print(f"압축 해제: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if target_dir and os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        tmp_dir = target_dir + "__tmp"
        zf.extractall(tmp_dir)

        # zip 내부가 폴더 하나로 감싸여 있으면 꺼내고, 아니면 그대로 이동
        extracted_items = os.listdir(tmp_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmp_dir, extracted_items[0])):
            inner = os.path.join(tmp_dir, extracted_items[0])
            shutil.move(inner, target_dir)
            shutil.rmtree(tmp_dir)
        else:
            shutil.move(tmp_dir, target_dir)

    os.remove(zip_path)
    file_count = sum(len(files) for _, _, files in os.walk(target_dir))
    print(f"  -> {target_dir} ({file_count}개 파일)")


def main():
    zip_files = []
    for root, dirs, files in os.walk(BASE):
        for f in files:
            if f.endswith('.zip'):
                zip_files.append(os.path.join(root, f))

    print(f"총 {len(zip_files)}개 zip 파일 처리 시작\n")
    for zp in sorted(zip_files):
        unzip_and_replace(zp)

    print("\n모든 zip 압축 해제 및 교체 완료")


if __name__ == "__main__":
    main()
