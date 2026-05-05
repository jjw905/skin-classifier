from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "aihub_split"
OUT_ROOT = PROJECT_ROOT / "datasets" / "aihub_split_tone_balanced"

CLASSES = ["acne", "atopy", "normal", "psoriasis", "rosacea", "seborrheic"]
LEVELS = [1, 2, 3]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUG_RE = re.compile(r"^aug_\d+_(.+)$")
SEED = 20260503
TARGET_PER_LEVEL = 150
AUG_RATIO_CAP = 3.0
ITA_Q1 = 16.14  # L1/L2 boundary (historical, full-set analysis)
ITA_Q2 = 29.11  # L2/L3 boundary


def base_name(name: str) -> str:
    match = AUG_RE.match(name)
    return match.group(1) if match else name


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mask = (s > 18) & (v > 35) & (v < 248) & (spread > 10)
    mask[:8, :] = False
    mask[-8:, :] = False
    mask[:, :8] = False
    mask[:, -8:] = False
    if mask.mean() < 0.04:
        mask = (v > 35) & (v < 245)
    return mask


def ita_value(rgb: np.ndarray) -> float:
    mask = skin_mask(rgb)
    if mask.mean() < 0.02:
        return math.nan
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_vals = lab[:, :, 0][mask] * 100.0 / 255.0
    b_vals = lab[:, :, 2][mask] - 128.0
    l_med = float(np.median(l_vals))
    b_med = float(np.median(b_vals))
    return float(math.degrees(math.atan2(l_med - 50.0, b_med if abs(b_med) > 1e-6 else 1e-6)))


def level_from_ita(ita: float, q1: float, q2: float) -> int:
    if ita <= q1:
        return 1
    if ita <= q2:
        return 2
    return 3


def encode_png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def rotate(img: Image.Image, angle: float) -> Image.Image:
    arr = to_np(img)
    h, w = arr.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return to_pil(cv2.warpAffine(arr, mat, (w, h), borderMode=cv2.BORDER_REPLICATE))


def crop_resize(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.LANCZOS)


def gamma(arr: np.ndarray, g: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(arr, lut)


def hsv_adjust(arr: np.ndarray, sat_s: float, val_s: float) -> np.ndarray:
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_s, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def noise(arr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(arr.astype(np.float32) + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)


def aug_candidates(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> list[tuple[str, Image.Image]]:
    arr = to_np(img)
    out = [
        ("mirror", ImageOps.mirror(img)),
        ("rotate", rotate(img, rng.uniform(-7, 7))),
        ("crop", crop_resize(img, rng.uniform(0.93, 0.985))),
        ("mirror_rotate", rotate(ImageOps.mirror(img), rng.uniform(-6, 6))),
        ("crop_mirror", ImageOps.mirror(crop_resize(img, rng.uniform(0.94, 0.99)))),
        ("noise", to_pil(noise(arr, rng.uniform(2.5, 7.5), np_rng))),
        ("gamma_mild", to_pil(gamma(arr, rng.uniform(0.94, 1.06)))),
        ("sat_val_mild", to_pil(hsv_adjust(arr, rng.uniform(0.92, 1.08), rng.uniform(0.97, 1.03)))),
        ("rot_noise", to_pil(noise(to_np(rotate(img, rng.uniform(-5, 5))), rng.uniform(2.0, 6.0), np_rng))),
    ]
    rng.shuffle(out)
    return out


def scan_originals() -> list[dict]:
    """Scan non-aug_ files from SOURCE_ROOT/train and compute ITA inline."""
    rows: list[dict] = []
    for cls in CLASSES:
        cls_dir = SOURCE_ROOT / "train" / cls
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("aug_")
        )
        valid = 0
        for i, path in enumerate(files):
            ita = ita_value(read_rgb(path))
            if math.isnan(ita):
                continue
            rows.append({"class": cls, "path": path, "ita": ita, "level": level_from_ita(ita, ITA_Q1, ITA_Q2)})
            valid += 1
            print(f"  {cls}: {i + 1}/{len(files)}", end="\r", flush=True)
        print(f"  {cls}: {valid}/{len(files)} valid                ", flush=True)
    return rows


def ensure_output_safe(out_root: Path) -> None:
    resolved = out_root.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"출력 경로가 프로젝트 밖임: {out_root}")
    if out_root.exists():
        raise RuntimeError(f"출력 폴더가 이미 있음. 안전상 덮어쓰지 않음: {out_root}")


def prepare_split_dirs(out_root: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (out_root / split / cls).mkdir(parents=True, exist_ok=False)


def copy_eval_splits(out_root: Path) -> None:
    for split in ("val", "test"):
        for cls in CLASSES:
            src_dir = SOURCE_ROOT / split / cls
            for src in sorted(src_dir.iterdir()):
                if src.is_file() and src.suffix.lower() in IMG_EXTS:
                    link_or_copy(src, out_root / split / cls / src.name)


def make_aug_name(index: int, source_name: str) -> str:
    return f"aug_{900000 + index:06d}_{base_name(source_name)}"


def build_train(
    rows: list[dict],
    out_root: Path,
    q1: float,
    q2: float,
) -> dict[tuple[str, int], dict]:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    pools: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        pools[(row["class"], row["level"])].append(row)
    for pool in pools.values():
        rng.shuffle(pool)

    used_md5: set[str] = set()
    new_index = 0
    stats: dict[tuple[str, int], dict] = {}

    for cls in CLASSES:
        for level in LEVELS:
            pool = pools[(cls, level)]
            n = len(pool)
            actual_target = min(TARGET_PER_LEVEL, math.ceil(n * AUG_RATIO_CAP))
            copy_count = min(n, actual_target)

            for row in pool[:copy_count]:
                src_path: Path = row["path"]
                link_or_copy(src_path, out_root / "train" / cls / src_path.name)
                used_md5.add(hashlib.md5(src_path.read_bytes()).hexdigest())

            need = actual_target - copy_count
            aug_saved = 0

            if need > 0:
                if not pool:
                    raise RuntimeError(f"증강 소스 없음: {cls} L{level}")
                attempts = 0
                max_attempts = need * 600
                while aug_saved < need and attempts < max_attempts:
                    src_row = pool[attempts % len(pool)]
                    attempts += 1
                    src_path = src_row["path"]
                    with Image.open(src_path) as im:
                        base_img = ImageOps.exif_transpose(im).convert("RGB")
                    for _method, candidate in aug_candidates(base_img, rng, np_rng):
                        ita = ita_value(to_np(candidate))
                        if math.isnan(ita) or level_from_ita(ita, q1, q2) != level:
                            continue
                        data = encode_png(candidate)
                        digest = hashlib.md5(data).hexdigest()
                        if digest in used_md5:
                            continue
                        out_name = make_aug_name(new_index, src_path.name)
                        new_index += 1
                        out_path = out_root / "train" / cls / out_name
                        if out_path.exists():
                            continue
                        out_path.write_bytes(data)
                        used_md5.add(digest)
                        aug_saved += 1
                        break
                if aug_saved != need:
                    raise RuntimeError(f"증강 생성 부족: {cls} L{level} {aug_saved}/{need}")

            total = copy_count + aug_saved
            stats[(cls, level)] = {"pool": n, "target": actual_target, "copy": copy_count, "aug": aug_saved, "total": total}
            print(f"{cls} L{level}: pool={n} target={actual_target} copy={copy_count} aug={aug_saved} total={total}", flush=True)

    print(f"\ntotal new aug created: {new_index}", flush=True)
    return stats


def print_summary(stats: dict[tuple[str, int], dict]) -> None:
    print("\ntone-balanced train summary:", flush=True)
    for cls in CLASSES:
        cls_total = sum(stats[(cls, lv)]["total"] for lv in LEVELS)
        parts = []
        for level in LEVELS:
            s = stats[(cls, level)]
            ratio = s["total"] / cls_total * 100 if cls_total else 0.0
            parts.append(f"L{level}={s['total']}({ratio:.0f}%)")
        print(f"  {cls}: " + " ".join(parts) + f"  total={cls_total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    ensure_output_safe(out_root)

    print(f"source: {SOURCE_ROOT.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"out: {out_root.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"target_per_level={TARGET_PER_LEVEL}  aug_ratio_cap={AUG_RATIO_CAP}", flush=True)

    print(f"level boundaries: L1 <= {ITA_Q1:.2f}, L2 <= {ITA_Q2:.2f}, L3 > {ITA_Q2:.2f}", flush=True)
    print("scanning originals (ITA computation may take several minutes)...", flush=True)
    rows = scan_originals()
    print(f"total orig rows: {len(rows)}", flush=True)

    prepare_split_dirs(out_root)
    copy_eval_splits(out_root)
    stats = build_train(rows, out_root, ITA_Q1, ITA_Q2)
    print_summary(stats)


if __name__ == "__main__":
    main()
