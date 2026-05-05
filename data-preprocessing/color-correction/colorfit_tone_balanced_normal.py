from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "datasets" / "aihub_split_tone_balanced"
OUT_ROOT = PROJECT_ROOT / "datasets" / "aihub_split_tone_balanced_colorfit"

CLASSES = ["acne", "atopy", "normal", "psoriasis", "rosacea", "seborrheic"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ITA_L1_MAX = 16.14


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
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(path, quality=95)
    else:
        img.save(path)


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


def skin_stats(rgb: np.ndarray) -> dict[str, float]:
    mask = skin_mask(rgb)
    if mask.mean() < 0.02:
        raise ValueError(f"skin_mask_too_small:{mask.mean():.4f}")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_vals = lab[:, :, 0][mask] * 100.0 / 255.0
    a_vals = lab[:, :, 1][mask] - 128.0
    b_vals = lab[:, :, 2][mask] - 128.0
    l_med = float(np.median(l_vals))
    a_med = float(np.median(a_vals))
    b_med = float(np.median(b_vals))
    ita = float(math.degrees(math.atan2(l_med - 50.0, b_med if abs(b_med) > 1e-6 else 1e-6)))
    return {
        "ita": ita,
        "L": l_med,
        "a": a_med,
        "b": b_med,
        "skin_mask_ratio": float(mask.mean()),
    }


def robust_reference(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    ref: dict[str, dict[str, float]] = {}
    for key in ("ita", "L", "a", "b"):
        vals = np.array([r[key] for r in rows], dtype=np.float32)
        ref[key] = {
            "min": float(vals.min()),
            "p05": float(np.percentile(vals, 5)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max()),
        }
    return ref


def collect_train_stats(root: Path) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {cls: [] for cls in CLASSES}
    for cls in CLASSES:
        cls_dir = root / "train" / cls
        for path in sorted(cls_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                out[cls].append(skin_stats(read_rgb(path)))
    return out


def soft_skin_alpha(rgb: np.ndarray) -> np.ndarray:
    mask = skin_mask(rgb).astype(np.float32)
    alpha = cv2.GaussianBlur(mask, (61, 61), 0)
    return np.clip(alpha[:, :, None], 0.0, 1.0)


def clipped_delta(value: float, target: float, strength: float, limit: float) -> float:
    return float(np.clip((target - value) * strength, -limit, limit))


def colorfit_normal(rgb: np.ndarray, ref: dict[str, dict[str, float]], strength: float) -> tuple[np.ndarray, dict[str, float]]:
    before = skin_stats(rgb)
    if before["ita"] <= ITA_L1_MAX:
        after = dict(before)
        after.update({"delta_L": 0.0, "delta_a": 0.0, "delta_b": 0.0})
        return rgb, after

    target_l = float(np.clip(ref["L"]["median"], ref["L"]["p05"], ref["L"]["p95"]))
    target_a = float(np.clip(ref["a"]["median"], ref["a"]["p05"], ref["a"]["p95"]))
    target_b = float(np.clip(ref["b"]["median"], ref["b"]["p05"], ref["b"]["p95"]))

    d_l = min(0.0, clipped_delta(before["L"], target_l, strength, 8.0))
    d_a = clipped_delta(before["a"], target_a, strength, 3.0)
    d_b = clipped_delta(before["b"], target_b, strength, 6.0)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] += d_l * 255.0 / 100.0
    lab[:, :, 1] += d_a
    lab[:, :, 2] += d_b
    transferred = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    alpha = soft_skin_alpha(rgb)
    adjusted = rgb.astype(np.float32) * (1.0 - alpha) + transferred * alpha
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    after = skin_stats(adjusted)
    after.update({"delta_L": d_l, "delta_a": d_a, "delta_b": d_b})
    return adjusted, after


def ensure_output_safe(out_root: Path) -> None:
    resolved = out_root.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"출력 경로가 프로젝트 밖임: {out_root}")
    if out_root.exists():
        raise RuntimeError(f"출력 폴더가 이미 있음. 덮어쓰지 않음: {out_root}")


def build_colorfit(src_root: Path, out_root: Path, strength: float) -> dict[str, object]:
    ensure_output_safe(out_root)
    stats_by_class = collect_train_stats(src_root)
    non_normal = [row for cls, rows in stats_by_class.items() if cls != "normal" for row in rows]
    ref = robust_reference(non_normal)

    manifest: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            src_dir = src_root / split / cls
            for src in sorted(src_dir.iterdir()):
                if not src.is_file() or src.suffix.lower() not in IMG_EXTS:
                    continue
                dst = out_root / split / cls / src.name
                if split == "train" and cls == "normal":
                    rgb = read_rgb(src)
                    before = skin_stats(rgb)
                    adjusted, after = colorfit_normal(rgb, ref, strength)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    save_rgb(adjusted, dst)
                    manifest.append(
                        {
                            "path": str(dst.relative_to(PROJECT_ROOT)),
                            "source_path": str(src.relative_to(PROJECT_ROOT)),
                            "before": before,
                            "after": after,
                        }
                    )
                else:
                    link_or_copy(src, dst)

    metadata = {
        "source_root": str(src_root.relative_to(PROJECT_ROOT)),
        "output_root": str(out_root.relative_to(PROJECT_ROOT)),
        "normal_train_adjusted": len(manifest),
        "strength": strength,
        "reference_non_normal_train": ref,
        "method": "skin-mask LAB median shift toward non-normal median, with per-channel delta limits",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "colorfit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out_root / "colorfit_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    make_preview(manifest, out_root / "normal_colorfit_preview.jpg")
    return metadata


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_preview(manifest: list[dict[str, object]], out_path: Path) -> None:
    if not manifest:
        return
    rows = sorted(
        manifest,
        key=lambda r: (
            float(r["before"]["ita"]),
            float(r["before"]["L"]),
        ),
    )
    picks = []
    if len(rows) <= 8:
        picks = rows
    else:
        idxs = np.linspace(0, len(rows) - 1, 8).round().astype(int)
        picks = [rows[int(i)] for i in idxs]

    cell = 170
    label_h = 38
    canvas = Image.new("RGB", (cell * 2, (cell + label_h) * len(picks) + 28), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    font = load_font(12)
    draw.text((cell // 2, 14), "before", fill=(30, 30, 30), font=font, anchor="mm")
    draw.text((cell + cell // 2, 14), "colorfit", fill=(30, 30, 30), font=font, anchor="mm")
    for row_idx, row in enumerate(picks):
        y = 28 + row_idx * (cell + label_h)
        paths = [PROJECT_ROOT / str(row["source_path"]), PROJECT_ROOT / str(row["path"])]
        labels = [
            f"ITA {float(row['before']['ita']):.1f} L {float(row['before']['L']):.1f}",
            f"ITA {float(row['after']['ita']):.1f} L {float(row['after']['L']):.1f}",
        ]
        for col, path in enumerate(paths):
            x = col * cell
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
                canvas.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
            draw.rectangle([x, y + cell, x + cell, y + cell + label_h], fill=(248, 248, 248))
            draw.text((x + 4, y + cell + 5), labels[col], fill=(0, 0, 0), font=font)
    canvas.save(out_path, quality=92)


def print_summary(out_root: Path) -> None:
    stats = collect_train_stats(out_root)
    print("colorfit train summary")
    for cls in CLASSES:
        vals = stats[cls]
        print(f"{cls}: n={len(vals)}")
        for key in ("ita", "L", "a", "b"):
            arr = np.array([v[key] for v in vals], dtype=np.float32)
            print(
                f"  {key}: min={arr.min():.1f} p05={np.percentile(arr, 5):.1f} "
                f"med={np.median(arr):.1f} p95={np.percentile(arr, 95):.1f} max={arr.max():.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default=str(SRC_ROOT))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--strength", type=float, default=0.55)
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    metadata = build_colorfit(src_root, out_root, args.strength)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print_summary(out_root)


if __name__ == "__main__":
    main()
