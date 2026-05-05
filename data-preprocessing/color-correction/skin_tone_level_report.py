from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "datasets" / "aihub_split" / "train"
STRICT_FAIL_JSON = PROJECT_ROOT / "reports" / "aihub_aug_strict_qc" / "strict_aug_failures.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "skin_tone_levels"
PER_IMAGE_JSON = REPORT_DIR / "skin_tone_per_image.json"
SUMMARY_JSON = REPORT_DIR / "skin_tone_level_summary.json"
SUMMARY_TXT = REPORT_DIR / "skin_tone_level_summary.txt"
REPORT_MD = REPORT_DIR / "skin_tone_level_report.md"
SHEET = REPORT_DIR / "skin_tone_level_sheet.jpg"

CLASSES = ["acne", "atopy", "normal", "psoriasis", "rosacea", "seborrheic"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEVELS = [1, 2, 3]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_excluded() -> set[Path]:
    if not STRICT_FAIL_JSON.exists():
        return set()
    excluded: set[Path] = set()
    for row in json.loads(STRICT_FAIL_JSON.read_text()):
        excluded.add((PROJECT_ROOT / row["aug_path"]).resolve())
    return excluded


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
    return mask


def skin_stats(path: Path) -> dict[str, float]:
    rgb = read_rgb(path)
    mask = skin_mask(rgb)
    if mask.mean() < 0.02:
        raise RuntimeError(f"skin mask too small: {path.relative_to(PROJECT_ROOT)} ratio={mask.mean():.4f}")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_vals = lab[:, :, 0][mask] * 100.0 / 255.0
    a_vals = lab[:, :, 1][mask] - 128.0
    b_vals = lab[:, :, 2][mask] - 128.0
    l_med = float(np.median(l_vals))
    a_med = float(np.median(a_vals))
    b_med = float(np.median(b_vals))
    b_for_ita = b_med if abs(b_med) > 1e-6 else 1e-6
    ita = float(math.degrees(math.atan2(l_med - 50.0, b_for_ita)))
    return {
        "ita": ita,
        "lab_l_median": l_med,
        "lab_a_median": a_med,
        "lab_b_median": b_med,
        "skin_mask_ratio": float(mask.mean()),
    }


def collect_rows(excluded: set[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cls in CLASSES:
        cls_dir = TRAIN_ROOT / cls
        if not cls_dir.exists():
            raise FileNotFoundError(cls_dir)
        for path in sorted(cls_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                continue
            if path.resolve() in excluded:
                continue
            stat = skin_stats(path)
            rows.append(
                {
                    "class": cls,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "is_aug": path.name.startswith("aug_"),
                    **stat,
                }
            )
    return rows


def assign_levels(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = np.array([float(r["ita"]) for r in rows], dtype=np.float32)
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    for row in rows:
        ita = float(row["ita"])
        if ita <= q1:
            level = 1
        elif ita <= q2:
            level = 2
        else:
            level = 3
        row["level"] = level
    return float(q1), float(q2)


def level_interval(level: int, q1: float, q2: float) -> str:
    if level == 1:
        return f"ITA <= {q1:.2f}"
    if level == 2:
        return f"{q1:.2f} < ITA <= {q2:.2f}"
    return f"ITA > {q2:.2f}"


def summarize(rows: list[dict[str, object]], q1: float, q2: float) -> list[dict[str, object]]:
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class"])].append(row)

    summary: list[dict[str, object]] = []
    for cls in CLASSES:
        class_rows = by_class[cls]
        denom = len(class_rows)
        for level in LEVELS:
            level_rows = [r for r in class_rows if int(r["level"]) == level]
            count = len(level_rows)
            ratio = count / denom * 100.0 if denom else 0.0
            summary.append(
                {
                    "class": cls,
                    "level": level,
                    "interval": level_interval(level, q1, q2),
                    "count": count,
                    "ratio_percent": ratio,
                    "median_ita": float(np.median([float(r["ita"]) for r in level_rows])) if level_rows else "",
                }
            )
    return summary


def representative(rows: list[dict[str, object]], cls: str, level: int) -> dict[str, object] | None:
    candidates = [r for r in rows if r["class"] == cls and int(r["level"]) == level]
    if not candidates:
        return None
    target = float(np.median([float(r["ita"]) for r in candidates]))
    originals = [r for r in candidates if not bool(r["is_aug"])]
    pool = originals or candidates
    return min(pool, key=lambda r: abs(float(r["ita"]) - target))


def write_outputs(rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PER_IMAGE_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["class level count ratio_percent median_ita"]
    for row in summary:
        lines.append(
            f"{row['class']} L{row['level']} {row['count']} "
            f"{float(row['ratio_percent']):.1f}% {row['median_ita']}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")


def make_report(rows: list[dict[str, object]], summary: list[dict[str, object]], q1: float, q2: float) -> None:
    lines = [
        "# Skin Tone Level Report",
        "",
        "대상: strict QC 실패 aug를 제외한 `datasets/aihub_split/train`.",
        "피부톤 지표: 피부 마스크 영역의 CIELAB median 기반 ITA. 값이 낮을수록 상대적으로 어둡고, 높을수록 밝음.",
        "",
        "## Level Intervals",
        f"- Level 1: ITA <= {q1:.2f}",
        f"- Level 2: {q1:.2f} < ITA <= {q2:.2f}",
        f"- Level 3: ITA > {q2:.2f}",
        "",
        "## Class Ratios",
        "| class | L1 count/% | L2 count/% | L3 count/% | total |",
        "|---|---:|---:|---:|---:|",
    ]
    by_key = {(str(r["class"]), int(r["level"])): r for r in summary}
    totals = defaultdict(int)
    for r in rows:
        totals[str(r["class"])] += 1
    for cls in CLASSES:
        cells = []
        for level in LEVELS:
            item = by_key[(cls, level)]
            cells.append(f"{int(item['count'])} / {float(item['ratio_percent']):.1f}%")
        lines.append(f"| {cls} | {cells[0]} | {cells[1]} | {cells[2]} | {totals[cls]} |")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def make_sheet(rows: list[dict[str, object]], summary: list[dict[str, object]], q1: float, q2: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = load_font(24, bold=True)
    header_font = load_font(18, bold=True)
    label_font = load_font(14)
    small_font = load_font(12)

    row_label_w = 150
    cell_w = 270
    thumb = 210
    label_h = 82
    header_h = 104
    row_h = thumb + label_h
    canvas = Image.new("RGB", (row_label_w + cell_w * 3, header_h + row_h * len(CLASSES)), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((16, 14), "Skin tone levels by class", fill=(0, 0, 0), font=title_font)
    draw.text(
        (16, 48),
        f"Strict-pass train / ITA intervals: L1 <= {q1:.2f}, L2 <= {q2:.2f}, L3 > {q2:.2f}",
        fill=(60, 60, 60),
        font=label_font,
    )

    level_titles = ["Level 1 darker", "Level 2 middle", "Level 3 lighter"]
    for idx, title in enumerate(level_titles):
        x = row_label_w + idx * cell_w
        draw.rectangle([x, 72, x + cell_w, header_h], fill=(238, 238, 238), outline=(210, 210, 210))
        draw.text((x + 10, 78), title, fill=(0, 0, 0), font=header_font)

    summary_by_key = {(str(r["class"]), int(r["level"])): r for r in summary}
    for row_idx, cls in enumerate(CLASSES):
        y = header_h + row_idx * row_h
        draw.rectangle([0, y, row_label_w, y + row_h], fill=(245, 245, 245), outline=(220, 220, 220))
        draw.text((14, y + 18), cls, fill=(0, 0, 0), font=header_font)
        for level_idx, level in enumerate(LEVELS):
            x = row_label_w + level_idx * cell_w
            draw.rectangle([x, y, x + cell_w, y + row_h], outline=(225, 225, 225))
            rep = representative(rows, cls, level)
            item = summary_by_key[(cls, level)]
            count = int(item["count"])
            ratio = float(item["ratio_percent"])
            if rep is not None:
                path = PROJECT_ROOT / str(rep["path"])
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                    px = x + (cell_w - im.width) // 2
                    py = y + 8 + (thumb - im.height) // 2
                    canvas.paste(im, (px, py))
                base = Path(str(rep["path"])).name
                if len(base) > 28:
                    base = base[:25] + "..."
                draw.text((x + 10, y + thumb + 14), f"{count} / {ratio:.1f}%   ITA {float(rep['ita']):.1f}", fill=(0, 0, 0), font=label_font)
                draw.text((x + 10, y + thumb + 38), base, fill=(80, 80, 80), font=small_font)
            else:
                draw.text((x + 70, y + 96), "No sample", fill=(120, 120, 120), font=header_font)
                draw.text((x + 10, y + thumb + 14), f"{count} / {ratio:.1f}%", fill=(0, 0, 0), font=label_font)

    canvas.save(SHEET, quality=94)


def main() -> None:
    excluded = load_excluded()
    rows = collect_rows(excluded)
    q1, q2 = assign_levels(rows)
    summary = summarize(rows, q1, q2)
    write_outputs(rows, summary)
    make_report(rows, summary, q1, q2)
    make_sheet(rows, summary, q1, q2)

    print("Skin tone level report")
    print(f"strict_excluded_aug_files: {len(excluded)}")
    print(f"total_images: {len(rows)}")
    print(f"Level 1: ITA <= {q1:.2f}")
    print(f"Level 2: {q1:.2f} < ITA <= {q2:.2f}")
    print(f"Level 3: ITA > {q2:.2f}")
    for cls in CLASSES:
        parts = []
        total = sum(1 for row in rows if row["class"] == cls)
        for level in LEVELS:
            count = sum(1 for row in rows if row["class"] == cls and int(row["level"]) == level)
            parts.append(f"L{level}={count}/{total} ({count / total * 100:.1f}%)")
        print(f"{cls}: " + ", ".join(parts))
    print(f"json: {PER_IMAGE_JSON.relative_to(PROJECT_ROOT)}")
    print(f"summary_json: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"summary_txt: {SUMMARY_TXT.relative_to(PROJECT_ROOT)}")
    print(f"report: {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"sheet: {SHEET.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
