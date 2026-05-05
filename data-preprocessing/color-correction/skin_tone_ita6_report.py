from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_JSON = PROJECT_ROOT / "reports" / "skin_tone_levels" / "skin_tone_per_image.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "skin_tone_ita6"
SUMMARY_JSON = REPORT_DIR / "skin_tone_ita6_summary.json"
SUMMARY_TXT = REPORT_DIR / "skin_tone_ita6_summary.txt"
REPORT_MD = REPORT_DIR / "skin_tone_ita6_report.md"
SHEET = REPORT_DIR / "skin_tone_ita6_sheet.jpg"

CLASSES = ["acne", "atopy", "normal", "psoriasis", "rosacea", "seborrheic"]
CATEGORIES = [
    ("dark", "ITA <= -30"),
    ("brown", "-30 < ITA <= 10"),
    ("tan", "10 < ITA <= 28"),
    ("intermediate", "28 < ITA <= 41"),
    ("light", "41 < ITA <= 55"),
    ("very_light", "ITA > 55"),
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def category(ita: float) -> str:
    if ita <= -30:
        return "dark"
    if ita <= 10:
        return "brown"
    if ita <= 28:
        return "tan"
    if ita <= 41:
        return "intermediate"
    if ita <= 55:
        return "light"
    return "very_light"


def load_rows() -> list[dict[str, object]]:
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(SOURCE_JSON)
    rows: list[dict[str, object]] = []
    for row in json.loads(SOURCE_JSON.read_text()):
        ita = float(row["ita"])
        rows.append(
            {
                "class": row["class"],
                "path": row["path"],
                "is_aug": bool(row["is_aug"]),
                "ita": ita,
                "category": category(ita),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        cls = str(row["class"])
        cat = str(row["category"])
        counts[cls][cat] += 1
        values[(cls, cat)].append(float(row["ita"]))

    summary: list[dict[str, object]] = []
    for cls in CLASSES:
        total = sum(counts[cls].values())
        for cat, interval in CATEGORIES:
            count = counts[cls][cat]
            summary.append(
                {
                    "class": cls,
                    "category": cat,
                    "interval": interval,
                    "count": count,
                    "ratio_percent": count / total * 100.0 if total else 0.0,
                    "median_ita": float(np.median(values[(cls, cat)])) if values[(cls, cat)] else "",
                }
            )
    return summary


def representative(rows: list[dict[str, object]], cls: str, cat: str) -> dict[str, object] | None:
    candidates = [r for r in rows if r["class"] == cls and r["category"] == cat]
    if not candidates:
        return None
    target = float(np.median([float(r["ita"]) for r in candidates]))
    originals = [r for r in candidates if not bool(r["is_aug"])]
    pool = originals or candidates
    return min(pool, key=lambda r: abs(float(r["ita"]) - target))


def write_outputs(rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["class category count ratio_percent median_ita"]
    for row in summary:
        lines.append(
            f"{row['class']} {row['category']} {row['count']} "
            f"{float(row['ratio_percent']):.1f}% {row['median_ita']}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")

    by_key = {(str(r["class"]), str(r["category"])): r for r in summary}
    lines = [
        "# Skin Tone ITA6 Report",
        "",
        "대상: strict QC 실패 aug를 제외한 train 이미지.",
        "기준: Individual Typology Angle 6구간.",
        "",
        "| class | dark | brown | tan | intermediate | light | very_light | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in CLASSES:
        total = sum(1 for r in rows if r["class"] == cls)
        cells = []
        for cat, _ in CATEGORIES:
            item = by_key[(cls, cat)]
            cells.append(f"{int(item['count'])} / {float(item['ratio_percent']):.1f}%")
        lines.append(f"| {cls} | " + " | ".join(cells) + f" | {total} |")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def make_sheet(rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    title_font = load_font(24, bold=True)
    header_font = load_font(15, bold=True)
    label_font = load_font(12)
    small_font = load_font(10)

    row_label_w = 130
    cell_w = 205
    thumb = 158
    label_h = 76
    header_h = 105
    row_h = thumb + label_h
    width = row_label_w + cell_w * len(CATEGORIES)
    height = header_h + row_h * len(CLASSES)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((14, 14), "Skin tone ITA 6 levels by class", fill=(0, 0, 0), font=title_font)
    draw.text((14, 50), "Strict-pass train / ITA: dark, brown, tan, intermediate, light, very_light", fill=(60, 60, 60), font=label_font)

    for idx, (cat, interval) in enumerate(CATEGORIES):
        x = row_label_w + idx * cell_w
        draw.rectangle([x, 72, x + cell_w, header_h], fill=(238, 238, 238), outline=(210, 210, 210))
        draw.text((x + 6, 76), cat, fill=(0, 0, 0), font=header_font)
        draw.text((x + 6, 94), interval[:22], fill=(70, 70, 70), font=small_font)

    summary_by_key = {(str(r["class"]), str(r["category"])): r for r in summary}
    for row_idx, cls in enumerate(CLASSES):
        y = header_h + row_idx * row_h
        draw.rectangle([0, y, row_label_w, y + row_h], fill=(245, 245, 245), outline=(220, 220, 220))
        draw.text((12, y + 18), cls, fill=(0, 0, 0), font=header_font)
        for col_idx, (cat, _) in enumerate(CATEGORIES):
            x = row_label_w + col_idx * cell_w
            draw.rectangle([x, y, x + cell_w, y + row_h], outline=(225, 225, 225))
            item = summary_by_key[(cls, cat)]
            count = int(item["count"])
            ratio = float(item["ratio_percent"])
            rep = representative(rows, cls, cat)
            if rep is None:
                draw.text((x + 50, y + 64), "No sample", fill=(125, 125, 125), font=header_font)
                draw.text((x + 8, y + thumb + 12), f"{count} / {ratio:.1f}%", fill=(0, 0, 0), font=label_font)
                continue
            path = PROJECT_ROOT / str(rep["path"])
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                px = x + (cell_w - im.width) // 2
                py = y + 8 + (thumb - im.height) // 2
                canvas.paste(im, (px, py))
            base = Path(str(rep["path"])).name
            if len(base) > 24:
                base = base[:21] + "..."
            draw.text((x + 8, y + thumb + 12), f"{count} / {ratio:.1f}%  ITA {float(rep['ita']):.1f}", fill=(0, 0, 0), font=label_font)
            draw.text((x + 8, y + thumb + 34), base, fill=(80, 80, 80), font=small_font)

    canvas.save(SHEET, quality=94)


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_outputs(rows, summary)
    make_sheet(rows, summary)

    print("Skin tone ITA6 report")
    for cls in CLASSES:
        total = sum(1 for r in rows if r["class"] == cls)
        parts = []
        for cat, _ in CATEGORIES:
            n = sum(1 for r in rows if r["class"] == cls and r["category"] == cat)
            parts.append(f"{cat}={n}/{total} ({n / total * 100:.1f}%)")
        print(f"{cls}: " + ", ".join(parts))
    print(f"summary_json: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"summary_txt: {SUMMARY_TXT.relative_to(PROJECT_ROOT)}")
    print(f"report: {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"sheet: {SHEET.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
