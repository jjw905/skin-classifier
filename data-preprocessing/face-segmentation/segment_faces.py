#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, pipeline


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_SETTINGS = {
    "models": {
        "face_parse_model": "jonathandinu/face-parsing",
        "sam2_model": "facebook/sam2-hiera-tiny",
    },
    "segmentation": {
        "bg_color": [200, 200, 200],
        "sam2_long_side": 1024,
        "sam2_points_per_batch": 32,
        "sam2_points_per_crop": 16,
        "mask_label_names": [
            "hair",
            "l_brow", "r_brow", "left_eyebrow", "right_eyebrow",
            "l_eye", "r_eye", "left_eye", "right_eye", "eye_g",
            "beard", "mustache", "moustache", "facial_hair",
        ],
    },
    "runtime": {
        "src": "/kaggle/input/skin-face-aihub-training/datasets/face_aihub/train",
        "dst": "/kaggle/working/datasets/face_aihub_segmented_v2/train",
        "labels_src": "/kaggle/input/skin-face-aihub-training/datasets/face_aihub/labels",
        "labels_dst": "/kaggle/working/datasets/face_aihub_segmented_v2/labels",
        "device": 0,
        "skip_existing": True,
        "limit": None,
    },
}


def load_settings(config_path: Optional[str]) -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if not config_path:
        return settings

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    user_cfg = json.loads(path.read_text(encoding="utf-8"))
    for section in ("models", "segmentation", "runtime"):
        if section in user_cfg:
            settings[section].update(user_cfg[section])
    return settings


def resize_long_side(img: Image.Image, long_side: int) -> Image.Image:
    w, h = img.size
    scale = long_side / max(w, h)
    if scale >= 1:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    m = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    m = m.resize(size, Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def load_face_parser(device: torch.device, face_parse_model: str, mask_label_names: set[str]):
    processor = SegformerImageProcessor.from_pretrained(face_parse_model)
    model = SegformerForSemanticSegmentation.from_pretrained(face_parse_model).to(device).eval()
    mask_ids = []
    matched = []
    for idx, label in model.config.id2label.items():
        label_name = label.lower()
        if label_name in mask_label_names:
            mask_ids.append(int(idx))
            matched.append(label_name)
    if not mask_ids:
        raise RuntimeError("mask labels not found")
    print("mask labels:", ", ".join(sorted(matched)))
    return processor, model, mask_ids


def mask_face_parts(img: Image.Image, processor, model, mask_ids, device: torch.device, bg_color: tuple[int, int, int]) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = F.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    part_mask = np.zeros((h, w), dtype=bool)
    for label_id in mask_ids:
        part_mask |= pred == label_id

    arr = np.array(rgb)
    arr[part_mask] = bg_color
    return Image.fromarray(arr)


def load_sam2(device_index: int, sam2_model: str, points_per_batch: int):
    return pipeline(
        "mask-generation",
        model=sam2_model,
        device=device_index,
        points_per_batch=points_per_batch,
    )


def sam2_foreground_mask(img: Image.Image, sam2_pipe, sam2_long_side: int, points_per_crop: int) -> np.ndarray:
    resized = resize_long_side(img.convert("RGB"), sam2_long_side)
    out = sam2_pipe(resized, points_per_crop=points_per_crop)
    masks = out.get("masks", [])
    if not masks:
        return np.ones((img.height, img.width), dtype=np.uint8)

    np_masks = [
        np.array(mask.numpy() if hasattr(mask, "numpy") else mask, dtype=np.uint8)
        for mask in masks
    ]
    best = np_masks[int(np.argmax([mask.sum() for mask in np_masks]))]
    return resize_mask(best, img.size)


def apply_background_mask(img: Image.Image, mask: np.ndarray, bg_color: tuple[int, int, int]) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    bg = np.full_like(arr, bg_color)
    out = np.where(mask[:, :, None] > 0, arr, bg)
    return Image.fromarray(out.astype(np.uint8))


def copy_labels(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def process(src: Path, dst: Path, labels_src: Optional[Path], labels_dst: Optional[Path],
            device_index: int, skip_existing: bool, limit: Optional[int], settings: dict) -> None:
    models = settings["models"]
    seg = settings["segmentation"]
    bg_color = tuple(seg["bg_color"])
    mask_label_names = {name.lower() for name in seg["mask_label_names"]}

    device = torch.device(f"cuda:{device_index}" if device_index >= 0 and torch.cuda.is_available() else "cpu")
    face_processor, face_model, mask_ids = load_face_parser(
        device=device,
        face_parse_model=models["face_parse_model"],
        mask_label_names=mask_label_names,
    )
    sam2_pipe = load_sam2(
        device_index=device_index if torch.cuda.is_available() else -1,
        sam2_model=models["sam2_model"],
        points_per_batch=int(seg["sam2_points_per_batch"]),
    )

    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if limit:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"no images: {src}")

    ok = skip = err = 0
    for idx, path in enumerate(images, 1):
        rel = path.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out_path.exists():
            skip += 1
            continue

        try:
            img = Image.open(path).convert("RGB")
            part_masked = mask_face_parts(img, face_processor, face_model, mask_ids, device, bg_color)
            mask = sam2_foreground_mask(
                part_masked,
                sam2_pipe,
                sam2_long_side=int(seg["sam2_long_side"]),
                points_per_crop=int(seg["sam2_points_per_crop"]),
            )
            result = apply_background_mask(part_masked, mask, bg_color)
            result.save(out_path)
            ok += 1
        except Exception as exc:
            err += 1
            print(f"\n[ERR] {rel}: {exc}")

        print(f"\r[{idx}/{len(images)}] ok={ok} skip={skip} err={err} {rel}", end="", flush=True)

    print(f"\nDone. ok={ok}, skip={skip}, err={err}, dst={dst}")
    if labels_src and labels_dst:
        copy_labels(labels_src, labels_dst)
        print(f"labels copied: {labels_dst}")


def main() -> None:
    default_runtime = DEFAULT_SETTINGS["runtime"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--src", default=None)
    parser.add_argument("--dst", default=None)
    parser.add_argument("--labels-src", default=None)
    parser.add_argument("--labels-dst", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.set_defaults(skip_existing=None)
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    skip_group.add_argument("--no-skip", dest="skip_existing", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings(args.config)
    runtime = settings["runtime"]
    src = args.src if args.src is not None else runtime.get("src", default_runtime["src"])
    dst = args.dst if args.dst is not None else runtime.get("dst", default_runtime["dst"])
    labels_src = args.labels_src if args.labels_src is not None else runtime.get("labels_src", default_runtime["labels_src"])
    labels_dst = args.labels_dst if args.labels_dst is not None else runtime.get("labels_dst", default_runtime["labels_dst"])
    device = args.device if args.device is not None else int(runtime.get("device", default_runtime["device"]))
    skip_existing = args.skip_existing if args.skip_existing is not None else bool(runtime.get("skip_existing", default_runtime["skip_existing"]))
    limit = args.limit if args.limit is not None else runtime.get("limit", default_runtime["limit"])

    process(
        src=Path(src),
        dst=Path(dst),
        labels_src=Path(labels_src) if labels_src else None,
        labels_dst=Path(labels_dst) if labels_dst else None,
        device_index=device,
        skip_existing=skip_existing,
        limit=limit,
        settings=settings,
    )


if __name__ == "__main__":
    main()
