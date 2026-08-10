from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_manifest


def _resize_array(image: Image.Image, size: int) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _crop(image: Image.Image, bbox: list[float], margin: float) -> Image.Image:
    x, y, width, height = bbox
    margin_x = width * margin
    margin_y = height * margin
    return image.crop(
        (
            max(0, int(np.floor(x - margin_x))),
            max(0, int(np.floor(y - margin_y))),
            min(image.width, int(np.ceil(x + width + margin_x))),
            min(image.height, int(np.ceil(y + height + margin_y))),
        )
    )


def build_cache(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entry_count = sum(
        1 if record["record_type"] == "classification" else 2 * len(record["annotations"])
        for record in records
    )
    array_filename = "images.npy"
    images = np.lib.format.open_memmap(
        args.output_dir / array_filename,
        mode="w+",
        dtype=np.uint8,
        shape=(entry_count, args.cache_size, args.cache_size, 3),
    )
    index: dict[str, int] = {}

    def store(key: str, image: Image.Image) -> None:
        row = len(index)
        images[row] = _resize_array(image, args.cache_size)
        index[key] = row

    for number, record in enumerate(records, start=1):
        source_path = args.dataset_root / record["image_path"]
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        if record["record_type"] == "classification":
            key = f"classification:{record['image_path']}"
            store(key, image)
        else:
            for annotation in record["annotations"]:
                identity = f"{record['image_id']}:{annotation['annotation_id']}"
                for variant, margin in (
                    ("exact", args.eval_margin_ratio),
                    ("train", args.train_margin_ratio),
                ):
                    key = f"roi-{variant}:{identity}"
                    roi = _crop(image, annotation["bbox_xywh"], margin)
                    store(key, roi)
        if number % 100 == 0:
            print(json.dumps({"records_processed": number, "cache_entries": len(index)}), flush=True)
    images.flush()
    metadata = {
        "schema_version": "1.0",
        "manifest": args.manifest.name,
        "cache_size": args.cache_size,
        "train_margin_ratio": args.train_margin_ratio,
        "eval_margin_ratio": args.eval_margin_ratio,
        "entry_count": len(index),
        "array_filename": array_filename,
        "index": index,
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ignored NumPy cache for classifier crops")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--train-margin-ratio", type=float, default=0.08)
    parser.add_argument("--eval-margin-ratio", type=float, default=0.05)
    build_cache(parser.parse_args())


if __name__ == "__main__":
    main()
