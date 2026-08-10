from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_manifest


def build_cache(args: argparse.Namespace) -> None:
    records = [
        record for record in read_manifest(args.manifest) if record["record_type"] == "detection"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    array_filename = "images.npy"
    images = np.lib.format.open_memmap(
        args.output_dir / array_filename,
        mode="w+",
        dtype=np.uint8,
        shape=(len(records), args.image_size, args.image_size, 3),
    )
    index: dict[str, int] = {}
    for row, record in enumerate(records):
        with Image.open(args.dataset_root / record["image_path"]) as source:
            image = source.convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BILINEAR
            )
        images[row] = np.asarray(image, dtype=np.uint8)
        index[str(record["image_id"])] = row
        if (row + 1) % 50 == 0:
            print(json.dumps({"images_processed": row + 1}), flush=True)
    images.flush()
    metadata = {
        "schema_version": "1.0",
        "manifest": args.manifest.name,
        "image_size": args.image_size,
        "entry_count": len(records),
        "array_filename": array_filename,
        "index": index,
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ignored mmap cache for detector images")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    build_cache(parser.parse_args())


if __name__ == "__main__":
    main()
