from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from bixolon_scanner.contracts.catalog import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a checksummed Catalog manifest from original support captures"
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    root = args.dataset_root.resolve()
    rows = []
    seen: set[str] = set()
    for line in args.source_manifest.read_text(encoding="utf-8").splitlines():
        source = json.loads(line)
        image_path = str(source["original_image_path"])
        image_sha256 = str(source["original_image_sha256"])
        if image_sha256 in seen:
            continue
        seen.add(image_sha256)
        path = (root / image_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Catalog support path escapes its dataset root")
        if sha256_file(path) != image_sha256:
            raise ValueError("Catalog support checksum mismatch")
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        rows.append(
            {
                "record_type": "classification",
                "source": "bix_bakery_original_catalog_support",
                "source_dataset": "bix_bakery_dataset",
                "split": "train_support",
                "class_id": source["class_id"],
                "class_name": source["class_name"],
                "category_id": source["category_id"],
                "image_path": image_path,
                "image_sha256": image_sha256,
                "perceptual_group_id": source["perceptual_group_id"],
                "capture_session_id": source["capture_session_id"],
                "physical_item_id": source["physical_item_id"],
                "width": width,
                "height": height,
            }
        )
    counts = Counter(str(row["class_id"]) for row in rows)
    if len(counts) != 20 or set(counts.values()) != {10}:
        raise ValueError("Catalog support requires exactly ten originals for each of 20 classes")
    rows.sort(key=lambda row: (str(row["class_id"]), str(row["image_sha256"])))
    args.output_dir.mkdir(parents=True)
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "dataset_version": "bread-bix-bakery-catalog-support-200",
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "manifest_sha256": sha256_file(manifest),
        "class_count": 20,
        "support_count": 200,
        "support_count_per_class": 10,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
