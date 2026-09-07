from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .ten_shot_manifest import inspect_image

EXPECTED_ROOT_NAME = "raw_data"
EXPECTED_CLASS_COUNT = 20
EXPECTED_IMAGES_PER_CLASS = 84
CLASS_DIRECTORY_PATTERN = re.compile(r"^Bread(?P<category>\d{2})_(?P<name>[A-Za-z][A-Za-z -]*)$")
EXPECTED_CLASS_NAMES = {
    1: "Walnut Donut",
    2: "Croffle",
    3: "Waffle",
    4: "Scon",
    5: "Half-moon Croissant",
    6: "Croissant",
    7: "Flower Bread",
    8: "Almond Scon",
    9: "Dinner Roll",
    10: "Sugar Donut",
    11: "Bagel",
    12: "Egg Tart",
    13: "Muffin",
    14: "Burger",
    15: "Sandwich",
    16: "Grain Campagne",
    17: "Almond Campagne",
    18: "Mini Bread",
    19: "Pastry Bread",
    20: "Plain Bread",
}
CANONICAL_CLASS_NAMES = {
    category_id: name.replace("Scon", "Scone") for category_id, name in EXPECTED_CLASS_NAMES.items()
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_raw_bread_classifier_source(
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit the complete 84-view-per-SKU classifier source.

    ``bread_dataset/single_objects*`` contains named subsets and exact copies of
    this source.  Keeping ``raw_data`` as one canonical manifest prevents copied
    files from being counted as independent evidence.
    """

    root = source_root.resolve()
    if not root.is_dir() or root.name != EXPECTED_ROOT_NAME:
        raise ValueError(f"source root must be a directory named {EXPECTED_ROOT_NAME}")
    entries = sorted(root.iterdir())
    if any(not path.is_dir() for path in entries):
        raise ValueError("raw_data may contain only bread class directories")
    if len(entries) != EXPECTED_CLASS_COUNT:
        raise ValueError("raw_data requires exactly 20 bread class directories")

    records: list[dict[str, Any]] = []
    seen_hashes: dict[str, Path] = {}
    observed_categories: set[int] = set()
    dimensions: Counter[str] = Counter()
    content_rows: list[dict[str, Any]] = []
    ignored_system_files: list[str] = []
    for directory in entries:
        match = CLASS_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"invalid raw bread class directory: {directory.name}")
        category_id = int(match.group("category"))
        if category_id in observed_categories:
            raise ValueError(f"duplicate raw bread category: {category_id}")
        observed_categories.add(category_id)
        expected_name = EXPECTED_CLASS_NAMES.get(category_id)
        if expected_name is None or match.group("name") != expected_name:
            raise ValueError(f"raw bread class directory does not match category: {directory.name}")
        directory_entries = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        unexpected = [
            path
            for path in directory_entries
            if not path.is_file()
            or (path.suffix.lower() not in {".jpg", ".jpeg"} and path.name != "Thumbs.db")
        ]
        if unexpected:
            raise ValueError(f"{directory.name} may contain JPEG images and Thumbs.db only")
        ignored_system_files.extend(
            path.relative_to(root).as_posix()
            for path in directory_entries
            if path.name == "Thumbs.db"
        )
        images = [path for path in directory_entries if path.suffix.lower() in {".jpg", ".jpeg"}]
        if len(images) != EXPECTED_IMAGES_PER_CLASS:
            raise ValueError(
                f"{directory.name} must contain exactly {EXPECTED_IMAGES_PER_CLASS} JPEG images"
            )
        for capture_index, path in enumerate(images):
            inspected = inspect_image(path)
            if inspected.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"raw bread image must decode as RGB: {path.name}")
            previous = seen_hashes.get(inspected.sha256)
            if previous is not None:
                raise ValueError(f"duplicate raw bread image: {previous.name} and {path.name}")
            seen_hashes[inspected.sha256] = path
            relative = path.relative_to(root).as_posix()
            dimensions[f"{inspected.width}x{inspected.height}"] += 1
            record = {
                "record_type": "classification",
                "source": "raw_bread_complete_view_original",
                "source_dataset": EXPECTED_ROOT_NAME,
                "image_path": relative,
                "image_sha256": inspected.sha256,
                "category_id": category_id,
                "class_id": f"bread_{category_id:02d}",
                "class_name": CANONICAL_CLASS_NAMES[category_id],
                "capture_session_id": f"bread_{category_id:02d}:raw_data:session",
                "physical_item_id": f"bread_{category_id:02d}:raw_data:item",
                "source_group": f"bread_{category_id:02d}:raw_data:item",
                "capture_index": capture_index,
                "split": "train_support",
                "fold": None,
                "width": inspected.width,
                "height": inspected.height,
            }
            records.append(record)
            content_rows.append(
                {
                    "path": relative,
                    "sha256": inspected.sha256,
                    "category_id": category_id,
                }
            )

    if observed_categories != set(range(1, EXPECTED_CLASS_COUNT + 1)):
        raise ValueError("raw bread categories must be contiguous from 1 through 20")
    digest = hashlib.sha256(_canonical_json(content_rows).encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": "1.0",
        "dataset_version": f"raw-bread-{digest[:12]}",
        "source_directory": EXPECTED_ROOT_NAME,
        "source_image_set_sha256": digest,
        "class_count": EXPECTED_CLASS_COUNT,
        "images_per_class": EXPECTED_IMAGES_PER_CLASS,
        "object_image_count": len(records),
        "labels": [
            {
                "category_id": category_id,
                "class_id": f"bread_{category_id:02d}",
                "class_name": CANONICAL_CLASS_NAMES[category_id],
            }
            for category_id in range(1, EXPECTED_CLASS_COUNT + 1)
        ],
        "dimension_distribution": dict(sorted(dimensions.items())),
        "allowed_source_root": root.as_posix(),
        "ignored_system_files": ignored_system_files,
        "duplicate_policy": (
            "canonical-complete-source; do not combine exact copies from "
            "bread_dataset/single_objects*"
        ),
    }
    return records, metadata


def write_raw_bread_classifier_registry(source_root: Path, output_dir: Path) -> str:
    records, metadata = audit_raw_bread_classifier_source(source_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = "".join(_canonical_json(record) + "\n" for record in records)
    metadata["manifest_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
    (output_dir / "manifest.jsonl").write_text(manifest, encoding="utf-8", newline="\n")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(metadata["dataset_version"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete raw bread classifier source")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(write_raw_bread_classifier_registry(args.source_root, args.output_dir))


if __name__ == "__main__":
    main()
