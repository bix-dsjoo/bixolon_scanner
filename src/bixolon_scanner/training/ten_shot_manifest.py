from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

SCHEMA_VERSION = "1.0"
DEFAULT_SHOTS_PER_CLASS = 10
CLASS_DIRECTORY_PATTERN = re.compile(r"^Bread(?P<number>\d{2})_(?P<name>.+)$")
IMAGE_NAME_PATTERN = re.compile(
    r"^bread(?P<number>\d{2})_(?P<side>normal|flipped)_"
    r"(?P<view>vertical|ground30_dir[1-4])\.jpg$",
    re.IGNORECASE,
)
SIDES = ("normal", "flipped")
VIEWS = ("vertical", "ground30_dir1", "ground30_dir2", "ground30_dir3", "ground30_dir4")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as source:
        image = (
            ImageOps.exif_transpose(source)
            .convert("L")
            .resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        )
    values = np.asarray(image, dtype=np.float32)
    bits = values >= float(values.mean())
    return np.packbits(bits.reshape(-1)).tobytes().hex()


def _hamming_hex(left: str, right: str) -> int:
    return sum(
        int(value).bit_count()
        for value in bytes(a ^ b for a, b in zip(bytes.fromhex(left), bytes.fromhex(right)))
    )


@dataclass(frozen=True)
class ImageAudit:
    width: int
    height: int
    mode: str
    sha256: str
    average_hash: str


def inspect_image(path: Path) -> ImageAudit:
    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            mode = image.mode
            image.convert("RGB").load()
    except Exception as error:
        raise ValueError(f"cannot decode JPEG: {path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {path}")
    return ImageAudit(
        width=width,
        height=height,
        mode=mode,
        sha256=_file_sha256(path),
        average_hash=_average_hash(path),
    )


def _expected_slots() -> set[tuple[str, str]]:
    return {(side, view) for side in SIDES for view in VIEWS}


def _load_labels(path: Path) -> list[dict[str, Any]]:
    if path is None:
        raise ValueError("labels metadata is required for strict 10-shot registration")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    labels = sorted(metadata.get("labels", []), key=lambda row: int(row["category_id"]))
    categories = [int(row["category_id"]) for row in labels]
    if not labels or categories != list(range(1, len(labels) + 1)):
        raise ValueError("labels must be non-empty, contiguous and one-based")
    required = {"category_id", "class_id", "class_name"}
    if any(not required.issubset(row) for row in labels):
        raise ValueError("each label requires category_id, class_id and class_name")
    return labels


def _near_duplicate_pairs(
    records: list[dict[str, Any]], *, maximum_hamming_distance: int
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            distance = _hamming_hex(left["average_hash"], right["average_hash"])
            if distance <= maximum_hamming_distance:
                pairs.append(
                    {
                        "left": left["image_path"],
                        "right": right["image_path"],
                        "left_category_id": left["category_id"],
                        "right_category_id": right["category_id"],
                        "hamming_distance": distance,
                    }
                )
    return pairs


def audit_ten_shot_dataset(
    dataset_root: Path,
    *,
    labels_metadata: Path,
    expected_classes: int | None = None,
    shots_per_class: int = DEFAULT_SHOTS_PER_CLASS,
    recommended_min_short_side: int = 224,
    near_duplicate_hamming_distance: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise ValueError(f"ten-shot dataset root does not exist: {root}")
    if shots_per_class != len(_expected_slots()):
        raise ValueError(
            f"shots_per_class must be {len(_expected_slots())} for the fixed capture slots"
        )
    labels = _load_labels(labels_metadata)
    metadata_class_count = len(labels)
    if expected_classes is not None and expected_classes != metadata_class_count:
        raise ValueError("expected class count does not match labels metadata")
    expected_classes = metadata_class_count
    labels_by_category = {int(row["category_id"]): row for row in labels}
    class_directories = sorted(path for path in root.iterdir() if path.is_dir())
    if len(class_directories) != expected_classes:
        raise ValueError(
            f"expected {expected_classes} class directories, got {len(class_directories)}"
        )

    audit_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    categories_seen: list[int] = []
    for class_directory in class_directories:
        directory_match = CLASS_DIRECTORY_PATTERN.fullmatch(class_directory.name)
        if directory_match is None:
            raise ValueError(f"invalid class directory: {class_directory.name}")
        category_id = int(directory_match.group("number"))
        categories_seen.append(category_id)
        label = labels_by_category.get(category_id)
        if label is None:
            raise ValueError(f"class directory has no label metadata: {class_directory.name}")
        directory_name = directory_match.group("name")
        expected_name = label.get("class_name")
        if expected_name is not None and directory_name != str(expected_name):
            raise ValueError(
                f"class name mismatch for category {category_id}: "
                f"directory={directory_name!r}, metadata={expected_name!r}"
            )
        files = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        )
        if len(files) != shots_per_class:
            raise ValueError(
                f"category {category_id} requires {shots_per_class} files, got {len(files)}"
            )
        slots: dict[tuple[str, str], Path] = {}
        for path in files:
            match = IMAGE_NAME_PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError(f"invalid ten-shot filename: {path.name}")
            if int(match.group("number")) != category_id:
                raise ValueError(f"filename category does not match directory: {path}")
            slot = (match.group("side").lower(), match.group("view").lower())
            if slot in slots:
                raise ValueError(f"duplicate capture slot for category {category_id}: {slot}")
            slots[slot] = path
        missing = sorted(_expected_slots() - set(slots))
        unexpected = sorted(set(slots) - _expected_slots())
        if missing or unexpected:
            raise ValueError(
                f"category {category_id} capture slots mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for side, view in sorted(slots):
            path = slots[(side, view)]
            inspected = inspect_image(path)
            relative_path = path.relative_to(root).as_posix()
            audit_row = {
                "image_path": relative_path,
                "category_id": category_id,
                "side": side,
                "view": view,
                "width": inspected.width,
                "height": inspected.height,
                "short_side": min(inspected.width, inspected.height),
                "mode": inspected.mode,
                "image_sha256": inspected.sha256,
                "average_hash": inspected.average_hash,
            }
            audit_records.append(audit_row)
            manifest_records.append(
                {
                    "record_type": "classification",
                    "source": "bread_project_3_ten_shot",
                    "source_dataset": root.name,
                    "image_path": relative_path,
                    "image_sha256": inspected.sha256,
                    "category_id": category_id,
                    "class_id": str(label["class_id"]),
                    "class_name": directory_name,
                    "capture_session_id": f"{root.name}:legacy",
                    "physical_item_id": f"bread_{category_id:02d}:legacy-item",
                    "side": side,
                    "view": view,
                    "source_group": f"bread_{category_id:02d}:{side}:{view}",
                    "split": "train_support",
                    "fold": None,
                    "width": inspected.width,
                    "height": inspected.height,
                }
            )

    if sorted(categories_seen) != list(range(1, expected_classes + 1)):
        raise ValueError("class directories must be contiguous and one-based")
    sha_counts = Counter(row["image_sha256"] for row in audit_records)
    duplicate_hashes = sorted(digest for digest, count in sha_counts.items() if count > 1)
    if duplicate_hashes:
        raise ValueError(f"exact duplicate images found: {duplicate_hashes[:3]}")

    lines = [_canonical_json(record) for record in manifest_records]
    manifest_body = "\n".join(lines) + "\n"
    manifest_sha256 = hashlib.sha256(manifest_body.encode("utf-8")).hexdigest()
    short_sides = [int(row["short_side"]) for row in audit_records]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": f"bread-10shot-{manifest_sha256[:12]}",
        "source_dataset": root.name,
        "record_count": len(manifest_records),
        "class_count": expected_classes,
        "shots_per_class": shots_per_class,
        "labels": labels,
        "manifest_sha256": manifest_sha256,
        "capture_contract": {
            "sides": list(SIDES),
            "views": list(VIEWS),
        },
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": metadata["dataset_version"],
        "record_count": len(audit_records),
        "minimum_short_side": min(short_sides),
        "recommended_min_short_side": recommended_min_short_side,
        "below_recommended_min_short_side": sum(
            value < recommended_min_short_side for value in short_sides
        ),
        "exact_duplicate_count": 0,
        "near_duplicate_hamming_distance": near_duplicate_hamming_distance,
        "near_duplicate_pairs": _near_duplicate_pairs(
            audit_records,
            maximum_hamming_distance=near_duplicate_hamming_distance,
        ),
        "images": audit_records,
    }
    return manifest_records, metadata, audit


def write_ten_shot_manifest(
    dataset_root: Path,
    output_dir: Path,
    *,
    labels_metadata: Path,
    expected_classes: int | None = None,
    shots_per_class: int = DEFAULT_SHOTS_PER_CLASS,
    recommended_min_short_side: int = 224,
    near_duplicate_hamming_distance: int = 2,
) -> str:
    records, metadata, audit = audit_ten_shot_dataset(
        dataset_root,
        labels_metadata=labels_metadata,
        expected_classes=expected_classes,
        shots_per_class=shots_per_class,
        recommended_min_short_side=recommended_min_short_side,
        near_duplicate_hamming_distance=near_duplicate_hamming_distance,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_body = "".join(_canonical_json(row) + "\n" for row in records)
    (output_dir / "manifest.jsonl").write_text(manifest_body, encoding="utf-8", newline="\n")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(metadata["dataset_version"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit an exact 10-shot bread registration dataset"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels-metadata", type=Path, required=True)
    parser.add_argument("--expected-classes", type=int)
    parser.add_argument("--shots-per-class", type=int, default=DEFAULT_SHOTS_PER_CLASS)
    parser.add_argument("--recommended-min-short-side", type=int, default=224)
    parser.add_argument("--near-duplicate-hamming-distance", type=int, default=2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    version = write_ten_shot_manifest(
        args.dataset_root,
        args.output_dir,
        labels_metadata=args.labels_metadata,
        expected_classes=args.expected_classes,
        shots_per_class=args.shots_per_class,
        recommended_min_short_side=args.recommended_min_short_side,
        near_duplicate_hamming_distance=args.near_duplicate_hamming_distance,
    )
    print(version)


if __name__ == "__main__":
    main()
