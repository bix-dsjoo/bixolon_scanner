from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _manifest_version(lines: list[str]) -> str:
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    return f"bread-{digest[:12]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_folds(records: list[dict[str, Any]], fold_count: int) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["split"] == "development":
            groups[record["capture_session_id"]].append(record)
    fold_sizes = [0] * fold_count
    fold_classes = [Counter() for _ in range(fold_count)]
    group_stats: list[tuple[str, int, Counter[int]]] = []
    for group_id, group_records in groups.items():
        counts: Counter[int] = Counter()
        total = 0
        for record in group_records:
            for annotation in record["annotations"]:
                counts[annotation["category_id"]] += 1
                total += 1
        group_stats.append((group_id, total, counts))
    group_to_fold: dict[str, int] = {}
    for group_id, total, counts in sorted(group_stats, key=lambda item: (-item[1], item[0])):
        def cost(fold: int) -> tuple[float, int]:
            class_penalty = sum((fold_classes[fold][key] + value) ** 2 for key, value in counts.items())
            return fold_sizes[fold] + total + 0.01 * class_penalty, fold

        selected = min(range(fold_count), key=cost)
        group_to_fold[group_id] = selected
        fold_sizes[selected] += total
        fold_classes[selected].update(counts)
    for record in records:
        if record["split"] == "development":
            record["fold"] = group_to_fold[record["capture_session_id"]]


def build_manifest(dataset_root: Path, *, test_date: str = "2026-07-21", fold_count: int = 3):
    root = dataset_root.resolve()
    annotation_path = root / "group" / "annotations" / "instances.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    categories = {int(category["id"]): category["name"] for category in coco["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(
            {
                "annotation_id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "bbox_xywh": [float(value) for value in annotation["bbox"]],
                "area": float(annotation["area"]),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )

    detection_records: list[dict[str, Any]] = []
    for image_record in sorted(coco["images"], key=lambda item: int(item["id"])):
        filename = image_record["file_name"]
        original_path = root / "group" / "images_original_exif" / filename
        training_path = root / "group" / "images" / filename
        with Image.open(original_path) as source:
            exif = source.getexif()
            captured = datetime.strptime(str(exif.get(306)), "%Y:%m:%d %H:%M:%S")
            camera = " ".join(value for value in (exif.get(271), exif.get(272)) if value)
        capture_date = captured.strftime("%Y-%m-%d")
        detection_records.append(
            {
                "record_type": "detection",
                "source": "coco_group",
                "image_path": (Path("group") / "images" / filename).as_posix(),
                "image_sha256": _file_sha256(training_path),
                "image_id": int(image_record["id"]),
                "width": int(image_record["width"]),
                "height": int(image_record["height"]),
                "capture_time": captured.isoformat(),
                "capture_session_id": captured.strftime("%Y-%m-%dT%H"),
                "camera": camera,
                "split": "test" if capture_date == test_date else "development",
                "fold": None,
                "annotations": annotations_by_image[int(image_record["id"])],
            }
        )
    _assign_folds(detection_records, fold_count)

    classification_records: list[dict[str, Any]] = []
    for class_dir in sorted((root / "train").iterdir()):
        if not class_dir.is_dir():
            continue
        match = re.match(r"Bread(\d{2})_(.+)", class_dir.name)
        if match is None:
            raise ValueError(f"invalid class directory: {class_dir.name}")
        category_id = int(match.group(1))
        for path in sorted(class_dir.glob("*.jpg")):
            stem_group = re.sub(r"\s*\([^)]*\)$", "", path.stem)
            classification_records.append(
                {
                    "record_type": "classification",
                    "source": "classification_aux",
                    "image_path": path.relative_to(root).as_posix(),
                    "image_sha256": _file_sha256(path),
                    "category_id": category_id,
                    "source_group": f"{category_id:02d}:{stem_group}",
                    "split": "train_aux",
                    "fold": None,
                }
            )
    labels = [
        {"category_id": category_id, "class_id": f"bread_{category_id:02d}", "class_name": name}
        for category_id, name in sorted(categories.items())
    ]
    return detection_records + classification_records, labels


def write_manifest(dataset_root: Path, output_dir: Path, *, test_date: str, fold_count: int) -> str:
    records, labels = build_manifest(dataset_root, test_date=test_date, fold_count=fold_count)
    lines = [_canonical_json(record) for record in records]
    version = _manifest_version(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "1.0",
        "dataset_version": version,
        "test_date": test_date,
        "fold_count": fold_count,
        "record_count": len(records),
        "labels": labels,
        "manifest_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the versioned bread dataset manifest")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-date", default="2026-07-21")
    parser.add_argument("--fold-count", type=int, default=3)
    args = parser.parse_args()
    version = write_manifest(
        args.dataset_root, args.output_dir, test_date=args.test_date, fold_count=args.fold_count
    )
    print(version)


if __name__ == "__main__":
    main()
