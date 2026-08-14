from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .ten_shot_manifest import inspect_image

SCHEMA_VERSION = "1.0"
EXPECTED_ROOT_NAME = "bread_dataset"
EXPECTED_CLASS_COUNT = 20
EXPECTED_SHOTS_PER_CLASS = 10
TRAINING_SOURCES = {
    "single_objects": 10,
    "single_objects_1": 7,
    "single_objects_2": 10,
    "single_objects_3": 12,
}
CLASS_DIRECTORY_PATTERN = re.compile(r"^bread_(?P<number>\d{2})_(?P<slug>[a-z0-9_]+)$")
IMAGE_NAME_PATTERN = re.compile(
    r"^bread_(?P<number>\d{2})_(?P<side>normal|flipped)_"
    r"(?P<view>vertical|ground_30_dir_(?:01|02|03|04))\.jpg$",
    re.IGNORECASE,
)
GENERAL_IMAGE_NAME_PATTERN = re.compile(
    r"^bread_(?P<number>\d{2})_(?P<capture>[a-z0-9_]+)\.jpg$",
    re.IGNORECASE,
)
SIDES = ("normal", "flipped")
VIEWS = (
    "vertical",
    "ground_30_dir_01",
    "ground_30_dir_02",
    "ground_30_dir_03",
    "ground_30_dir_04",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return {"scon": "scone", "almond_scon": "almond_scone"}.get(result, result)


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required annotation file is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"annotation file must contain an object: {path.name}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"annotation file is missing {key}: {path.name}")
    return value


def _safe_annotation_image(root: Path, annotation_file: Path, name: str) -> Path:
    candidate = (annotation_file.parent / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"annotation image escapes bread_dataset: {name}") from error
    if not candidate.is_file():
        raise ValueError(f"annotation image is missing: {name}")
    return candidate


def _evaluation_image_identity(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as source:
            width, height = source.size
            if int(source.getexif().get(274, 1)) in {5, 6, 7, 8}:
                width, height = height, width
            source.verify()
    except Exception as error:
        raise ValueError(f"cannot decode evaluation image: {path}") from error
    return width, height, _sha256_file(path)


def _validate_categories(coco: dict[str, Any]) -> list[dict[str, Any]]:
    categories = sorted(coco["categories"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in categories]
    if ids != list(range(1, EXPECTED_CLASS_COUNT + 1)):
        raise ValueError("bread categories must be contiguous from 1 through 20")
    return [
        {
            "category_id": int(row["id"]),
            "class_id": f"bread_{int(row['id']):02d}",
            "class_name": str(row["name"]),
        }
        for row in categories
    ]


def _evaluation_summary(
    root: Path,
    annotation_file: Path,
    coco: dict[str, Any],
    *,
    expected_parent: str,
) -> dict[str, Any]:
    image_ids: set[int] = set()
    content_rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    for row in coco["images"]:
        image_id = int(row["id"])
        if image_id in image_ids:
            raise ValueError(f"duplicate image id in {annotation_file.name}: {image_id}")
        image_ids.add(image_id)
        path = _safe_annotation_image(root, annotation_file, str(row["file_name"]))
        width, height, image_sha256 = _evaluation_image_identity(path)
        if width != int(row["width"]) or height != int(row["height"]):
            raise ValueError(f"annotation image dimensions changed: {row['file_name']}")
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] != expected_parent:
            raise ValueError(
                f"{annotation_file.name} may only reference {expected_parent}: {relative}"
            )
        if row.get("status") is not None:
            statuses[str(row["status"])] += 1
        if expected_parent == "multi_object_scenes" and len(relative.parts) > 1:
            difficulties[relative.parts[1]] += 1
        content_rows.append({"path": relative.as_posix(), "sha256": image_sha256})
    annotation_image_ids = {int(row["image_id"]) for row in coco["annotations"]}
    if not annotation_image_ids <= image_ids:
        raise ValueError(f"orphan annotation in {annotation_file.name}")
    return {
        "annotation_file": annotation_file.relative_to(root).as_posix(),
        "annotation_sha256": _sha256_file(annotation_file),
        "image_content_sha256": hashlib.sha256(
            _canonical_json(sorted(content_rows, key=lambda row: row["path"])).encode("utf-8")
        ).hexdigest(),
        "image_count": len(image_ids),
        "annotation_count": len(coco["annotations"]),
        "status_counts": dict(sorted(statuses.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "training_allowed": False,
    }


def audit_bread_dataset(
    dataset_root: Path,
    *,
    training_source: str = "single_objects",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = dataset_root.resolve()
    if not root.is_dir() or root.name != EXPECTED_ROOT_NAME:
        raise ValueError(f"dataset root must be a directory named {EXPECTED_ROOT_NAME}")
    if training_source not in TRAINING_SOURCES:
        raise ValueError(f"unsupported bread training source: {training_source}")
    allowed_top_level = {
        *TRAINING_SOURCES,
        "multi_object_scenes",
        "scan_log_samples",
        "annotations",
    }
    actual_top_level = {path.name for path in root.iterdir()}
    unexpected = actual_top_level - allowed_top_level
    missing = allowed_top_level - actual_top_level
    if missing or unexpected:
        raise ValueError(
            "bread_dataset top-level entries mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    multi_annotation = root / "annotations" / "multi_object_instances.json"
    scan_annotation = root / "annotations" / "scan_log_instances.json"
    multi_coco = _load_coco(multi_annotation)
    scan_coco = _load_coco(scan_annotation)
    labels = _validate_categories(multi_coco)
    scan_labels = _validate_categories(scan_coco)
    if [(_["category_id"], _slug(_["class_name"])) for _ in scan_labels] != [
        (_["category_id"], _slug(_["class_name"])) for _ in labels
    ]:
        raise ValueError("multi-object and scan-log category contracts differ")

    single_root = root / training_source
    class_directories = sorted(path for path in single_root.iterdir() if path.is_dir())
    if len(class_directories) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"{training_source} must contain exactly 20 class directories")

    records: list[dict[str, Any]] = []
    content_rows: list[dict[str, Any]] = []
    labels_by_id = {int(row["category_id"]): row for row in labels}
    expected_slots = {(side, view) for side in SIDES for view in VIEWS}
    shots_per_class = TRAINING_SOURCES[training_source]
    seen_categories: set[int] = set()
    seen_hashes: set[str] = set()
    for directory in class_directories:
        match = CLASS_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"invalid single-object directory: {directory.name}")
        category_id = int(match.group("number"))
        label = labels_by_id.get(category_id)
        if label is None or _slug(str(label["class_name"])) != match.group("slug"):
            raise ValueError(f"class directory does not match category metadata: {directory.name}")
        seen_categories.add(category_id)
        files = sorted(path for path in directory.iterdir() if path.is_file())
        if len(files) != shots_per_class:
            raise ValueError(
                f"{directory.name} must contain exactly {shots_per_class} original JPEGs"
            )
        slots: set[tuple[str, str]] = set()
        for path in files:
            image_match = GENERAL_IMAGE_NAME_PATTERN.fullmatch(path.name)
            if image_match is None or int(image_match.group("number")) != category_id:
                raise ValueError(f"invalid single-object filename: {path.name}")
            capture = image_match.group("capture").lower()
            if capture.startswith("normal_"):
                slot = ("normal", capture.removeprefix("normal_"))
            elif capture.startswith("flipped_"):
                slot = ("flipped", capture.removeprefix("flipped_"))
            else:
                slot = ("unpaired", capture)
            if slot in slots:
                raise ValueError(f"duplicate capture slot in {directory.name}: {slot}")
            slots.add(slot)
            inspected = inspect_image(path)
            if inspected.sha256 in seen_hashes:
                raise ValueError(f"duplicate training original: {path.name}")
            seen_hashes.add(inspected.sha256)
            relative = path.relative_to(root).as_posix()
            record = {
                "record_type": "classification",
                "source": "bread_dataset_single_original",
                "source_dataset": EXPECTED_ROOT_NAME,
                "image_path": relative,
                "image_sha256": inspected.sha256,
                "category_id": category_id,
                "class_id": str(label["class_id"]),
                "class_name": str(label["class_name"]),
                "capture_session_id": f"bread_dataset:{training_source}",
                "physical_item_id": f"bread_{category_id:02d}:{training_source}:original_item",
                "side": slot[0],
                "view": slot[1],
                "source_group": f"bread_{category_id:02d}:{slot[0]}:{slot[1]}",
                "split": "train_support",
                "fold": None,
                "width": inspected.width,
                "height": inspected.height,
            }
            records.append(record)
            content_rows.append(
                {"path": relative, "sha256": inspected.sha256, "category_id": category_id}
            )
        if training_source == "single_objects" and slots != expected_slots:
            raise ValueError(f"capture slots mismatch in {directory.name}")
    if seen_categories != set(range(1, EXPECTED_CLASS_COUNT + 1)):
        raise ValueError("single-object categories must be contiguous from 1 through 20")

    evaluation_sets = {
        "multi_object_scenes": _evaluation_summary(
            root,
            multi_annotation,
            multi_coco,
            expected_parent="multi_object_scenes",
        ),
        "scan_log_samples": _evaluation_summary(
            root,
            scan_annotation,
            scan_coco,
            expected_parent="scan_log_samples",
        ),
    }
    digest_payload = {
        "training": sorted(content_rows, key=lambda row: row["path"]),
        "evaluation_annotations": {
            name: {
                "annotation_sha256": summary["annotation_sha256"],
                "image_content_sha256": summary["image_content_sha256"],
            }
            for name, summary in evaluation_sets.items()
        },
    }
    dataset_digest = hashlib.sha256(_canonical_json(digest_payload).encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": f"bread-1.0-{dataset_digest[:12]}",
        "dataset_digest_sha256": dataset_digest,
        "root_name": EXPECTED_ROOT_NAME,
        "root_policy": "only-this-root",
        "ignored_top_level_entries": sorted(
            (actual_top_level & set(TRAINING_SOURCES)) - {training_source}
        ),
        "training_contract": {
            "allowed_directory": training_source,
            "class_count": EXPECTED_CLASS_COUNT,
            "shots_per_class": shots_per_class,
            "original_image_count": len(records),
            "derived_evaluation_images_are_training_forbidden": True,
        },
        "class_count": EXPECTED_CLASS_COUNT,
        "shots_per_class": shots_per_class,
        "labels": labels,
        "evaluation_sets": evaluation_sets,
    }
    return records, metadata


def audit_bread_evaluation_set(dataset_root: Path, name: str) -> dict[str, Any]:
    root = dataset_root.resolve()
    choices = {
        "multi_object_scenes": ("multi_object_instances.json", "multi_object_scenes"),
        "scan_log_samples": ("scan_log_instances.json", "scan_log_samples"),
    }
    if name not in choices:
        raise ValueError(f"unsupported bread evaluation set: {name}")
    annotation_name, expected_parent = choices[name]
    annotation = root / "annotations" / annotation_name
    return _evaluation_summary(
        root,
        annotation,
        _load_coco(annotation),
        expected_parent=expected_parent,
    )


def build_detection_evaluation_records(dataset_root: Path) -> list[dict[str, Any]]:
    root = dataset_root.resolve()
    annotation_file = root / "annotations" / "multi_object_instances.json"
    coco = _load_coco(annotation_file)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    records: list[dict[str, Any]] = []
    for image in sorted(coco["images"], key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        path = _safe_annotation_image(root, annotation_file, str(image["file_name"]))
        width, height, image_sha256 = _evaluation_image_identity(path)
        annotations = sorted(annotations_by_image.get(image_id, []), key=lambda row: int(row["id"]))
        category_signature = ",".join(
            str(int(row["category_id"]))
            for row in sorted(annotations, key=lambda row: int(row["category_id"]))
        )
        group_id = f"multi_scene:{category_signature}"
        fold = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 3
        records.append(
            {
                "record_type": "detection",
                "source": "bread_dataset_multi_object_evaluation",
                "source_dataset": EXPECTED_ROOT_NAME,
                "image_id": image_id,
                "image_path": path.relative_to(root).as_posix(),
                "image_sha256": image_sha256,
                "capture_session_id": group_id,
                "physical_item_ids": sorted(
                    {f"bread_{int(row['category_id']):02d}:original_item" for row in annotations}
                ),
                "split": "development",
                "fold": fold,
                "width": width,
                "height": height,
                "exclude_from_detector_training": True,
                "training_allowed": False,
                "annotations": [
                    {
                        "annotation_id": int(row["id"]),
                        "category_id": int(row["category_id"]),
                        "bbox_xywh": [float(value) for value in row["bbox"]],
                        "area": float(row["area"]),
                        "iscrowd": int(row.get("iscrowd", 0)),
                    }
                    for row in annotations
                ],
            }
        )
    return records


def write_bread_dataset_registry(
    dataset_root: Path,
    output_dir: Path,
    *,
    training_source: str = "single_objects",
) -> str:
    records, metadata = audit_bread_dataset(dataset_root, training_source=training_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = "".join(_canonical_json(row) + "\n" for row in records)
    metadata["manifest_sha256"] = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    (output_dir / "manifest.jsonl").write_text(manifest, encoding="utf-8", newline="\n")
    evaluation_manifest = "".join(
        _canonical_json(row) + "\n" for row in build_detection_evaluation_records(dataset_root)
    )
    (output_dir / "evaluation_manifest.jsonl").write_text(
        evaluation_manifest, encoding="utf-8", newline="\n"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(metadata["dataset_version"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the canonical bread_dataset root")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-source", choices=tuple(TRAINING_SOURCES), default="single_objects"
    )
    args = parser.parse_args()
    print(
        write_bread_dataset_registry(
            args.dataset_root,
            args.output_dir,
            training_source=args.training_source,
        )
    )


if __name__ == "__main__":
    main()
