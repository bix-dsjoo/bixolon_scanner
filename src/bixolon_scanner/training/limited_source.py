"""Freeze a small original-image budget before any training or model evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _scene_features(row: dict, annotations: list[dict]) -> set[str]:
    features = {f"class:{int(a['category_id'])}" for a in annotations}
    features.add(f"count:{len(annotations)}")
    features.add(f"layout:{Path(row['file_name']).parent.name}")
    width, height = row["width"], row["height"]
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError("scene annotation is outside the image")
        features.add(
            f"position:{min(2, int(3 * (x + w / 2) / width))}:{min(2, int(3 * (y + h / 2) / height))}"
        )
        features.add(f"shape:{min(4, int(max(w / h, h / w)))}")
    for i, left in enumerate(annotations):
        x, y, w, h = left["bbox"]
        for right in annotations[i + 1 :]:
            a, b, c, d = right["bbox"]
            overlap = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
            if overlap / min(w * h, c * d) >= 0.2:
                features.add("overlap:yes")
    return features


def select_scenes(coco: dict, count: int, seed: int) -> list[dict]:
    """Greedy class coverage and scene diversity; never reads predictions or images."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    images = {int(row["id"]): row for row in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("duplicate scene image ids")
    eligible = {i: row for i, row in images.items() if len(by_image[i]) >= 2}
    if len(eligible) < count:
        raise ValueError("not enough annotated multi-object images")
    features = {i: _scene_features(row, by_image[i]) for i, row in eligible.items()}
    covered: Counter[str] = Counter()
    selected = []
    for _ in range(count):

        def rank(image_id: int):
            fs = features[image_id]
            new_classes = sum(f.startswith("class:") and not covered[f] for f in fs)
            diversity = sum((4 if f.startswith("class:") else 1) / (1 + covered[f]) for f in fs)
            tie = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
            return new_classes, diversity, tie

        image_id = max(eligible, key=rank)
        row = eligible.pop(image_id)
        selected.append({**row, "annotations": by_image[image_id]})
        covered.update(features[image_id])
    categories = {f"class:{int(row['id'])}" for row in coco["categories"]}
    if not categories.issubset(covered):
        raise ValueError("selected scenes do not cover every class")
    return selected


def prepare_sources(config_path: Path, output: Path) -> dict:
    config = load_json_config(config_path)
    if config["schema_version"] != "1.0":
        raise ValueError("unsupported limited-source configuration")
    if output.exists():
        raise FileExistsError(output)
    root = Path(config["dataset_root"]).resolve()
    singles = (root / config["single_directory"]).resolve()
    singles.relative_to(root)
    annotation_path = (root / config["multi_annotations"]).resolve()
    annotation_path.relative_to(root)
    coco = load_json_config(annotation_path)
    class_count = int(config["class_count"])
    labels = sorted(coco["categories"], key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in labels] != list(range(1, class_count + 1)):
        raise ValueError("category ids must be contiguous")
    by_class = {int(row["id"]): row for row in labels}
    source_rows = []
    classifier_rows = []
    directories = sorted(path for path in singles.iterdir() if path.is_dir())
    if len(directories) != class_count:
        raise ValueError("single-image directory has the wrong class count")
    seen_classes = set()
    for directory in directories:
        match = re.fullmatch(r"bread_(\d+)_([a-z0-9_]+)", directory.name)
        if match is None:
            raise ValueError("unexpected class directory")
        category = int(match[1])
        if category in seen_classes or category not in by_class:
            raise ValueError("duplicate or unknown source class")
        seen_classes.add(category)
        paths = sorted(p for p in directory.iterdir() if p.is_file())
        if len(paths) != config["shots_per_class"]:
            raise ValueError("each class must have exactly the configured number of originals")
        for path in paths:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("unexpected source file")
            path.resolve().relative_to(root)
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                width, height = image.size
            source_rows.append(
                {
                    "kind": "single",
                    "image_path": path.relative_to(root).as_posix(),
                    "image_sha256": sha256_file(path),
                    "category_id": category,
                    "width": width,
                    "height": height,
                    "physical_item_ids": [f"bread_{category:02d}:shared_item"],
                    "capture_session_id": "single_objects_2:shared_session",
                    "source_group": "shared_physical_item_collection",
                }
            )
            classifier_rows.append(
                {
                    **source_rows[-1],
                    "record_type": "classification",
                    "split": "train",
                    "class_id": f"bread_{category:02d}",
                    "class_name": by_class[category]["name"],
                    "original_image_sha256": source_rows[-1]["image_sha256"],
                }
            )
    scenes = select_scenes(coco, config["multi_count"], config["seed"])
    detector_rows = []
    for row in scenes:
        path = (annotation_path.parent / row["file_name"]).resolve()
        path.relative_to(root / "multi_object_scenes")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            if image.size != (row["width"], row["height"]):
                raise ValueError("scene annotation and decoded image dimensions differ")
        source_rows.append(
            {
                "kind": "multi",
                "image_path": path.relative_to(root).as_posix(),
                "image_sha256": sha256_file(path),
                "width": row["width"],
                "height": row["height"],
                "capture_session_id": "multi_object_scenes:shared_session",
                "physical_item_ids": sorted(
                    {f"bread_{int(a['category_id']):02d}:shared_item" for a in row["annotations"]}
                ),
                "source_group": "shared_physical_item_collection",
            }
        )
        detector_rows.append(
            {
                **source_rows[-1],
                "record_type": "detection",
                "split": "train",
                "image_id": row["id"],
                "annotations": [
                    {
                        "annotation_id": a["id"],
                        "category_id": a["category_id"],
                        "bbox_xywh": a["bbox"],
                        "area": a["bbox"][2] * a["bbox"][3],
                        "iscrowd": 0,
                    }
                    for a in row["annotations"]
                ],
            }
        )
    hashes = [row["image_sha256"] for row in source_rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate original content in source budget")
    expected = class_count * config["shots_per_class"] + config["multi_count"]
    if len(hashes) != expected or expected != config["original_budget"]:
        raise ValueError("source image budget mismatch")
    output.mkdir(parents=True)
    write_jsonl(output / "originals.jsonl", source_rows)
    write_jsonl(output / "classifier.jsonl", classifier_rows)
    write_jsonl(output / "detector.jsonl", detector_rows)
    write_json(
        output / "selected-scenes.coco.json",
        {
            "images": [{k: v for k, v in row.items() if k != "annotations"} for row in scenes],
            "annotations": [a for row in scenes for a in row["annotations"]],
            "categories": labels,
        },
    )
    report = {
        "schema_version": "1.0",
        "dataset_root": str(root),
        "original_count": len(hashes),
        "single_count": len(classifier_rows),
        "multi_count": len(detector_rows),
        "multi_object_count": sum(len(r["annotations"]) for r in detector_rows),
        "class_count": class_count,
        "selection_seed": config["seed"],
        "selection_used_model_predictions": False,
        "unselected_image_pixels_read": False,
        "config_sha256": sha256_file(config_path),
        "annotation_sha256": sha256_file(annotation_path),
        "source_manifest_sha256": sha256_file(output / "originals.jsonl"),
        "classifier_manifest_sha256": sha256_file(output / "classifier.jsonl"),
        "detector_manifest_sha256": sha256_file(output / "detector.jsonl"),
        "independent_validation_available": False,
        "independent_test_available": False,
        "evaluation_role": "same_physical_item_development_diagnostic",
        "split_policy": "All related physical items and sessions remain in the training group.",
        "labels": labels,
    }
    write_json(output / "source-report.json", report)
    return report


def verify_sources(prepared: Path) -> dict:
    report = load_json_config(prepared / "source-report.json")
    for filename, key in (
        ("originals.jsonl", "source_manifest_sha256"),
        ("classifier.jsonl", "classifier_manifest_sha256"),
        ("detector.jsonl", "detector_manifest_sha256"),
    ):
        if sha256_file(prepared / filename) != report[key]:
            raise ValueError("prepared source manifest changed")
    root = Path(report["dataset_root"]).resolve()
    records = read_jsonl(prepared / "originals.jsonl")
    if len(records) != report["original_count"] or len({r["image_sha256"] for r in records}) != len(
        records
    ):
        raise ValueError("original budget or uniqueness changed")
    for record in records:
        path = (root / record["image_path"]).resolve()
        path.relative_to(root)
        if sha256_file(path) != record["image_sha256"]:
            raise ValueError("original image changed")
    return report
