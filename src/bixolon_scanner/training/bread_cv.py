from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

from .bread_dataset import (
    _canonical_json,
    _evaluation_image_identity,
    _load_coco,
    _safe_annotation_image,
    audit_bread_dataset,
)

SCHEMA_VERSION = "1.1"
CLASSIFIER_SOURCES = ("single_objects", "single_objects_2")


@lru_cache(maxsize=None)
def difference_hash(path: Path, *, size: int = 8) -> int:
    """Return an orientation-normalized dHash used only for split leakage auditing."""
    if size < 2:
        raise ValueError("difference hash size must be at least two")
    with Image.open(path) as source:
        image = (
            ImageOps.exif_transpose(source)
            .convert("L")
            .resize((size + 1, size), Image.Resampling.LANCZOS)
        )
        values = np.asarray(image, dtype=np.int16)
    bits = values[:, 1:] >= values[:, :-1]
    result = 0
    for value in bits.flat:
        result = (result << 1) | int(value)
    return result


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def assign_perceptual_groups(
    rows: list[dict[str, Any]],
    *,
    maximum_hamming_distance: int = 2,
) -> dict[str, Any]:
    if maximum_hamming_distance < 0:
        raise ValueError("maximum Hamming distance must be non-negative")
    groups = _DisjointSet(len(rows))
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            exact = rows[left]["image_sha256"] == rows[right]["image_sha256"]
            near = (
                hamming_distance(rows[left]["perceptual_hash"], rows[right]["perceptual_hash"])
                <= maximum_hamming_distance
            )
            if exact or near:
                groups.union(left, right)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[groups.find(index)].append(index)
    exact_duplicate_groups = 0
    near_duplicate_groups = 0
    for indices in members.values():
        hashes = {rows[index]["image_sha256"] for index in indices}
        if len(indices) > 1:
            if len(hashes) < len(indices):
                exact_duplicate_groups += 1
            if len(hashes) > 1:
                near_duplicate_groups += 1
        identifier = hashlib.sha256(
            "\n".join(sorted(rows[index]["image_sha256"] for index in indices)).encode("ascii")
        ).hexdigest()[:16]
        for index in indices:
            rows[index]["perceptual_group_id"] = f"phash:{identifier}"
    return {
        "image_count": len(rows),
        "group_count": len(members),
        "exact_duplicate_group_count": exact_duplicate_groups,
        "near_duplicate_group_count": near_duplicate_groups,
        "maximum_hamming_distance": maximum_hamming_distance,
    }


def _balance_features(row: dict[str, Any]) -> Counter[str]:
    features: Counter[str] = Counter()
    features[f"source:{row['evaluation_set']}"] += 1
    features[f"status:{row['expected_image_status']}"] += 1
    features[f"difficulty:{row['difficulty']}"] += 1
    features[f"count:{min(len(row['annotations']), 8)}"] += 1
    for annotation in row["annotations"]:
        features[f"class:{int(annotation['category_id']):02d}"] += 1
    return features


def assign_balanced_folds(rows: list[dict[str, Any]], *, fold_count: int) -> dict[str, Any]:
    if fold_count < 2:
        raise ValueError("cross-validation requires at least two folds")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["perceptual_group_id"])].append(row)
    group_features: dict[str, Counter[str]] = {}
    for group_id, members in grouped.items():
        features: Counter[str] = Counter()
        for member in members:
            features.update(_balance_features(member))
        group_features[group_id] = features
    totals: Counter[str] = Counter()
    for features in group_features.values():
        totals.update(features)
    fold_features = [Counter() for _ in range(fold_count)]
    fold_images = [0] * fold_count
    ordered = sorted(
        grouped,
        key=lambda group_id: (
            -sum(group_features[group_id].values()),
            -len(grouped[group_id]),
            group_id,
        ),
    )
    assignments: dict[str, int] = {}
    for group_id in ordered:
        features = group_features[group_id]

        def cost(fold: int) -> tuple[float, int, int]:
            imbalance = 0.0
            for name, total in totals.items():
                target = totals[name] / fold_count
                for candidate_fold in range(fold_count):
                    projected = fold_features[candidate_fold][name]
                    if candidate_fold == fold:
                        projected += features[name]
                    imbalance += ((projected - target) / max(target, 1.0)) ** 2
            target_images = len(rows) / fold_count
            for candidate_fold in range(fold_count):
                projected_images = fold_images[candidate_fold]
                if candidate_fold == fold:
                    projected_images += len(grouped[group_id])
                imbalance += ((projected_images - target_images) / target_images) ** 2
            return imbalance, fold_images[fold] + len(grouped[group_id]), fold

        selected = min(range(fold_count), key=cost)
        assignments[group_id] = selected
        fold_features[selected].update(features)
        fold_images[selected] += len(grouped[group_id])
    for row in rows:
        row["fold"] = assignments[str(row["perceptual_group_id"])]
    return {
        "fold_count": fold_count,
        "image_counts": fold_images,
        "group_counts": [
            sum(assignment == fold for assignment in assignments.values())
            for fold in range(fold_count)
        ],
        "assignment_key": "perceptual_group_id",
    }


def _detection_records(
    root: Path,
    annotation_name: str,
    *,
    evaluation_set: str,
    image_id_offset: int,
) -> list[dict[str, Any]]:
    annotation_path = root / "annotations" / annotation_name
    coco = _load_coco(annotation_path)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    rows: list[dict[str, Any]] = []
    for image in sorted(coco["images"], key=lambda value: int(value["id"])):
        source_image_id = int(image["id"])
        path = _safe_annotation_image(root, annotation_path, str(image["file_name"]))
        width, height, image_sha256 = _evaluation_image_identity(path)
        relative = path.relative_to(root)
        status = str(image.get("status", "ANNOTATED"))
        annotations = sorted(
            annotations_by_image.get(source_image_id, []), key=lambda value: int(value["id"])
        )
        if status == "RECAPTURE" and annotations:
            raise ValueError("IMAGE_RECAPTURE training rows must not contain object annotations")
        difficulty = (
            relative.parts[1].upper() if evaluation_set == "multi_object_scenes" else "SCAN_LOG"
        )
        rows.append(
            {
                "record_type": "detection",
                "source": "bread_dataset_cross_validation",
                "source_dataset": "bread_dataset",
                "evaluation_set": evaluation_set,
                "difficulty": difficulty,
                "source_image_id": source_image_id,
                "image_id": image_id_offset + source_image_id,
                "image_path": relative.as_posix(),
                "image_sha256": image_sha256,
                "perceptual_hash": difference_hash(path),
                "perceptual_group_id": None,
                "capture_session_id": None,
                "split": "development",
                "fold": None,
                "width": width,
                "height": height,
                "expected_image_status": status,
                "expected_reason_codes": [str(value) for value in image.get("reason_codes", [])],
                "exclude_from_detector_training": False,
                "training_allowed": True,
                "annotations": [
                    {
                        "annotation_id": image_id_offset * 10_000 + int(value["id"]),
                        "category_id": int(value["category_id"]),
                        "bbox_xywh": [float(item) for item in value["bbox"]],
                        "area": float(value["area"]),
                        "iscrowd": int(value.get("iscrowd", 0)),
                    }
                    for value in annotations
                ],
            }
        )
    return rows


def _classifier_records(
    dataset_root: Path,
    *,
    classifier_source: str,
    fold_count: int,
    maximum_hamming_distance: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, source_metadata = audit_bread_dataset(dataset_root, training_source=classifier_source)
    rows = []
    for record in records:
        path = dataset_root / record["image_path"]
        rows.append(
            record
            | {
                "split": "development",
                "fold": None,
                "perceptual_hash": difference_hash(path),
                "perceptual_group_id": None,
            }
        )
    duplicate_audit = assign_perceptual_groups(
        rows, maximum_hamming_distance=maximum_hamming_distance
    )
    # Classifier folds are used to audit support-image leakage. Model selection
    # remains on natural ROI records and never mixes the alternative support source.
    for index, row in enumerate(
        sorted(rows, key=lambda value: (value["source_group"], value["image_path"]))
    ):
        row["fold"] = index % fold_count
    groups: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        groups[str(row["perceptual_group_id"])].add(int(row["fold"]))
    if any(len(folds) != 1 for folds in groups.values()):
        # Keep near duplicates in one fold even when the initial round-robin would split them.
        for group_rows in (
            [row for row in rows if row["perceptual_group_id"] == group_id] for group_id in groups
        ):
            selected = min(int(row["fold"]) for row in group_rows)
            for row in group_rows:
                row["fold"] = selected
    return rows, {
        "source_dataset_version": source_metadata["dataset_version"],
        "image_count": len(rows),
        "duplicate_audit": duplicate_audit,
    }


def _manifest_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def build_bread_cross_validation_registry(
    dataset_root: Path,
    *,
    classifier_source: str = "single_objects",
    fold_count: int = 3,
    maximum_hamming_distance: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = dataset_root.resolve()
    if classifier_source not in CLASSIFIER_SOURCES:
        raise ValueError(
            "zero-error classifier source must be exactly single_objects or single_objects_2"
        )
    classifier_rows, classifier_audit = _classifier_records(
        root,
        classifier_source=classifier_source,
        fold_count=fold_count,
        maximum_hamming_distance=maximum_hamming_distance,
    )
    detector_rows = _detection_records(
        root,
        "multi_object_instances.json",
        evaluation_set="multi_object_scenes",
        image_id_offset=0,
    )
    detector_duplicate_audit = assign_perceptual_groups(
        detector_rows, maximum_hamming_distance=maximum_hamming_distance
    )
    detector_fold_audit = assign_balanced_folds(detector_rows, fold_count=fold_count)
    recapture_count = sum(row["expected_image_status"] == "RECAPTURE" for row in detector_rows)
    digest_payload = {
        "schema_version": SCHEMA_VERSION,
        "classifier_source": classifier_source,
        "fold_count": fold_count,
        "maximum_hamming_distance": maximum_hamming_distance,
        "classifier": [
            (row["image_path"], row["image_sha256"], row["fold"]) for row in classifier_rows
        ],
        "detector": [
            (row["image_path"], row["image_sha256"], row["fold"]) for row in detector_rows
        ],
    }
    digest = hashlib.sha256(_canonical_json(digest_payload).encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": f"bread-1.1-{digest[:12]}",
        "dataset_digest_sha256": digest,
        "classifier": {
            "selected_source": classifier_source,
            "allowed_sources": list(CLASSIFIER_SOURCES),
            "mixed_sources": False,
            "excluded_alternative_sources": [
                name
                for name in (
                    "single_objects",
                    "single_objects_1",
                    "single_objects_2",
                    "single_objects_3",
                )
                if name != classifier_source
            ],
            **classifier_audit,
        },
        "detector": {
            "image_count": len(detector_rows),
            "annotated_image_count": len(detector_rows) - recapture_count,
            "expected_recapture_image_count": recapture_count,
            "annotation_count": sum(len(row["annotations"]) for row in detector_rows),
            "sources": ["multi_object_scenes"],
            "all_images_participate_in_cross_validation": True,
            "duplicate_audit": detector_duplicate_audit,
            "folds": detector_fold_audit,
        },
        "evaluation_policy": {
            "held_out_test_set": False,
            "outer_validation": f"{fold_count}-fold out-of-fold",
            "same_fold_train_validation_overlap_allowed": False,
            "report_raw_and_selective_metrics": True,
            "report_recapture_rates": True,
        },
    }
    return classifier_rows, detector_rows, metadata


def write_bread_cross_validation_registry(
    dataset_root: Path,
    output_dir: Path,
    *,
    classifier_source: str = "single_objects",
    fold_count: int = 3,
    maximum_hamming_distance: int = 2,
) -> str:
    classifier_rows, detector_rows, metadata = build_bread_cross_validation_registry(
        dataset_root,
        classifier_source=classifier_source,
        fold_count=fold_count,
        maximum_hamming_distance=maximum_hamming_distance,
    )
    classifier_manifest = _manifest_text(classifier_rows)
    detector_manifest = _manifest_text(detector_rows)
    metadata["classifier"]["manifest_sha256"] = hashlib.sha256(
        classifier_manifest.encode("utf-8")
    ).hexdigest()
    metadata["detector"]["manifest_sha256"] = hashlib.sha256(
        detector_manifest.encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classifier_manifest.jsonl").write_text(
        classifier_manifest, encoding="utf-8", newline="\n"
    )
    (output_dir / "detector_manifest.jsonl").write_text(
        detector_manifest, encoding="utf-8", newline="\n"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return str(metadata["dataset_version"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the no-test, leakage-audited bread cross-validation registry"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier-source", choices=CLASSIFIER_SOURCES, default="single_objects")
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--maximum-hamming-distance", type=int, default=2)
    args = parser.parse_args()
    print(
        write_bread_cross_validation_registry(
            args.dataset_root,
            args.output_dir,
            classifier_source=args.classifier_source,
            fold_count=args.fold_count,
            maximum_hamming_distance=args.maximum_hamming_distance,
        )
    )


if __name__ == "__main__":
    main()
