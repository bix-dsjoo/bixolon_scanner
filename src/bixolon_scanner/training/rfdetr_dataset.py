from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_group_folds(records: list[dict[str, Any]]) -> int:
    folds_by_group: dict[str, set[int]] = {}
    for row in records:
        group = str(row.get("perceptual_group_id") or row.get("capture_session_id") or "")
        if not group:
            raise ValueError(f"detector record {row.get('image_id')} has no physical group")
        folds_by_group.setdefault(group, set()).add(int(row["fold"]))
    overlap = sum(len(folds) > 1 for folds in folds_by_group.values())
    if overlap:
        raise ValueError(f"detector group-aware fold leakage detected for {overlap} groups")
    return len(folds_by_group)


def _categories(class_agnostic: bool) -> list[dict[str, Any]]:
    if class_agnostic:
        return [{"id": 1, "name": "bread_object", "supercategory": "bread"}]
    return [
        {
            "id": category_id,
            "name": f"bread_{category_id:02d}",
            "supercategory": "bread",
        }
        for category_id in range(1, 21)
    ]


def _hardlink_images(
    records: list[dict[str, Any]], source_root: Path, split_dir: Path
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for row in records:
        relative_path = Path(str(row["image_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe detector image path: {relative_path}")
        source = source_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"detector image does not exist: {source}")
        destination = split_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise FileExistsError(f"detector hard-link destination is occupied: {destination}")
        else:
            os.link(source, destination)
        images.append(
            {
                "id": int(row["image_id"]),
                "file_name": relative_path.as_posix(),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "status": str(row.get("expected_image_status", "ANNOTATED")),
                "reason_codes": [str(value) for value in row.get("expected_reason_codes", [])],
            }
        )
    return images


def _coco_payload(
    records: list[dict[str, Any]],
    source_root: Path,
    split_dir: Path,
    *,
    class_agnostic: bool,
) -> dict[str, Any]:
    images = _hardlink_images(records, source_root, split_dir)
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for row in records:
        for annotation in row["annotations"]:
            bbox = [float(value) for value in annotation["bbox_xywh"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(row["image_id"]),
                    "category_id": 1 if class_agnostic else int(annotation["category_id"]),
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": _categories(class_agnostic),
    }


def export_rfdetr_coco_fold(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    validation_fold: int,
    class_agnostic: bool = False,
) -> dict[str, Any]:
    records = _read_manifest(manifest_path)
    if not records:
        raise ValueError("RF-DETR export requires a non-empty detector manifest")
    group_count = _validate_group_folds(records)
    folds = {int(row["fold"]) for row in records}
    if validation_fold not in folds:
        raise ValueError(f"validation fold {validation_fold} is absent from detector manifest")

    training = [row for row in records if int(row["fold"]) != validation_fold]
    validation = [row for row in records if int(row["fold"]) == validation_fold]
    train_dir = output_dir / "train"
    validation_dir = output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    train_payload = _coco_payload(training, dataset_root, train_dir, class_agnostic=class_agnostic)
    validation_payload = _coco_payload(
        validation, dataset_root, validation_dir, class_agnostic=class_agnostic
    )
    train_annotations = train_dir / "_annotations.coco.json"
    validation_annotations = validation_dir / "_annotations.coco.json"
    train_annotations.write_text(
        json.dumps(train_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    validation_annotations.write_text(
        json.dumps(validation_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    report = {
        "schema_version": "1.0",
        "format": "rfdetr_roboflow_coco",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "dataset_root": str(dataset_root),
        "validation_fold": validation_fold,
        "observed_folds": sorted(folds),
        "group_count": group_count,
        "group_fold_overlap_count": 0,
        "training_image_count": len(training),
        "validation_image_count": len(validation),
        "training_annotation_count": len(train_payload["annotations"]),
        "validation_annotation_count": len(validation_payload["annotations"]),
        "training_empty_image_count": sum(not row["annotations"] for row in training),
        "validation_empty_image_count": sum(not row["annotations"] for row in validation),
        "class_agnostic": class_agnostic,
        "image_materialization": "ntfs_hardlink",
        "held_out_test_set": False,
        "evaluation_images_used_for_development_cross_validation": True,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a group-aware RF-DETR COCO fold")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, required=True)
    parser.add_argument("--class-agnostic", action="store_true")
    args = parser.parse_args()
    report = export_rfdetr_coco_fold(
        args.manifest,
        args.dataset_root,
        args.output_dir,
        validation_fold=args.validation_fold,
        class_agnostic=args.class_agnostic,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
