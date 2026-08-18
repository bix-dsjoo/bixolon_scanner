from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _coco(records: list[dict[str, Any]], *, class_agnostic: bool = False) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for row in records:
        image_id = int(row["image_id"])
        image = {
            "id": image_id,
            "file_name": str(row["image_path"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        if row.get("expected_image_status") is not None:
            image["status"] = str(row["expected_image_status"])
            image["reason_codes"] = [str(value) for value in row.get("expected_reason_codes", [])]
        images.append(image)
        for annotation in row["annotations"]:
            bbox = [float(value) for value in annotation["bbox_xywh"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": (0 if class_agnostic else int(annotation["category_id"]) - 1),
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": (
            [{"id": 0, "name": "bread_object", "supercategory": "bread"}]
            if class_agnostic
            else [
                {
                    "id": category_index,
                    "name": f"bread_{category_index + 1:02d}",
                    "supercategory": "bread",
                }
                for category_index in range(20)
            ]
        ),
    }


def export_dfine_coco_splits(
    manifest_path: Path,
    output_dir: Path,
    *,
    validation_fold: int,
    class_agnostic: bool = False,
    evaluation_images_used: bool = False,
) -> dict[str, Any]:
    records = _read_manifest(manifest_path)
    training = [row for row in records if int(row["fold"]) != validation_fold]
    validation = [row for row in records if int(row["fold"]) == validation_fold]
    if not training or not validation:
        raise ValueError("D-FINE export requires non-empty training and validation splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "instances_train.json"
    validation_path = output_dir / "instances_validation.json"
    all_path = output_dir / "instances_all.json"
    train_path.write_text(
        json.dumps(_coco(training, class_agnostic=class_agnostic), separators=(",", ":")),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(_coco(validation, class_agnostic=class_agnostic), separators=(",", ":")),
        encoding="utf-8",
    )
    all_path.write_text(
        json.dumps(_coco(records, class_agnostic=class_agnostic), separators=(",", ":")),
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path),
        "validation_fold": validation_fold,
        "training_image_count": len(training),
        "validation_image_count": len(validation),
        "training_annotation_count": sum(len(row["annotations"]) for row in training),
        "validation_annotation_count": sum(len(row["annotations"]) for row in validation),
        "all_image_count": len(records),
        "all_annotation_count": sum(len(row["annotations"]) for row in records),
        "evaluation_images_used": evaluation_images_used,
        "class_agnostic": class_agnostic,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export synthetic bread data for D-FINE")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=2)
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument(
        "--allow-evaluation-images",
        action="store_true",
        help="Record the explicit no-held-out-test policy for natural-image CV training",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export_dfine_coco_splits(
                args.manifest,
                args.output_dir,
                validation_fold=args.validation_fold,
                class_agnostic=args.class_agnostic,
                evaluation_images_used=args.allow_evaluation_images,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
