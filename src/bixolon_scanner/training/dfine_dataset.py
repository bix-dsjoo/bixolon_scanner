from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _coco(records: list[dict[str, Any]]) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for row in records:
        image_id = int(row["image_id"])
        images.append(
            {
                "id": image_id,
                "file_name": str(row["image_path"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )
        for annotation in row["annotations"]:
            bbox = [float(value) for value in annotation["bbox_xywh"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(annotation["category_id"]) - 1,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": category_index,
                "name": f"bread_{category_index + 1:02d}",
                "supercategory": "bread",
            }
            for category_index in range(20)
        ],
    }


def export_dfine_coco_splits(
    manifest_path: Path,
    output_dir: Path,
    *,
    validation_fold: int,
) -> dict[str, Any]:
    records = _read_manifest(manifest_path)
    training = [row for row in records if int(row["fold"]) != validation_fold]
    validation = [row for row in records if int(row["fold"]) == validation_fold]
    if not training or not validation:
        raise ValueError("D-FINE export requires non-empty training and validation splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "instances_train.json"
    validation_path = output_dir / "instances_validation.json"
    train_path.write_text(json.dumps(_coco(training), separators=(",", ":")), encoding="utf-8")
    validation_path.write_text(
        json.dumps(_coco(validation), separators=(",", ":")), encoding="utf-8"
    )
    report = {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path),
        "validation_fold": validation_fold,
        "training_image_count": len(training),
        "validation_image_count": len(validation),
        "training_annotation_count": sum(len(row["annotations"]) for row in training),
        "validation_annotation_count": sum(len(row["annotations"]) for row in validation),
        "evaluation_images_used": False,
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
    args = parser.parse_args()
    print(
        json.dumps(
            export_dfine_coco_splits(
                args.manifest,
                args.output_dir,
                validation_fold=args.validation_fold,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
