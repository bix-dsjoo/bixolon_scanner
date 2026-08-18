from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.model_package import load_model_package
from ...evaluation.detected_roi_dataset import crop_tensor, match_detections
from ...pipeline.ports import Detection
from ...training.data import read_manifest


def recaptured_image_ids(report: dict[str, Any]) -> set[int]:
    fixed_union = report.get("recaptured_image_ids")
    if fixed_union is not None:
        return {int(value) for value in fixed_union}
    diagnostic = report.get("disagreement_recapture_diagnostic")
    if not diagnostic:
        raise ValueError("detector report has no recapture gate result")
    return {int(value) for value in diagnostic["recaptured_image_ids"]}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    predictions = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    detector_report = json.loads(args.detector_report.read_text(encoding="utf-8"))
    recaptured = recaptured_image_ids(detector_report)
    package = load_model_package(args.package)
    classifier = package.metadata.classifier
    tensors = []
    output_records = []
    accepted_image_count = 0
    accepted_truth_count = 0
    for record in records:
        image_id = int(record["image_id"])
        if image_id in recaptured:
            continue
        prediction = predictions[image_id]
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                prediction["boxes_xyxy"],
                prediction["scores"],
                prediction["class_ids"],
            )
        ]
        matches = match_detections(
            detections,
            record["annotations"],
            match_iou_threshold=args.match_iou_threshold,
        )
        if len(detections) != len(record["annotations"]) or len(matches) != len(
            record["annotations"]
        ):
            raise ValueError(f"accepted detector image {image_id} is not FP/FN exact")
        with Image.open(args.dataset_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            for detection_index, detection in enumerate(detections):
                annotation_index, overlap = matches[detection_index]
                annotation = record["annotations"][annotation_index]
                tensors.append(
                    crop_tensor(
                        image,
                        detection,
                        crop_margin_ratio=classifier.crop_margin_ratio,
                        input_size=classifier.input_size[0],
                    )
                )
                output_records.append(
                    {
                        "tensor_index": len(tensors) - 1,
                        "image_id": image_id,
                        "fold": int(record["fold"]),
                        "group_id": str(record["perceptual_group_id"]),
                        "detection_index": detection_index,
                        "target": int(annotation["category_id"]) - 1,
                        "match_iou": overlap,
                    }
                )
        finally:
            image.close()
        accepted_image_count += 1
        accepted_truth_count += len(record["annotations"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / "evaluation_tensors.npy",
        np.stack(tensors).astype(np.float32),
    )
    (args.output_dir / "evaluation_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_records),
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_selective_detector_classifier_dataset",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "annotated_image_count": len(records),
        "detector_recaptured_image_count": len(recaptured),
        "detector_recaptured_image_ids": sorted(recaptured),
        "detector_recapture_rate": len(recaptured) / len(records),
        "accepted_image_count": accepted_image_count,
        "accepted_ground_truth_count": accepted_truth_count,
        "classifier_roi_count": len(output_records),
        "classifier_version": classifier.version,
        "crop_margin_ratio": classifier.crop_margin_ratio,
        "input_size": list(classifier.input_size),
        "match_iou_threshold": args.match_iou_threshold,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare classifier ROIs after the selective detector gate"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
