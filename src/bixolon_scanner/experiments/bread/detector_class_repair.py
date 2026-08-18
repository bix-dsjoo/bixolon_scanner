from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.model_package import load_model_package
from ...evaluation.detected_roi_dataset import crop_tensor
from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...training.data import read_manifest
from .selective_classifier_dataset import recaptured_image_ids


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return IoU between one XYXY box and an aligned array of XYXY boxes."""
    candidate = np.asarray(box, dtype=np.float32)
    references = np.asarray(boxes, dtype=np.float32)
    if candidate.shape != (4,) or references.ndim != 2 or references.shape[1:] != (4,):
        raise ValueError("boxes must use XYXY shape")
    top_left = np.maximum(candidate[:2], references[:, :2])
    bottom_right = np.minimum(candidate[2:], references[:, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    candidate_area = float(np.prod(np.maximum(0.0, candidate[2:] - candidate[:2])))
    reference_areas = np.prod(np.maximum(0.0, references[:, 2:] - references[:, :2]), axis=1)
    union = candidate_area + reference_areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def select_non_overlapping_candidate(
    base_prediction: dict[str, Any],
    ranked_prediction: dict[str, Any],
    *,
    maximum_iou: float,
) -> int | None:
    """Select the highest-score proposal spatially independent of every base box."""
    if not 0.0 <= maximum_iou <= 1.0:
        raise ValueError("maximum_iou must be between zero and one")
    base_boxes = np.asarray(base_prediction["boxes_xyxy"], dtype=np.float32)
    ranked_boxes = np.asarray(ranked_prediction["boxes_xyxy"], dtype=np.float32)
    ranked_scores = np.asarray(ranked_prediction["scores"], dtype=np.float64)
    if len(ranked_boxes) != len(ranked_scores):
        raise ValueError("ranked boxes and scores are not aligned")
    if not len(ranked_boxes):
        return None
    if not len(base_boxes):
        return int(np.argmax(ranked_scores))
    eligible = [
        index
        for index, box in enumerate(ranked_boxes)
        if float(box_iou_xyxy(box, base_boxes).max()) < maximum_iou
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda index: (ranked_scores[index], -index))


def candidate_is_novel(base_classes: np.ndarray, candidate_class: int) -> bool:
    """Use classifier predictions only to enforce the bread one-class-per-image contract."""
    values = np.asarray(base_classes, dtype=np.int64)
    return not bool(np.any(values == int(candidate_class)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _predictions_by_id(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in _read_jsonl(path)}


def _selected_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    return [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    records = _selected_records(args)
    records_by_id = {int(row["image_id"]): row for row in records}
    base = _predictions_by_id(args.base_predictions)
    ranked = _predictions_by_id(args.ranked_predictions)
    report = json.loads(args.detector_report.read_text(encoding="utf-8"))
    ambiguity_ids = recaptured_image_ids(report)
    if not ambiguity_ids <= records_by_id.keys():
        raise ValueError("ambiguity images are outside the selected manifest records")
    package = load_model_package(args.package)
    classifier = package.metadata.classifier

    tensors: list[np.ndarray] = []
    output_records = []
    verification_predictions = []
    candidate_diagnostics = []
    for image_id in sorted(ambiguity_ids):
        record = records_by_id[image_id]
        base_prediction = base[image_id]
        ranked_prediction = ranked[image_id]
        candidate_index = select_non_overlapping_candidate(
            base_prediction,
            ranked_prediction,
            maximum_iou=args.maximum_iou,
        )
        if candidate_index is None:
            raise ValueError(f"ambiguity image {image_id} has no independent candidate")
        candidate_box = ranked_prediction["boxes_xyxy"][candidate_index]
        candidate_score = float(ranked_prediction["scores"][candidate_index])
        union_prediction = {
            "image_id": image_id,
            "boxes_xyxy": [*base_prediction["boxes_xyxy"], candidate_box],
            "scores": [*base_prediction["scores"], candidate_score],
            "class_ids": [
                *base_prediction["class_ids"],
                int(ranked_prediction["class_ids"][candidate_index]),
            ],
        }
        verification_predictions.append(union_prediction)
        candidate_diagnostics.append(
            {
                "image_id": image_id,
                "fold": int(record["fold"]),
                "candidate_index": candidate_index,
                "candidate_score": candidate_score,
                "candidate_box_xyxy": candidate_box,
                "base_detection_count": len(base_prediction["scores"]),
            }
        )
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                union_prediction["boxes_xyxy"],
                union_prediction["scores"],
                union_prediction["class_ids"],
            )
        ]
        with Image.open(args.dataset_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            for detection_index, detection in enumerate(detections):
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
                        "target": -1,
                        "is_candidate": detection_index == len(detections) - 1,
                    }
                )
        finally:
            image.close()

    duplicate_gt_images = [
        int(row["image_id"])
        for row in records
        if len({int(item["category_id"]) for item in row["annotations"]}) != len(row["annotations"])
    ]
    if duplicate_gt_images:
        raise ValueError("bread unique-class contract is not satisfied by the selected records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "evaluation_tensors.npy", np.stack(tensors).astype(np.float32))
    (args.output_dir / "evaluation_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_records),
        encoding="utf-8",
    )
    (args.output_dir / "verification_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verification_predictions),
        encoding="utf-8",
    )
    output = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_class_novelty_repair_preparation",
        "selection_scope": "grouped_development_oof_not_locked_test",
        "ambiguity_image_count": len(ambiguity_ids),
        "ambiguity_image_ids": sorted(ambiguity_ids),
        "candidate_selection": {
            "rule": "highest_score_with_iou_below_all_base_boxes",
            "maximum_iou_exclusive": args.maximum_iou,
            "ground_truth_used": False,
        },
        "candidate_diagnostics": candidate_diagnostics,
        "unique_class_contract_checked_image_count": len(records),
        "unique_class_contract_violation_count": 0,
        "classifier_roi_count": len(output_records),
        "locked_test_accessed": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return output


def _fit_lda_scores(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    model.fit(training_features, training_targets)
    return model.decision_function(validation_features)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    training = np.load(args.training_features)
    verification = np.load(args.verification_features)
    training_x = np.asarray(training[args.feature_family], dtype=np.float64)
    training_y = np.asarray(training["targets"], dtype=np.int64)
    training_folds = np.asarray(training["folds"], dtype=np.int64)
    verification_x = np.asarray(verification[args.feature_family], dtype=np.float64)
    verification_folds = np.asarray(verification["folds"], dtype=np.int64)
    training_x /= np.linalg.norm(training_x, axis=1, keepdims=True).clip(min=1e-12)
    verification_x /= np.linalg.norm(verification_x, axis=1, keepdims=True).clip(min=1e-12)
    if set(np.unique(training_folds)) != {0, 1, 2}:
        raise ValueError("repair verifier requires grouped folds 0, 1, and 2")

    scores = np.zeros((len(verification_x), int(training_y.max()) + 1), dtype=np.float64)
    for fold in (0, 1, 2):
        validation = verification_folds == fold
        if not np.any(validation):
            continue
        scores[validation] = _fit_lda_scores(
            training_x[training_folds != fold],
            training_y[training_folds != fold],
            verification_x[validation],
            shrinkage=args.shrinkage,
        )

    rows = _read_jsonl(args.verification_records)
    if len(rows) != len(scores):
        raise ValueError("verification features and records are not aligned")
    union = _predictions_by_id(args.verification_predictions)
    base = _predictions_by_id(args.base_predictions)
    predicted_classes = np.argmax(scores, axis=1)
    row_classes: dict[int, list[tuple[dict[str, Any], int]]] = {}
    for row, predicted_class in zip(rows, predicted_classes):
        row_classes.setdefault(int(row["image_id"]), []).append((row, int(predicted_class)))

    decisions = []
    repaired = dict(base)
    for image_id, classified_rows in row_classes.items():
        candidate_rows = [item for item in classified_rows if item[0]["is_candidate"]]
        base_rows = [item for item in classified_rows if not item[0]["is_candidate"]]
        if len(candidate_rows) != 1:
            raise ValueError(f"image {image_id} must have exactly one candidate")
        candidate_row, candidate_class = candidate_rows[0]
        base_classes = np.asarray([value for _, value in base_rows], dtype=np.int64)
        accepted = candidate_is_novel(base_classes, candidate_class)
        if accepted:
            repaired[image_id] = union[image_id]
        candidate_index = int(candidate_row["detection_index"])
        decisions.append(
            {
                "image_id": image_id,
                "fold": int(candidate_row["fold"]),
                "base_predicted_classes": base_classes.tolist(),
                "candidate_predicted_class": candidate_class,
                "candidate_score_margin": float(
                    np.partition(scores[int(candidate_row["tensor_index"])], -2)[-1]
                    - np.partition(scores[int(candidate_row["tensor_index"])], -2)[-2]
                ),
                "candidate_detector_score": float(union[image_id]["scores"][candidate_index]),
                "accepted": accepted,
                "reason": "NOVEL_CLASS" if accepted else "DUPLICATE_CLASS",
            }
        )

    records = _selected_records(args)
    repaired_rows = [repaired[int(row["image_id"])] for row in records]
    metrics = _metrics(
        records,
        repaired_rows,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=600,
    )
    errors = detection_error_rows(
        records,
        repaired_rows,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=args.match_iou_threshold,
    )
    output = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_class_novelty_repair_grouped_oof",
        "selection_scope": "three_fold_grouped_development_oof_not_locked_test",
        "feature_family": args.feature_family,
        "classifier": "LinearDiscriminantAnalysis",
        "shrinkage": args.shrinkage,
        "training_source": "accepted_ROIs_from_other_two_folds_only",
        "candidate_policy": "accept_highest_independent_candidate_only_for_novel_top1_class",
        "decisions": sorted(decisions, key=lambda row: row["image_id"]),
        "metrics": metrics,
        "error_images": errors,
        "segmentation_image_count": len(records),
        "image_recapture_count": 0,
        "recaptured_image_ids": [],
        "locked_test_accessed": False,
        "promotion_ready": False,
        "promotion_blocker": "full classifier OOF, final package, parity, latency, and locked test pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row) + "\n" for row in repaired_rows), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return output


def _add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify detector recovery by class novelty")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    _add_scope_arguments(prepare_parser)
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--base-predictions", type=Path, required=True)
    prepare_parser.add_argument("--ranked-predictions", type=Path, required=True)
    prepare_parser.add_argument("--detector-report", type=Path, required=True)
    prepare_parser.add_argument("--package", type=Path, required=True)
    prepare_parser.add_argument("--maximum-iou", type=float, default=0.3)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    _add_scope_arguments(verify_parser)
    verify_parser.add_argument("--training-features", type=Path, required=True)
    verify_parser.add_argument("--verification-features", type=Path, required=True)
    verify_parser.add_argument("--verification-records", type=Path, required=True)
    verify_parser.add_argument("--verification-predictions", type=Path, required=True)
    verify_parser.add_argument("--base-predictions", type=Path, required=True)
    verify_parser.add_argument("--feature-family", choices=["raw", "adapted"], default="adapted")
    verify_parser.add_argument("--shrinkage", type=float, default=0.001)
    verify_parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    verify_parser.add_argument("--predictions-output", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
