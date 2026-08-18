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
from .detector_class_repair import box_iou_xyxy
from .selective_classifier_dataset import recaptured_image_ids


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _by_id(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in _read_jsonl(path)}


def filtered_proposal_indices(
    prediction: dict[str, Any], *, minimum_score: float, minimum_support: int
) -> list[int]:
    scores = np.asarray(prediction["scores"], dtype=np.float64)
    support = np.asarray(prediction["support_counts"], dtype=np.int64)
    if len(scores) != len(support):
        raise ValueError("proposal scores and support counts are not aligned")
    return [
        int(index)
        for index in np.flatnonzero((scores >= minimum_score) & (support >= minimum_support))
    ]


def candidate_mask_context(
    base_boxes: np.ndarray, candidate_box: np.ndarray, *, duplicate_iou: float = 0.9
) -> tuple[list[list[float]], int]:
    base = np.asarray(base_boxes, dtype=np.float32)
    candidate = np.asarray(candidate_box, dtype=np.float32)
    kept = (
        base[box_iou_xyxy(candidate, base) < duplicate_iou]
        if len(base)
        else np.zeros((0, 4), dtype=np.float32)
    )
    boxes = [*kept.tolist(), candidate.tolist()]
    return boxes, len(boxes) - 1


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
    records_by_id = {int(row["image_id"]): row for row in records}
    base = _by_id(args.base_predictions)
    raw = _by_id(args.raw_predictions)
    detector_report = json.loads(args.detector_report.read_text(encoding="utf-8"))
    ambiguity_ids = recaptured_image_ids(detector_report)
    package = load_model_package(args.package)
    classifier = package.metadata.classifier

    tensors = []
    output_records = []
    counts = {}
    for image_id in sorted(ambiguity_ids):
        record = records_by_id[image_id]
        base_boxes = np.asarray(base[image_id]["boxes_xyxy"], dtype=np.float32)
        candidate_indices = filtered_proposal_indices(
            raw[image_id],
            minimum_score=args.minimum_score,
            minimum_support=args.minimum_support,
        )
        counts[str(image_id)] = len(candidate_indices)
        with Image.open(args.dataset_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            for proposal_index in candidate_indices:
                box = raw[image_id]["boxes_xyxy"][proposal_index]
                score = float(raw[image_id]["scores"][proposal_index])
                class_id = int(raw[image_id]["class_ids"][proposal_index])
                detection = Detection(*box, score, class_id)
                mask_boxes, mask_target_index = candidate_mask_context(
                    base_boxes,
                    np.asarray(box, dtype=np.float32),
                    duplicate_iou=args.duplicate_iou,
                )
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
                        "detection_index": proposal_index,
                        "proposal_index": proposal_index,
                        "target": -1,
                        "detector_score": score,
                        "support_count": int(raw[image_id]["support_counts"][proposal_index]),
                        "mask_boxes_xyxy": mask_boxes,
                        "mask_target_index": mask_target_index,
                    }
                )
        finally:
            image.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "evaluation_tensors.npy", np.stack(tensors).astype(np.float32))
    (args.output_dir / "evaluation_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_records),
        encoding="utf-8",
    )
    output = {
        "schema_version": "1.0",
        "evaluation": "bread_ambiguity_proposal_classification_preparation",
        "selection_scope": "development_ambiguity_images_not_locked_test",
        "ambiguity_image_ids": sorted(ambiguity_ids),
        "minimum_score": args.minimum_score,
        "minimum_support": args.minimum_support,
        "duplicate_iou": args.duplicate_iou,
        "proposal_counts": counts,
        "proposal_count": len(output_records),
        "ground_truth_used_for_candidate_filtering": False,
        "locked_test_accessed": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return output


def score(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    training = np.load(args.training_features)
    proposals = np.load(args.proposal_features)
    training_x = np.asarray(training[args.feature_family], dtype=np.float64)
    training_y = np.asarray(training["targets"], dtype=np.int64)
    training_folds = np.asarray(training["folds"], dtype=np.int64)
    proposal_x = np.asarray(proposals[args.feature_family], dtype=np.float64)
    proposal_folds = np.asarray(proposals["folds"], dtype=np.int64)
    proposal_rows = _read_jsonl(args.proposal_records)
    if len(proposal_rows) != len(proposal_x):
        raise ValueError("proposal features and records are not aligned")
    training_x /= np.linalg.norm(training_x, axis=1, keepdims=True).clip(min=1e-12)
    proposal_x /= np.linalg.norm(proposal_x, axis=1, keepdims=True).clip(min=1e-12)

    scores = np.zeros((len(proposal_x), int(training_y.max()) + 1), dtype=np.float64)
    fold_diagnostics = []
    for fold in (0, 1, 2):
        training_mask = training_folds != fold
        validation_mask = proposal_folds == fold
        if not np.any(validation_mask):
            continue
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=args.shrinkage)
        model.fit(training_x[training_mask], training_y[training_mask])
        scores[validation_mask] = model.decision_function(proposal_x[validation_mask])
        fold_diagnostics.append(
            {
                "held_out_fold": fold,
                "training_sample_count": int(np.count_nonzero(training_mask)),
                "proposal_count": int(np.count_nonzero(validation_mask)),
            }
        )
    image_ids = np.asarray([int(row["image_id"]) for row in proposal_rows], dtype=np.int64)
    proposal_indices = np.asarray(
        [int(row["proposal_index"]) for row in proposal_rows], dtype=np.int64
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        scores=scores.astype(np.float32),
        folds=proposal_folds,
        image_ids=image_ids,
        proposal_indices=proposal_indices,
    )
    predicted = np.argmax(scores, axis=1)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_ambiguity_proposal_grouped_oof_class_scores",
        "feature_family": args.feature_family,
        "classifier": "LinearDiscriminantAnalysis",
        "shrinkage": args.shrinkage,
        "proposal_count": len(scores),
        "predicted_class_counts": {
            str(class_id): int(np.count_nonzero(predicted == class_id))
            for class_id in range(scores.shape[1])
        },
        "fold_diagnostics": fold_diagnostics,
        "locked_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _area(box: np.ndarray) -> float:
    return float(np.prod(np.maximum(0.0, box[2:] - box[:2])))


def _proposal_entries(
    raw_prediction: dict[str, Any],
    proposal_indices: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    entries = []
    for proposal_index, class_scores in zip(proposal_indices, scores):
        order = np.argsort(-class_scores, kind="stable")
        index = int(proposal_index)
        entries.append(
            {
                "proposal_index": index,
                "box": np.asarray(raw_prediction["boxes_xyxy"][index], dtype=np.float32),
                "detector_score": float(raw_prediction["scores"][index]),
                "support_count": int(raw_prediction["support_counts"][index]),
                "predicted_class": int(order[0]),
                "class_margin": float(class_scores[order[0]] - class_scores[order[1]]),
            }
        )
    return entries


def select_class_verified_prediction(
    base_prediction: dict[str, Any],
    raw_prediction: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    minimum_support: int,
    base_match_iou: float,
    group_relation_iou: float,
    group_area_ratio: float,
    group_margin_ratio: float,
    group_novel_margin: float,
    group_minimum_score: float,
    independent_maximum_iou: float,
    independent_margin: float,
    independent_minimum_score: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_boxes = np.asarray(base_prediction["boxes_xyxy"], dtype=np.float32)
    raw_boxes = np.asarray(raw_prediction["boxes_xyxy"], dtype=np.float32)
    entries_by_index = {int(row["proposal_index"]): row for row in entries}
    mapped = []
    unsupported_count = 0
    for box in base_boxes:
        overlaps = box_iou_xyxy(box, raw_boxes)
        raw_index = int(np.argmax(overlaps))
        if float(overlaps[raw_index]) < base_match_iou:
            raise ValueError("selected base box cannot be mapped to the raw proposal union")
        entry = entries_by_index.get(raw_index)
        if entry is None or int(raw_prediction["support_counts"][raw_index]) < minimum_support:
            unsupported_count += 1
            continue
        mapped.append(entry)

    by_class: dict[int, dict[str, Any]] = {}
    duplicate_count = 0
    for entry in mapped:
        class_id = int(entry["predicted_class"])
        current = by_class.get(class_id)
        if current is None or entry["detector_score"] > current["detector_score"]:
            if current is not None:
                duplicate_count += 1
            by_class[class_id] = entry
        else:
            duplicate_count += 1
    selected = list(by_class.values())

    group_splits = []
    used_indices = {int(row["proposal_index"]) for row in selected}
    for base_entry in list(selected):
        base_box = base_entry["box"]
        base_area = _area(base_box)
        if base_area <= 0.0:
            continue
        alternatives = []
        novel = []
        current_classes = {int(row["predicted_class"]) for row in selected}
        for candidate in entries:
            if int(candidate["proposal_index"]) in used_indices:
                continue
            relation = float(box_iou_xyxy(candidate["box"], base_box[None, :])[0])
            if relation < group_relation_iou:
                continue
            if _area(candidate["box"]) > base_area * group_area_ratio:
                continue
            if int(candidate["predicted_class"]) == int(base_entry["predicted_class"]):
                if candidate["class_margin"] >= base_entry["class_margin"] * group_margin_ratio:
                    alternatives.append(candidate)
            elif (
                int(candidate["predicted_class"]) not in current_classes
                and candidate["class_margin"] >= group_novel_margin
                and candidate["detector_score"] >= group_minimum_score
            ):
                novel.append(candidate)
        if not alternatives or not novel:
            continue
        alternative = max(alternatives, key=lambda row: row["detector_score"])
        compatible_novel = [
            row
            for row in novel
            if float(box_iou_xyxy(row["box"], alternative["box"][None, :])[0]) < 0.5
        ]
        if not compatible_novel:
            continue
        novel_entry = max(compatible_novel, key=lambda row: row["detector_score"])
        selected.remove(base_entry)
        selected.extend((alternative, novel_entry))
        used_indices.discard(int(base_entry["proposal_index"]))
        used_indices.update(
            (int(alternative["proposal_index"]), int(novel_entry["proposal_index"]))
        )
        group_splits.append(
            {
                "removed_proposal_index": int(base_entry["proposal_index"]),
                "replacement_proposal_indices": [
                    int(alternative["proposal_index"]),
                    int(novel_entry["proposal_index"]),
                ],
                "replacement_classes": [
                    int(alternative["predicted_class"]),
                    int(novel_entry["predicted_class"]),
                ],
            }
        )

    current_classes = {int(row["predicted_class"]) for row in selected}
    current_boxes = np.asarray([row["box"] for row in selected], dtype=np.float32)
    independent = [
        row
        for row in entries
        if int(row["proposal_index"]) not in used_indices
        and int(row["predicted_class"]) not in current_classes
        and row["class_margin"] >= independent_margin
        and row["detector_score"] >= independent_minimum_score
        and (
            not len(current_boxes)
            or float(box_iou_xyxy(row["box"], current_boxes).max()) < independent_maximum_iou
        )
    ]
    independent_addition = None
    if independent:
        best_class = int(max(independent, key=lambda row: row["class_margin"])["predicted_class"])
        same_class = [row for row in independent if int(row["predicted_class"]) == best_class]
        independent_addition = max(same_class, key=lambda row: _area(row["box"]))
        selected.append(independent_addition)

    selected.sort(key=lambda row: row["detector_score"], reverse=True)
    output = {
        "image_id": base_prediction["image_id"],
        "boxes_xyxy": [row["box"].tolist() for row in selected],
        "scores": [float(row["detector_score"]) for row in selected],
        "class_ids": [int(row["predicted_class"]) for row in selected],
    }
    diagnostics = {
        "image_id": int(base_prediction["image_id"]),
        "input_base_count": len(base_boxes),
        "unsupported_base_removed_count": unsupported_count,
        "duplicate_base_removed_count": duplicate_count,
        "group_splits": group_splits,
        "independent_addition": (
            None
            if independent_addition is None
            else {
                "proposal_index": int(independent_addition["proposal_index"]),
                "predicted_class": int(independent_addition["predicted_class"]),
                "class_margin": float(independent_addition["class_margin"]),
            }
        ),
        "output_count": len(selected),
    }
    return output, diagnostics


def select(args: argparse.Namespace) -> dict[str, Any]:
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
    base = _by_id(args.base_predictions)
    raw = _by_id(args.raw_predictions)
    proposal_rows = _read_jsonl(args.proposal_records)
    cache = np.load(args.proposal_scores)
    if len(proposal_rows) != len(cache["scores"]):
        raise ValueError("proposal score cache and records are not aligned")
    by_image: dict[int, list[int]] = {}
    for position, row in enumerate(proposal_rows):
        by_image.setdefault(int(row["image_id"]), []).append(position)

    detector_report = json.loads(args.detector_report.read_text(encoding="utf-8"))
    ambiguity_ids = recaptured_image_ids(detector_report)
    outputs = []
    diagnostics = []
    for record in records:
        image_id = int(record["image_id"])
        if image_id not in ambiguity_ids:
            outputs.append(base[image_id])
            continue
        positions = by_image[image_id]
        entries = _proposal_entries(
            raw[image_id],
            np.asarray([proposal_rows[index]["proposal_index"] for index in positions]),
            np.asarray(cache["scores"])[positions],
        )
        prediction, image_diagnostics = select_class_verified_prediction(
            base[image_id],
            raw[image_id],
            entries,
            minimum_support=args.minimum_support,
            base_match_iou=args.base_match_iou,
            group_relation_iou=args.group_relation_iou,
            group_area_ratio=args.group_area_ratio,
            group_margin_ratio=args.group_margin_ratio,
            group_novel_margin=args.group_novel_margin,
            group_minimum_score=args.group_minimum_score,
            independent_maximum_iou=args.independent_maximum_iou,
            independent_margin=args.independent_margin,
            independent_minimum_score=args.independent_minimum_score,
        )
        outputs.append(prediction)
        diagnostics.append(image_diagnostics)

    metrics = _metrics(
        records,
        outputs,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=600,
    )
    errors = detection_error_rows(
        records,
        outputs,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=args.match_iou_threshold,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_fixed_ensemble_class_verified_proposal_selection",
        "selection_scope": "development_policy_selection_not_locked_test",
        "ambiguity_image_count": len(ambiguity_ids),
        "policy": {
            "minimum_support": args.minimum_support,
            "base_match_iou": args.base_match_iou,
            "group_relation_iou": args.group_relation_iou,
            "group_area_ratio": args.group_area_ratio,
            "group_margin_ratio": args.group_margin_ratio,
            "group_novel_margin": args.group_novel_margin,
            "group_minimum_score": args.group_minimum_score,
            "independent_maximum_iou": args.independent_maximum_iou,
            "independent_margin": args.independent_margin,
            "independent_minimum_score": args.independent_minimum_score,
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
        "error_images": errors,
        "segmentation_image_count": len(records),
        "image_recapture_count": 0,
        "recaptured_image_ids": [],
        "locked_test_accessed": False,
        "promotion_ready": False,
        "promotion_blocker": "final model serialization, Worker integration, and locked test pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row) + "\n" for row in outputs), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify bread ambiguity proposals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--folds", type=int, nargs="+", required=True)
    prepare_parser.add_argument("--difficulties", nargs="+")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--base-predictions", type=Path, required=True)
    prepare_parser.add_argument("--raw-predictions", type=Path, required=True)
    prepare_parser.add_argument("--detector-report", type=Path, required=True)
    prepare_parser.add_argument("--package", type=Path, required=True)
    prepare_parser.add_argument("--minimum-score", type=float, default=0.02)
    prepare_parser.add_argument("--minimum-support", type=int, default=3)
    prepare_parser.add_argument("--duplicate-iou", type=float, default=0.9)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--training-features", type=Path, required=True)
    score_parser.add_argument("--proposal-features", type=Path, required=True)
    score_parser.add_argument("--proposal-records", type=Path, required=True)
    score_parser.add_argument("--feature-family", choices=["raw", "adapted"], default="adapted")
    score_parser.add_argument("--shrinkage", type=float, default=0.003)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--report", type=Path, required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--manifest", type=Path, required=True)
    select_parser.add_argument("--folds", type=int, nargs="+", required=True)
    select_parser.add_argument("--difficulties", nargs="+")
    select_parser.add_argument("--base-predictions", type=Path, required=True)
    select_parser.add_argument("--raw-predictions", type=Path, required=True)
    select_parser.add_argument("--detector-report", type=Path, required=True)
    select_parser.add_argument("--proposal-records", type=Path, required=True)
    select_parser.add_argument("--proposal-scores", type=Path, required=True)
    select_parser.add_argument("--minimum-support", type=int, default=3)
    select_parser.add_argument("--base-match-iou", type=float, default=0.9)
    select_parser.add_argument("--group-relation-iou", type=float, default=0.3)
    select_parser.add_argument("--group-area-ratio", type=float, default=0.8)
    select_parser.add_argument("--group-margin-ratio", type=float, default=1.5)
    select_parser.add_argument("--group-novel-margin", type=float, default=1500.0)
    select_parser.add_argument("--group-minimum-score", type=float, default=0.04)
    select_parser.add_argument("--independent-maximum-iou", type=float, default=0.3)
    select_parser.add_argument("--independent-margin", type=float, default=4000.0)
    select_parser.add_argument("--independent-minimum-score", type=float, default=0.05)
    select_parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    select_parser.add_argument("--predictions-output", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "score":
        score(args)
    else:
        select(args)


if __name__ == "__main__":
    main()
