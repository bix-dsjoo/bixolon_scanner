from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor

from ...contracts.model_package import load_model_package
from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import ClassificationResult, Detection
from ...runtime.onnx import OnnxClassifier
from ...training.data import read_manifest
from .proposal_ranker import (
    proposal_assignment_labels,
    proposal_intersection_matrix,
    proposal_iou_matrix,
    proposal_labels,
    proposal_qualities,
    select_ranked_predictions,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def proposal_context_features(prediction: dict[str, Any]) -> np.ndarray:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    if not len(boxes):
        return np.zeros((0, 18), dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    support = np.asarray(
        prediction.get("support_counts", np.ones(len(boxes))),
        dtype=np.float32,
    )
    overlap = proposal_iou_matrix(boxes, boxes)
    np.fill_diagonal(overlap, 0.0)
    intersection = proposal_intersection_matrix(boxes, boxes)
    np.fill_diagonal(intersection, 0.0)
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    contains = np.divide(
        intersection,
        areas[None, :],
        out=np.zeros_like(intersection),
        where=areas[None, :] > 0.0,
    )
    inside = np.divide(
        intersection,
        areas[:, None],
        out=np.zeros_like(intersection),
        where=areas[:, None] > 0.0,
    )
    higher_score = scores[None, :] > scores[:, None]
    higher_support = support[None, :] > support[:, None]
    neighboring_support = np.where(overlap >= 0.3, support[None, :], 0.0)
    return np.column_stack(
        (
            overlap.max(axis=1),
            (overlap >= 0.3).sum(axis=1),
            (overlap >= 0.5).sum(axis=1),
            (overlap >= 0.7).sum(axis=1),
            np.where(higher_score, overlap, 0.0).max(axis=1),
            np.where(higher_support, overlap, 0.0).max(axis=1),
            contains.max(axis=1),
            (contains >= 0.8).sum(axis=1),
            (contains >= 0.9).sum(axis=1),
            ((contains >= 0.8) & higher_score).sum(axis=1),
            ((contains >= 0.8) & higher_support).sum(axis=1),
            inside.max(axis=1),
            (inside >= 0.8).sum(axis=1),
            (inside >= 0.9).sum(axis=1),
            ((inside >= 0.8) & higher_score).sum(axis=1),
            ((inside >= 0.8) & higher_support).sum(axis=1),
            neighboring_support.max(axis=1),
            (overlap * support[None, :]).sum(axis=1),
        )
    ).astype(np.float32)


def verifier_features(
    record: dict[str, Any],
    prediction: dict[str, Any],
    logits: np.ndarray,
    ranking_logits: np.ndarray,
) -> np.ndarray:
    probabilities = _softmax(logits)
    ranking_probabilities = _softmax(ranking_logits)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranking_order = np.argsort(-ranking_probabilities, axis=1, kind="stable")
    rows = np.arange(len(logits))
    top1 = order[:, 0]
    detector_classes = np.asarray(prediction["class_ids"], dtype=np.int64)
    source_ids = np.asarray(prediction.get("source_ids", np.zeros(len(logits))), dtype=np.float32)
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    normalized_boxes = boxes / np.asarray(
        [record["width"], record["height"], record["width"], record["height"]],
        dtype=np.float32,
    )
    sizes = np.maximum(0.0, normalized_boxes[:, 2:] - normalized_boxes[:, :2])
    areas = sizes[:, 0] * sizes[:, 1]
    aspects = np.divide(
        sizes[:, 0],
        sizes[:, 1],
        out=np.zeros(len(sizes), dtype=np.float32),
        where=sizes[:, 1] > 0.0,
    )
    entropy = -np.sum(probabilities * np.log(probabilities.clip(1e-12)), axis=1)
    ranking_entropy = -np.sum(
        ranking_probabilities * np.log(ranking_probabilities.clip(1e-12)), axis=1
    )
    centered = logits - logits.mean(axis=1, keepdims=True)
    normalized_logits = centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-6)
    scalar = np.column_stack(
        (
            np.asarray(prediction["scores"], dtype=np.float32),
            source_ids,
            probabilities[rows, order[:, 0]],
            probabilities[rows, order[:, 1]],
            probabilities[rows, order[:, 0]] - probabilities[rows, order[:, 1]],
            entropy,
            ranking_probabilities[rows, ranking_order[:, 0]],
            ranking_probabilities[rows, ranking_order[:, 2]],
            ranking_entropy,
            top1 == ranking_order[:, 0],
            detector_classes == top1,
            np.any(ranking_order[:, :3] == detector_classes[:, None], axis=1),
            normalized_boxes,
            sizes,
            areas,
            aspects,
            np.full(len(logits), len(logits), dtype=np.float32),
        )
    ).astype(np.float32)
    parts = [scalar, normalized_logits.astype(np.float32), proposal_context_features(prediction)]
    ensemble_fields = ("support_counts", "source_masks", "member_scores")
    if any(field in prediction for field in ensemble_fields):
        if not all(field in prediction for field in ensemble_fields):
            raise ValueError("ensemble verifier fields must be supplied together")
        support = np.asarray(prediction["support_counts"], dtype=np.float32)
        source_masks = np.asarray(prediction["source_masks"], dtype=np.int64)
        member_scores = np.asarray(prediction["member_scores"], dtype=np.float32)
        if (
            len(support) != len(logits)
            or len(source_masks) != len(logits)
            or member_scores.ndim != 2
            or len(member_scores) != len(logits)
        ):
            raise ValueError("ensemble verifier fields are not aligned with proposals")
        model_count = member_scores.shape[1]
        source_bits = np.column_stack(
            [((source_masks >> index) & 1).astype(np.float32) for index in range(model_count)]
        )
        present = member_scores > 0.0
        present_count = np.maximum(present.sum(axis=1), 1)
        present_minimum = np.min(
            np.where(present, member_scores, np.inf),
            axis=1,
        )
        ensemble = np.column_stack(
            (
                support,
                support / max(float(model_count), 1.0),
                source_bits,
                member_scores,
                member_scores.sum(axis=1) / present_count,
                member_scores.std(axis=1),
                member_scores.max(axis=1),
                present_minimum,
            )
        ).astype(np.float32)
        parts.append(ensemble)
    return np.column_stack(parts)


def collect_logits(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    dataset_root: Path,
    classifier: OnnxClassifier,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final_parts = []
    ranking_parts = []
    counts = []
    for record, prediction in zip(records, predictions):
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                prediction["boxes_xyxy"],
                prediction["scores"],
                prediction["class_ids"],
            )
        ]
        with Image.open(dataset_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            try:
                result = classifier.classify(image, detections)
            finally:
                image.close()
        if isinstance(result, ClassificationResult):
            final = result.logits
            ranking = result.ranking_logits
        else:
            final = ranking = result
        final_parts.append(np.asarray(final, dtype=np.float32))
        ranking_parts.append(np.asarray(ranking, dtype=np.float32))
        counts.append(len(detections))
    return (
        np.concatenate(final_parts),
        np.concatenate(ranking_parts),
        np.asarray(counts, dtype=np.int64),
    )


def classifier_metadata_for_view(metadata: Any, view_mode: str) -> Any:
    if view_mode == "package":
        return metadata
    if view_mode == "box_resize":
        return metadata.model_copy(update={"neighbor_mask_inference": None})
    raise ValueError(f"unsupported classifier view mode: {view_mode}")


def _load_predictions(
    path: Path,
    records: list[dict[str, Any]],
    *,
    allow_superset: bool = False,
) -> list[dict[str, Any]]:
    by_id = {
        str(row["image_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    }
    expected = {str(record["image_id"]) for record in records}
    coverage_matches = expected.issubset(by_id) if allow_superset else set(by_id) == expected
    if not coverage_matches:
        raise ValueError("verifier prediction coverage differs from manifest")
    return [by_id[str(record["image_id"])] for record in records]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    predictions = _load_predictions(args.predictions, records)
    if args.logits_cache.is_file() and not args.refresh_cache:
        cache = np.load(args.logits_cache)
        final_logits = cache["logits"]
        ranking_logits = cache["ranking_logits"]
        counts = cache["counts"]
    else:
        package = load_model_package(args.package)
        classifier = OnnxClassifier(
            package.classifier_path,
            classifier_metadata_for_view(
                package.metadata.classifier,
                args.classifier_view_mode,
            ),
            args.provider,
            args.cuda_dll_dir,
        )
        final_logits, ranking_logits, counts = collect_logits(
            records,
            predictions,
            dataset_root=args.dataset_root,
            classifier=classifier,
        )
        args.logits_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.logits_cache,
            logits=final_logits,
            ranking_logits=ranking_logits,
            counts=counts,
        )
    if counts.tolist() != [len(row["scores"]) for row in predictions]:
        raise ValueError("classifier verifier cache does not match prediction counts")
    secondary_final_logits = None
    secondary_ranking_logits = None
    if args.secondary_logits_cache is not None:
        secondary_cache = np.load(args.secondary_logits_cache)
        secondary_final_logits = secondary_cache["logits"].astype(np.float32)
        secondary_ranking_logits = secondary_cache["ranking_logits"].astype(np.float32)
        secondary_counts = secondary_cache["counts"]
        if secondary_counts.tolist() != counts.tolist():
            raise ValueError("secondary classifier verifier cache does not match predictions")

    feature_parts = []
    label_parts = []
    assignment_parts = []
    fold_parts = []
    offset = 0
    for record, prediction, count in zip(records, predictions, counts):
        end = offset + int(count)
        view_features = [
            verifier_features(
                record, prediction, final_logits[offset:end], ranking_logits[offset:end]
            )
        ]
        if secondary_final_logits is not None and secondary_ranking_logits is not None:
            view_features.append(
                verifier_features(
                    record,
                    prediction,
                    secondary_final_logits[offset:end],
                    secondary_ranking_logits[offset:end],
                )
            )
        feature_parts.append(np.column_stack(view_features))
        label_parts.append(
            proposal_labels(record, np.asarray(prediction["boxes_xyxy"], dtype=np.float32))
        )
        assignment_parts.append(
            proposal_assignment_labels(
                record,
                np.asarray(prediction["boxes_xyxy"], dtype=np.float32),
            )
        )
        fold_parts.append(np.full(int(count), int(record["fold"]), dtype=np.int64))
        offset = end
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    assignment_labels = np.concatenate(assignment_parts)
    quality_parts = []
    for record, prediction in zip(records, predictions):
        quality_parts.append(
            proposal_qualities(
                record,
                np.asarray(prediction["boxes_xyxy"], dtype=np.float32),
            )
        )
    qualities = np.concatenate(quality_parts)
    candidate_folds = np.concatenate(fold_parts)

    verified_scores = np.zeros(len(labels), dtype=np.float64)
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training = candidate_folds != held_out_fold
        held_out = candidate_folds == held_out_fold
        if args.model_kind in {"extra_trees_classifier", "extra_trees_assignment"}:
            training_targets = (
                assignment_labels if args.model_kind == "extra_trees_assignment" else labels
            )
            model = ExtraTreesClassifier(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                class_weight="balanced",
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], training_targets[training])
            verified_scores[held_out] = model.predict_proba(features[held_out])[:, 1]
        else:
            model_type = (
                ExtraTreesRegressor
                if args.model_kind == "extra_trees_regressor"
                else RandomForestRegressor
            )
            model = model_type(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], qualities[training])
            verified_scores[held_out] = np.clip(model.predict(features[held_out]), 0.0, 1.0)
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_candidate_count": int(training.sum()),
                "training_positive_count": int(
                    (assignment_labels if args.model_kind == "extra_trees_assignment" else labels)[
                        training
                    ].sum()
                ),
                "training_mean_quality": float(qualities[training].mean()),
            }
        )

    ranked = []
    offset = 0
    for prediction, count in zip(predictions, counts):
        end = offset + int(count)
        ranked.append({**prediction, "scores": verified_scores[offset:end].tolist()})
        offset = end

    candidates = []
    for score_threshold, nms_threshold in product(args.score_thresholds, args.nms_thresholds):
        selected_predictions = select_ranked_predictions(
            ranked,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_threshold,
        )
        metrics = _metrics(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        candidates.append(
            {
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "metrics": metrics,
            }
        )
    zero_error = [
        row
        for row in candidates
        if row["metrics"]["false_positive_count"] == 0
        and row["metrics"]["false_negative_count"] == 0
    ]
    selected = max(
        zero_error or candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )
    selected_predictions = select_ranked_predictions(
        ranked,
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_classifier_verifier",
        "folds": sorted(folds),
        "input_candidate_count": len(labels),
        "input_positive_count": int(
            (assignment_labels if args.model_kind == "extra_trees_assignment" else labels).sum()
        ),
        "input_any_iou50_positive_count": int(labels.sum()),
        "input_assignment_positive_count": int(assignment_labels.sum()),
        "model": args.model_kind,
        "classifier_package": args.package.name,
        "classifier_view_mode": args.classifier_view_mode,
        "secondary_logits_cache": (
            args.secondary_logits_cache.name if args.secondary_logits_cache else None
        ),
        "fold_diagnostics": fold_diagnostics,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "selected": selected,
        "ranked_predictions_output": (
            args.ranked_predictions_output.name if args.ranked_predictions_output else None
        ),
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.ranked_predictions_output:
        args.ranked_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.ranked_predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in ranked),
            encoding="utf-8",
        )
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_predictions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit an image-content verifier for detector proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--logits-cache", type=Path, required=True)
    parser.add_argument("--secondary-logits-cache", type=Path)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--classifier-view-mode",
        choices=["package", "box_resize"],
        default="package",
    )
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--ranked-predictions-output", type=Path)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--model-kind",
        choices=[
            "extra_trees_classifier",
            "extra_trees_assignment",
            "extra_trees_regressor",
            "random_forest_regressor",
        ],
        default="extra_trees_classifier",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
