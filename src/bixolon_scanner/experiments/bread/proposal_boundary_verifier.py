from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions, verifier_features
from .proposal_count_selector import count_constrained_select
from .proposal_embedding_verifier import embedding_features
from .proposal_group_verifier import (
    group_content_relation_features,
    group_geometry_features,
)


def find_candidate_index(prediction: dict[str, Any], box: list[float], score: float) -> int:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    target = np.asarray(box, dtype=np.float32)
    distances = np.max(np.abs(boxes - target), axis=1)
    score_distances = np.abs(scores - float(score))
    order = np.lexsort((score_distances, distances))
    selected = int(order[0])
    if distances[selected] > 1e-3 or score_distances[selected] > 1e-5:
        raise ValueError("selected proposal cannot be aligned with source candidates")
    return selected


def boundary_context_features(
    scores: list[float], position: int, *, length: int = 12
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    padded = np.zeros(length, dtype=np.float32)
    padded[: min(length, len(values))] = values[:length]
    current = float(values[position])
    previous = float(values[position - 1]) if position else 1.0
    following = float(values[position + 1]) if position + 1 < len(values) else 0.0
    return np.concatenate(
        (
            np.asarray(
                [
                    position,
                    position / max(float(len(values)), 1.0),
                    len(values),
                    current,
                    previous - current,
                    current - following,
                ],
                dtype=np.float32,
            ),
            padded,
        )
    )


def _selection_candidates(
    records: list[dict[str, Any]],
    available: list[dict[str, Any]],
    probabilities: np.ndarray,
    slices: list[slice],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for threshold in np.concatenate(([-1.0], np.unique(probabilities))):
        predictions = []
        for prediction, image_slice in zip(available, slices):
            keep = probabilities[image_slice] > threshold
            predictions.append(
                {
                    "image_id": prediction["image_id"],
                    "boxes_xyxy": [
                        box for box, selected in zip(prediction["boxes_xyxy"], keep) if selected
                    ],
                    "scores": [
                        score for score, selected in zip(prediction["scores"], keep) if selected
                    ],
                    "class_ids": [
                        class_id
                        for class_id, selected in zip(prediction["class_ids"], keep)
                        if selected
                    ],
                }
            )
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        candidates.append(
            {
                "probability_threshold_exclusive": float(threshold),
                "metrics": metrics,
                "predictions": predictions,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
        ),
    )
    report_candidates = [
        {key: value for key, value in row.items() if key != "predictions"} for row in candidates
    ]
    report_selected = {key: value for key, value in selected.items() if key != "predictions"}
    return report_candidates, report_selected, selected["predictions"]


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
    ranked = _load_predictions(args.ranked_predictions, records)
    content_predictions = _load_predictions(args.content_predictions, records)
    logits_cache = np.load(args.logits_cache)
    logits = logits_cache["logits"].astype(np.float32)
    ranking_logits = logits_cache["ranking_logits"].astype(np.float32)
    counts = logits_cache["counts"].tolist()
    expected_counts = [len(row["scores"]) for row in content_predictions]
    if counts != expected_counts:
        raise ValueError("proposal logits cache does not match content predictions")

    use_embeddings = args.embedding_cache is not None
    if use_embeddings:
        embedding_cache = np.load(args.embedding_cache)
        raw_embeddings = embedding_cache["raw_embeddings"].astype(np.float32)
        adapted_embeddings = embedding_cache["adapted_embeddings"].astype(np.float32)
        if embedding_cache["counts"].tolist() != expected_counts:
            raise ValueError("proposal embedding cache does not match content predictions")

    available = [
        count_constrained_select(
            prediction,
            predicted_count=600,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            containment_threshold=args.containment_threshold,
            group_minimum=args.group_minimum,
        )
        for prediction in ranked
    ]
    feature_parts = []
    label_parts = []
    fold_parts = []
    slices = []
    offset = 0
    candidate_offset = 0
    for record, ranked_row, content_row, available_row, count in zip(
        records, ranked, content_predictions, available, counts
    ):
        candidate_end = candidate_offset + count
        classifier = verifier_features(
            record,
            content_row,
            logits[candidate_offset:candidate_end],
            ranking_logits[candidate_offset:candidate_end],
        )
        relations = group_content_relation_features(
            content_row, logits[candidate_offset:candidate_end]
        )
        if use_embeddings:
            classifier = embedding_features(
                classifier,
                raw_embeddings[candidate_offset:candidate_end],
                adapted_embeddings[candidate_offset:candidate_end],
            )
        content = np.column_stack((classifier, relations))
        geometry = group_geometry_features(record, ranked_row)
        rows = []
        for position, (box, score) in enumerate(
            zip(available_row["boxes_xyxy"], available_row["scores"])
        ):
            ranked_index = find_candidate_index(ranked_row, box, score)
            content_index = find_candidate_index(content_row, box, score)
            rows.append(
                np.concatenate(
                    (
                        boundary_context_features(available_row["scores"], position),
                        geometry[ranked_index],
                        content[content_index],
                    )
                )
            )
        image_features = np.stack(rows)
        feature_parts.append(image_features)
        target_count = len(record["annotations"])
        label_parts.append((np.arange(len(image_features)) < target_count).astype(np.int64))
        fold_parts.append(np.full(len(image_features), int(record["fold"]), dtype=np.int64))
        slices.append(slice(offset, offset + len(image_features)))
        offset += len(image_features)
        candidate_offset = candidate_end

    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    candidate_folds = np.concatenate(fold_parts)
    probabilities = np.zeros(len(labels), dtype=np.float32)
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training = candidate_folds != held_out_fold
        held_out = candidate_folds == held_out_fold
        model_type = (
            ExtraTreesClassifier if args.model_kind == "extra_trees" else RandomForestClassifier
        )
        model = model_type(
            n_estimators=args.estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + held_out_fold,
        )
        model.fit(features[training], labels[training])
        probabilities[held_out] = model.predict_proba(features[held_out])[:, 1]
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_candidate_count": int(training.sum()),
                "validation_candidate_count": int(held_out.sum()),
                "validation_positive_count": int(labels[held_out].sum()),
                "validation_negative_count": int((~labels[held_out].astype(bool)).sum()),
            }
        )

    candidates, selected, selected_predictions = _selection_candidates(
        records, available, probabilities, slices
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_boundary_verifier",
        "folds": sorted(folds),
        "model": args.model_kind,
        "feature_count": int(features.shape[1]),
        "content_embeddings": use_embeddings,
        "candidate_count": int(len(labels)),
        "positive_candidate_count": int(labels.sum()),
        "negative_candidate_count": int((~labels.astype(bool)).sum()),
        "fold_diagnostics": fold_diagnostics,
        "threshold_candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
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
        description="Cross-fit the true/false boundary of final detector proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--content-predictions", type=Path, required=True)
    parser.add_argument("--logits-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--containment-threshold", type=float, default=0.9)
    parser.add_argument("--group-minimum", type=int, default=3)
    parser.add_argument(
        "--model-kind",
        choices=["extra_trees", "random_forest"],
        default="extra_trees",
    )
    parser.add_argument("--estimators", type=int, default=1000)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
