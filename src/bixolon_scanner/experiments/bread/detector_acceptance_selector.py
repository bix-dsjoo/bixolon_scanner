from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_count_selector import count_features


def recapture_at_threshold(
    probabilities: np.ndarray, baseline: np.ndarray, *, threshold: float
) -> np.ndarray:
    if probabilities.shape != baseline.shape:
        raise ValueError("probabilities and baseline mask must have the same shape")
    return baseline | (probabilities >= threshold)


def _read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {int(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate prediction image ids in {path}")
    return by_id


def _image_has_error(record: dict[str, Any], prediction: dict[str, Any]) -> bool:
    metrics = _metrics(
        [record],
        [prediction],
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=0.5,
        max_queries=600,
    )
    return bool(metrics["false_positive_count"] or metrics["false_negative_count"])


def _image_error_counts(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[int, int]:
    fp_images = 0
    fn_images = 0
    for record, prediction in zip(records, predictions):
        metrics = _metrics(
            [record],
            [prediction],
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        fp_images += metrics["false_positive_count"] > 0
        fn_images += metrics["false_negative_count"] > 0
    return int(fp_images), int(fn_images)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
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
    raw_by_id = _read_predictions(args.raw_predictions)
    ranked_by_id = _read_predictions(args.ranked_predictions)
    selected_by_id = _read_predictions(args.selected_predictions)
    expected_ids = {int(row["image_id"]) for row in records}
    if any(
        not expected_ids.issubset(values) for values in (raw_by_id, ranked_by_id, selected_by_id)
    ):
        raise ValueError("acceptance-selector predictions do not cover the selected scope")
    raw = [raw_by_id[int(record["image_id"])] for record in records]
    ranked = [ranked_by_id[int(record["image_id"])] for record in records]
    selected = [selected_by_id[int(record["image_id"])] for record in records]
    proposal_features = np.stack(
        [
            count_features(
                record,
                ranked_row,
                raw_row,
                include_group_count_signals=args.include_group_count_signals,
            )
            for record, ranked_row, raw_row in zip(records, ranked, raw)
        ]
    )
    image_embedding_dimension = 0
    if args.image_embedding_cache is not None:
        cache = np.load(args.image_embedding_cache)
        image_ids = cache.get("image_ids")
        expected_order = np.asarray([int(record["image_id"]) for record in records])
        if image_ids is None or not np.array_equal(image_ids, expected_order):
            raise ValueError("image embedding cache ids do not match acceptance-selector records")
        raw_embeddings = cache["raw_embeddings"].astype(np.float32)
        adapted_embeddings = cache["adapted_embeddings"].astype(np.float32)
        similarity = np.sum(raw_embeddings * adapted_embeddings, axis=1, keepdims=True)
        displacement = np.linalg.norm(raw_embeddings - adapted_embeddings, axis=1, keepdims=True)
        features = np.column_stack(
            (proposal_features, raw_embeddings, adapted_embeddings, similarity, displacement)
        )
        image_embedding_dimension = int(raw_embeddings.shape[1])
    else:
        features = proposal_features
    labels = np.asarray(
        [_image_has_error(record, prediction) for record, prediction in zip(records, selected)],
        dtype=np.int64,
    )
    record_folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    probabilities = np.zeros(len(records), dtype=np.float64)
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training = record_folds != held_out_fold
        held_out = record_folds == held_out_fold
        model_type = (
            ExtraTreesClassifier if args.model_kind == "extra_trees" else RandomForestClassifier
        )
        model = model_type(
            n_estimators=args.estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=("balanced" if args.model_kind == "extra_trees" else "balanced_subsample"),
            n_jobs=-1,
            random_state=args.seed + held_out_fold,
        )
        model.fit(features[training], labels[training])
        class_index = int(np.flatnonzero(model.classes_ == 1)[0])
        probabilities[held_out] = model.predict_proba(features[held_out])[:, class_index]
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_image_count": int(training.sum()),
                "training_error_image_count": int(labels[training].sum()),
                "validation_image_count": int(held_out.sum()),
                "validation_error_image_count": int(labels[held_out].sum()),
            }
        )
    baseline_report = json.loads(args.baseline_recapture_report.read_text(encoding="utf-8"))
    baseline_ids = {
        int(image_id) for image_id in baseline_report["selected"]["recaptured_image_ids"]
    }
    if not baseline_ids <= expected_ids:
        raise ValueError("baseline recapture report contains images outside the selected scope")
    baseline = np.asarray(
        [int(record["image_id"]) in baseline_ids for record in records], dtype=bool
    )
    thresholds = np.concatenate(([1.0000001], np.unique(probabilities)[::-1], [-1.0]))
    candidates = []
    masks = {}
    for threshold in thresholds:
        recaptured = recapture_at_threshold(probabilities, baseline, threshold=float(threshold))
        segmentation_rate = float((~recaptured).mean())
        if segmentation_rate < args.minimum_segmentation_rate:
            continue
        accepted_records = [row for row, flag in zip(records, recaptured) if not flag]
        accepted_predictions = [row for row, flag in zip(selected, recaptured) if not flag]
        metrics = _metrics(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        fp_images, fn_images = _image_error_counts(accepted_records, accepted_predictions)
        masks[float(threshold)] = recaptured
        candidates.append(
            {
                "error_probability_threshold_inclusive": float(threshold),
                "segmentation_image_count": int((~recaptured).sum()),
                "image_recapture_count": int(recaptured.sum()),
                "segmentation_rate": segmentation_rate,
                "false_positive_image_count": fp_images,
                "false_negative_image_count": fn_images,
                "metrics": metrics,
            }
        )
    zero_error = [
        row
        for row in candidates
        if row["false_positive_image_count"] == 0 and row["false_negative_image_count"] == 0
    ]
    selected_candidate = max(
        zero_error or candidates,
        key=lambda row: (
            -(row["false_positive_image_count"] + row["false_negative_image_count"]),
            row["segmentation_rate"],
            row["error_probability_threshold_inclusive"],
        ),
    )
    recaptured = masks[selected_candidate["error_probability_threshold_inclusive"]]
    accepted_records = [row for row, flag in zip(records, recaptured) if not flag]
    accepted_predictions = [row for row, flag in zip(selected, recaptured) if not flag]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_grouped_oof_acceptance_selector",
        "selection_scope": "proposal-only features without labels or image identifiers at inference",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "model": args.model_kind,
        "feature_count": int(features.shape[1]),
        "image_embedding_dimension": image_embedding_dimension,
        "group_count_signals": args.include_group_count_signals,
        "image_count": len(records),
        "error_image_count": int(labels.sum()),
        "fold_diagnostics": fold_diagnostics,
        "baseline_recapture_count": int(baseline.sum()),
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "selected": {
            **selected_candidate,
            "recaptured_image_ids": [
                int(row["image_id"]) for row, flag in zip(records, recaptured) if flag
            ],
        },
        "remaining_error_images": detection_error_rows(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
        "probabilities": [
            {
                "image_id": int(record["image_id"]),
                "fold": int(record["fold"]),
                "difficulty": record["difficulty"],
                "error_target": bool(label),
                "error_probability": float(probability),
            }
            for record, label, probability in zip(records, labels, probabilities)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit a proposal-only detector acceptance/IMAGE_RECAPTURE selector"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--selected-predictions", type=Path, required=True)
    parser.add_argument("--baseline-recapture-report", type=Path, required=True)
    parser.add_argument("--image-embedding-cache", type=Path)
    parser.add_argument("--include-group-count-signals", action="store_true")
    parser.add_argument(
        "--model-kind", choices=("extra_trees", "random_forest"), default="extra_trees"
    )
    parser.add_argument("--estimators", type=int, default=1500)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--minimum-segmentation-rate", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
