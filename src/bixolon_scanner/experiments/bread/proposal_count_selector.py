from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import nms
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions, _softmax
from .proposal_group_verifier import group_content_relation_features
from .proposal_ranker import suppress_group_boxes


def _padded_sorted(values: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.float32)
    selected = np.sort(values.astype(np.float32))[::-1][:length]
    output[: len(selected)] = selected
    return output


def _nms_count_signals(prediction: dict[str, Any]) -> np.ndarray:
    candidates = [
        Detection(*box, float(score), int(class_id))
        for box, score, class_id in zip(
            prediction["boxes_xyxy"], prediction["scores"], prediction["class_ids"]
        )
    ]
    values = []
    for score_threshold in np.linspace(0.1, 0.8, 8):
        filtered = [item for item in candidates if item.score >= score_threshold]
        for nms_threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            values.append(len(nms(filtered, nms_threshold)))
    return np.asarray(values, dtype=np.float32)


def _group_count_signals(prediction: dict[str, Any]) -> np.ndarray:
    candidates = [
        Detection(*box, float(score), int(class_id))
        for box, score, class_id in zip(
            prediction["boxes_xyxy"], prediction["scores"], prediction["class_ids"]
        )
    ]
    values = []
    for score_threshold in (0.3, 0.4, 0.5):
        filtered = [item for item in candidates if item.score >= score_threshold]
        for containment_threshold in (0.8, 0.9, 0.95):
            for group_minimum in (2, 3):
                suppressed = suppress_group_boxes(
                    filtered,
                    containment_threshold=containment_threshold,
                    group_minimum=group_minimum,
                )
                for nms_threshold in (0.25, 0.3, 0.35, 0.4):
                    values.append(len(nms(suppressed, nms_threshold)))
    return np.asarray(values, dtype=np.float32)


def _spatial_rank_features(
    record: dict[str, Any], prediction: dict[str, Any], *, length: int
) -> np.ndarray:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    sources = np.asarray(prediction.get("source_ids", np.zeros(len(boxes))), dtype=np.float32)
    normalized = boxes / np.asarray(
        [record["width"], record["height"], record["width"], record["height"]],
        dtype=np.float32,
    )
    sizes = np.maximum(0.0, normalized[:, 2:] - normalized[:, :2])
    centers = (normalized[:, :2] + normalized[:, 2:]) * 0.5
    areas = sizes[:, 0] * sizes[:, 1]
    aspects = np.divide(
        sizes[:, 0],
        sizes[:, 1],
        out=np.zeros(len(boxes), dtype=np.float32),
        where=sizes[:, 1] > 0.0,
    )
    rows = np.column_stack((scores, sources, normalized, centers, sizes, areas, aspects)).astype(
        np.float32
    )
    output = np.zeros((length, rows.shape[1]), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:length]
    output[: len(order)] = rows[order]
    return output.ravel()


def count_features(
    record: dict[str, Any],
    ranked: dict[str, Any],
    raw: dict[str, Any],
    *,
    rank_length: int = 24,
    include_group_count_signals: bool = False,
) -> np.ndarray:
    scores = np.asarray(ranked["scores"], dtype=np.float32)
    sources = np.asarray(ranked.get("source_ids", np.zeros(len(scores))), dtype=np.int64)
    raw_scores = np.asarray(raw["scores"], dtype=np.float32)
    thresholds = np.linspace(0.1, 0.9, 17, dtype=np.float32)
    ranked_all = _padded_sorted(scores, rank_length)
    ranked_left = _padded_sorted(scores[sources == 0], rank_length)
    ranked_right = _padded_sorted(scores[sources == 1], rank_length)
    raw_sorted = _padded_sorted(raw_scores, rank_length)
    gaps = ranked_all[:-1] - ranked_all[1:]
    count_signals = np.concatenate(
        [
            np.asarray([(scores >= threshold).sum() for threshold in thresholds]),
            np.asarray([(scores[sources == 0] >= threshold).sum() for threshold in thresholds]),
            np.asarray([(scores[sources == 1] >= threshold).sum() for threshold in thresholds]),
            np.asarray([(raw_scores >= threshold).sum() for threshold in thresholds]),
        ]
    ).astype(np.float32)
    image = np.asarray(
        [
            record["width"] / 5000.0,
            record["height"] / 5000.0,
            record["width"] / max(float(record["height"]), 1.0),
            len(scores) / 100.0,
        ],
        dtype=np.float32,
    )
    parts = [
        ranked_all,
        ranked_left,
        ranked_right,
        raw_sorted,
        gaps,
        count_signals,
        _nms_count_signals(ranked),
        _nms_count_signals(raw),
    ]
    if include_group_count_signals:
        parts.append(_group_count_signals(ranked))
    parts.extend((_spatial_rank_features(record, ranked, length=rank_length), image))
    return np.concatenate(parts)


def count_constrained_select(
    prediction: dict[str, Any],
    *,
    predicted_count: int,
    score_threshold: float,
    nms_iou_threshold: float,
    containment_threshold: float = 0.8,
    group_minimum: int = 0,
) -> dict[str, Any]:
    candidates = [
        Detection(*box, float(score), int(class_id))
        for box, score, class_id in zip(
            prediction["boxes_xyxy"], prediction["scores"], prediction["class_ids"]
        )
        if score >= score_threshold
    ]
    if group_minimum:
        candidates = suppress_group_boxes(
            candidates,
            containment_threshold=containment_threshold,
            group_minimum=group_minimum,
        )
    selected = nms(candidates, nms_iou_threshold)[:predicted_count]
    return {
        "image_id": prediction["image_id"],
        "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in selected],
        "scores": [item.score for item in selected],
        "class_ids": [item.class_id for item in selected],
    }


def proposal_content_count_features(
    prediction: dict[str, Any],
    logits: np.ndarray,
    ranking_logits: np.ndarray,
    *,
    rank_length: int = 16,
) -> np.ndarray:
    if len(logits) != len(prediction["scores"]) or len(ranking_logits) != len(logits):
        raise ValueError("proposal content arrays are not aligned with predictions")
    probabilities = _softmax(logits)
    ranking_probabilities = _softmax(ranking_logits)
    relations = group_content_relation_features(prediction, logits)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    sources = np.asarray(prediction.get("source_ids", np.zeros(len(scores))), dtype=np.float32)

    def probability_summary(values: np.ndarray) -> np.ndarray:
        ordered = np.sort(values, axis=1)
        entropy = -np.sum(values * np.log(values.clip(1e-12)), axis=1)
        return np.column_stack(
            (
                ordered[:, -1],
                ordered[:, -1] - ordered[:, -2],
                entropy,
            )
        )

    rows = np.column_stack(
        (
            scores,
            sources,
            probability_summary(probabilities),
            probability_summary(ranking_probabilities),
            relations,
        )
    ).astype(np.float32)
    ordered_rows = np.zeros((rank_length, rows.shape[1]), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:rank_length]
    ordered_rows[: len(order)] = rows[order]
    summary = np.concatenate(
        (
            rows.mean(axis=0),
            rows.std(axis=0),
            rows.max(axis=0),
            np.quantile(rows, 0.5, axis=0),
            np.quantile(rows, 0.75, axis=0),
        )
    ).astype(np.float32)
    top1 = np.argmax(probabilities, axis=1)
    top1_confidence = probabilities[np.arange(len(probabilities)), top1]
    class_support = np.zeros((3, probabilities.shape[1]), dtype=np.float32)
    for class_id in range(probabilities.shape[1]):
        selected = top1 == class_id
        if selected.any():
            weighted = scores[selected] * top1_confidence[selected]
            class_support[:, class_id] = (
                weighted.max(),
                weighted.sum(),
                float((top1_confidence[selected] >= 0.5).sum()),
            )
    return np.concatenate((ordered_rows.ravel(), summary, class_support.ravel()))


def _selection_search(
    records: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    counts: np.ndarray,
    *,
    score_thresholds: list[float],
    nms_thresholds: list[float],
    containment_thresholds: list[float],
    group_minimums: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for score_threshold, nms_threshold, containment_threshold, group_minimum in product(
        score_thresholds,
        nms_thresholds,
        containment_thresholds,
        group_minimums,
    ):
        predictions = [
            count_constrained_select(
                prediction,
                predicted_count=int(count),
                score_threshold=score_threshold,
                nms_iou_threshold=nms_threshold,
                containment_threshold=containment_threshold,
                group_minimum=group_minimum,
            )
            for prediction, count in zip(ranked, counts)
        ]
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
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "containment_threshold": containment_threshold,
                "group_minimum": group_minimum,
                "metrics": metrics,
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
    selected_predictions = [
        count_constrained_select(
            prediction,
            predicted_count=int(count),
            score_threshold=selected["score_threshold"],
            nms_iou_threshold=selected["nms_iou_threshold"],
            containment_threshold=selected["containment_threshold"],
            group_minimum=selected["group_minimum"],
        )
        for prediction, count in zip(ranked, counts)
    ]
    return candidates, selected, selected_predictions


def _selective_diagnostic(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    signals: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    candidates = []
    for signal_name, values in signals.items():
        for threshold in np.concatenate(([-1.0], np.unique(values))):
            accepted = values > threshold
            if not accepted.any():
                continue
            accepted_records = [record for record, keep in zip(records, accepted) if keep]
            accepted_predictions = [
                prediction for prediction, keep in zip(predictions, accepted) if keep
            ]
            metrics = _metrics(
                accepted_records,
                accepted_predictions,
                score_threshold=0.0,
                nms_iou_threshold=1.0,
                match_iou_threshold=0.5,
                max_queries=600,
            )
            if metrics["false_positive_count"] == 0 and metrics["false_negative_count"] == 0:
                recaptured = ~accepted
                candidates.append(
                    {
                        "signal": signal_name,
                        "recapture_if_less_than_or_equal": float(threshold),
                        "accepted_image_count": int(accepted.sum()),
                        "recaptured_image_count": int(recaptured.sum()),
                        "recapture_rate": float(recaptured.mean()),
                        "recaptured_image_ids": [
                            record["image_id"]
                            for record, should_recapture in zip(records, recaptured)
                            if should_recapture
                        ],
                        "accepted_metrics": metrics,
                    }
                )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row["recaptured_image_count"],
            -row["accepted_metrics"]["exact_image_rate"],
        ),
    )


def disagreement_recapture_mask(
    available: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    minimum_selected_count: int,
    extra_candidate_count: int,
    next_score_threshold: float,
) -> np.ndarray:
    decisions = []
    for available_row, selected_row in zip(available, selected):
        selected_count = len(selected_row["scores"])
        extra_count = len(available_row["scores"]) - selected_count
        next_score = float(available_row["scores"][selected_count]) if extra_count > 0 else -1.0
        decisions.append(
            selected_count >= minimum_selected_count
            and extra_count == extra_candidate_count
            and next_score >= next_score_threshold
        )
    return np.asarray(decisions, dtype=bool)


def _disagreement_recapture_diagnostic(
    records: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
) -> dict[str, Any] | None:
    available = [
        count_constrained_select(
            prediction,
            predicted_count=600,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            containment_threshold=containment_threshold,
            group_minimum=group_minimum,
        )
        for prediction in ranked
    ]
    next_scores = sorted(
        {
            float(available_row["scores"][len(selected_row["scores"])])
            for available_row, selected_row in zip(available, selected)
            if len(available_row["scores"]) > len(selected_row["scores"])
        }
    )
    candidates = []
    for minimum_selected_count in range(1, 8):
        for extra_candidate_count in range(1, 5):
            for next_score_threshold in next_scores:
                recaptured = disagreement_recapture_mask(
                    available,
                    selected,
                    minimum_selected_count=minimum_selected_count,
                    extra_candidate_count=extra_candidate_count,
                    next_score_threshold=next_score_threshold,
                )
                if not recaptured.any():
                    continue
                accepted_records = [
                    record
                    for record, should_recapture in zip(records, recaptured)
                    if not should_recapture
                ]
                accepted_predictions = [
                    prediction
                    for prediction, should_recapture in zip(selected, recaptured)
                    if not should_recapture
                ]
                metrics = _metrics(
                    accepted_records,
                    accepted_predictions,
                    score_threshold=0.0,
                    nms_iou_threshold=1.0,
                    match_iou_threshold=0.5,
                    max_queries=600,
                )
                if metrics["false_positive_count"] == 0 and metrics["false_negative_count"] == 0:
                    candidates.append(
                        {
                            "minimum_selected_count": minimum_selected_count,
                            "extra_candidate_count": extra_candidate_count,
                            "next_score_threshold_inclusive": next_score_threshold,
                            "recaptured_image_count": int(recaptured.sum()),
                            "recapture_rate": float(recaptured.mean()),
                            "recaptured_image_ids": [
                                record["image_id"]
                                for record, should_recapture in zip(records, recaptured)
                                if should_recapture
                            ],
                            "accepted_metrics": metrics,
                        }
                    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row["recaptured_image_count"],
            -row["next_score_threshold_inclusive"],
        ),
    )


def _calibrate_ordinal_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_count: int,
    maximum_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if maximum_counts is None:
        maximum_counts = np.full(len(targets), targets.max(), dtype=np.int64)
    thresholds = []
    diagnostics = []
    for index in range(probabilities.shape[1]):
        boundary = minimum_count + index
        expected = targets > boundary
        candidates = []
        for threshold in np.concatenate(([-1.0], np.unique(probabilities[:, index]))):
            predicted = probabilities[:, index] > threshold
            effective = predicted & (maximum_counts > boundary)
            false_positive = int((effective & ~expected).sum())
            false_negative = int((~effective & expected).sum())
            candidates.append((false_positive + false_negative, false_negative, float(threshold)))
        error_count, false_negative, threshold = min(candidates)
        thresholds.append(threshold)
        diagnostics.append(
            {
                "boundary": boundary,
                "probability_threshold_exclusive": threshold,
                "error_count": error_count,
                "false_negative_count": false_negative,
                "false_positive_count": error_count - false_negative,
            }
        )
    predicted_counts = minimum_count + np.sum(
        (probabilities > np.asarray(thresholds)[None, :])
        & (
            maximum_counts[:, None]
            > np.arange(
                minimum_count,
                minimum_count + probabilities.shape[1],
            )[None, :]
        ),
        axis=1,
    )
    return predicted_counts.astype(np.int64), diagnostics


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
    ranked = _load_predictions(
        args.ranked_predictions,
        records,
        allow_superset=args.allow_prediction_superset,
    )
    raw = _load_predictions(
        args.raw_predictions,
        records,
        allow_superset=args.allow_prediction_superset,
    )
    candidate_features = np.stack(
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
    feature_parts = [candidate_features]
    image_embedding_dimension = 0
    image_views: dict[str, np.ndarray] = {}
    if args.image_embedding_cache is not None:
        image_cache = np.load(args.image_embedding_cache)
        image_raw = image_cache["raw_embeddings"].astype(np.float32)
        image_adapted = image_cache["adapted_embeddings"].astype(np.float32)
        image_counts = image_cache["counts"]
        if image_counts.tolist() != [1] * len(records):
            raise ValueError("full-image embedding cache does not match detector records")
        similarity = np.sum(image_raw * image_adapted, axis=1, keepdims=True)
        displacement = np.linalg.norm(image_raw - image_adapted, axis=1, keepdims=True)
        image_both = np.column_stack((image_raw, image_adapted, similarity, displacement))
        feature_parts.append(image_both)
        image_embedding_dimension = int(image_raw.shape[1])
        image_views = {
            "image_raw": np.column_stack((image_raw, similarity, displacement)),
            "image_adapted": np.column_stack((image_adapted, similarity, displacement)),
            "image_both": image_both,
        }
    proposal_feature_dimension = 0
    proposal_features = None
    proposal_content_requested = (
        args.proposal_content_predictions is not None or args.proposal_logits_cache is not None
    )
    if proposal_content_requested:
        if args.proposal_content_predictions is None or args.proposal_logits_cache is None:
            raise ValueError(
                "proposal content predictions and logits cache must be supplied together"
            )
        proposal_predictions = _load_predictions(args.proposal_content_predictions, records)
        logits_cache = np.load(args.proposal_logits_cache)
        logits = logits_cache["logits"].astype(np.float32)
        ranking_logits = logits_cache["ranking_logits"].astype(np.float32)
        counts = logits_cache["counts"].tolist()
        expected_counts = [len(row["scores"]) for row in proposal_predictions]
        if counts != expected_counts:
            raise ValueError("proposal logits cache does not match proposal predictions")
        rows = []
        offset = 0
        for prediction, count in zip(proposal_predictions, counts):
            end = offset + count
            rows.append(
                proposal_content_count_features(
                    prediction,
                    logits[offset:end],
                    ranking_logits[offset:end],
                )
            )
            offset = end
        proposal_features = np.stack(rows)
        proposal_feature_dimension = int(proposal_features.shape[1])
        feature_parts.append(proposal_features)
    features = np.column_stack(feature_parts)
    if args.feature_view == "candidate":
        features = candidate_features
    elif args.feature_view in image_views:
        features = image_views[args.feature_view]
    elif args.feature_view == "proposal":
        if proposal_features is None:
            raise ValueError("proposal feature view requires proposal content inputs")
        features = proposal_features
    elif args.feature_view == "candidate_proposal":
        if proposal_features is None:
            raise ValueError("candidate-proposal view requires proposal content inputs")
        features = np.column_stack((candidate_features, proposal_features))
    elif args.feature_view != "all":
        raise ValueError(f"unavailable feature view {args.feature_view!r}")
    targets = np.asarray([len(record["annotations"]) for record in records], dtype=np.int64)
    record_folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    predicted_counts = np.zeros(len(records), dtype=np.int64)
    predicted_confidences = np.zeros(len(records), dtype=np.float32)
    predicted_margins = np.zeros(len(records), dtype=np.float32)
    ordinal = args.model_kind in {
        "ordinal_extra_trees",
        "ordinal_random_forest",
    }
    ordinal_probabilities = np.zeros(
        (len(records), int(targets.max() - targets.min())), dtype=np.float32
    )
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training = record_folds != held_out_fold
        held_out = record_folds == held_out_fold
        if ordinal:
            model_type = (
                ExtraTreesClassifier
                if args.model_kind == "ordinal_extra_trees"
                else RandomForestClassifier
            )
            for index, boundary in enumerate(range(int(targets.min()), int(targets.max()))):
                binary_targets = targets > boundary
                model = model_type(
                    n_estimators=args.estimators,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=args.max_features,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=args.seed + held_out_fold * 10 + boundary,
                )
                model.fit(features[training], binary_targets[training])
                ordinal_probabilities[held_out, index] = model.predict_proba(features[held_out])[
                    :, 1
                ]
            predicted_counts[held_out] = targets.min() + np.sum(
                ordinal_probabilities[held_out] > 0.5, axis=1
            )
            fold_diagnostics.append(
                {
                    "fold": held_out_fold,
                    "training_image_count": int(training.sum()),
                    "validation_image_count": int(held_out.sum()),
                    "count_accuracy": float(
                        np.mean(predicted_counts[held_out] == targets[held_out])
                    ),
                }
            )
            continue
        elif args.model_kind == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                class_weight="balanced",
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], targets[training])
        elif args.model_kind == "random_forest":
            model = RandomForestClassifier(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], targets[training])
        elif args.model_kind == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                max_iter=args.estimators,
                learning_rate=args.learning_rate,
                min_samples_leaf=args.min_samples_leaf,
                l2_regularization=args.l2_regularization,
                random_state=args.seed + held_out_fold,
            )
            model.fit(
                features[training],
                targets[training],
                sample_weight=compute_sample_weight("balanced", targets[training]),
            )
        elif args.model_kind in {
            "svc_rbf",
            "svc_linear",
            "logistic_regression",
            "knn",
        }:
            steps: list[Any] = [StandardScaler()]
            if args.pca_components:
                steps.append(
                    PCA(
                        n_components=min(
                            args.pca_components,
                            int(training.sum()) - 1,
                            features.shape[1],
                        ),
                        whiten=True,
                        random_state=args.seed + held_out_fold,
                    )
                )
            if args.model_kind.startswith("svc"):
                steps.append(
                    SVC(
                        C=args.regularization_c,
                        kernel=args.model_kind.removeprefix("svc_"),
                        gamma=args.gamma,
                        class_weight="balanced",
                        probability=True,
                        random_state=args.seed + held_out_fold,
                    )
                )
            elif args.model_kind == "logistic_regression":
                steps.append(
                    LogisticRegression(
                        C=args.regularization_c,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=args.seed + held_out_fold,
                    )
                )
            else:
                steps.append(
                    KNeighborsClassifier(
                        n_neighbors=args.neighbors,
                        weights="distance",
                    )
                )
            model = make_pipeline(*steps)
            model.fit(features[training], targets[training])
        elif args.model_kind == "extra_trees_regressor":
            model = ExtraTreesRegressor(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], targets[training])
        elif args.model_kind == "random_forest_regressor":
            model = RandomForestRegressor(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], targets[training])
        else:
            model = HistGradientBoostingRegressor(
                max_iter=args.estimators,
                learning_rate=args.learning_rate,
                min_samples_leaf=args.min_samples_leaf,
                l2_regularization=args.l2_regularization,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], targets[training])
        fold_predictions = model.predict(features[held_out])
        if args.model_kind.endswith("regressor"):
            raw_predictions = fold_predictions.copy()
            fold_predictions = np.rint(fold_predictions)
            predicted_confidences[held_out] = 1.0 - np.minimum(
                np.abs(raw_predictions - fold_predictions), 1.0
            )
            predicted_margins[held_out] = predicted_confidences[held_out]
        else:
            probabilities = model.predict_proba(features[held_out])
            ordered = np.sort(probabilities, axis=1)
            predicted_confidences[held_out] = ordered[:, -1]
            predicted_margins[held_out] = ordered[:, -1] - ordered[:, -2]
        predicted_counts[held_out] = np.clip(fold_predictions, targets.min(), targets.max()).astype(
            np.int64
        )
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_image_count": int(training.sum()),
                "validation_image_count": int(held_out.sum()),
                "count_accuracy": float(np.mean(predicted_counts[held_out] == targets[held_out])),
            }
        )

    ordinal_diagnostics = None
    if ordinal:
        ordinal_maximum_counts = np.asarray(
            [
                len(
                    count_constrained_select(
                        prediction,
                        predicted_count=600,
                        score_threshold=args.score_thresholds[0],
                        nms_iou_threshold=args.nms_thresholds[0],
                        containment_threshold=args.containment_thresholds[0],
                        group_minimum=args.group_minimums[0],
                    )["scores"]
                )
                for prediction in ranked
            ],
            dtype=np.int64,
        )
        predicted_counts, ordinal_diagnostics = _calibrate_ordinal_thresholds(
            ordinal_probabilities,
            targets,
            minimum_count=int(targets.min()),
            maximum_counts=ordinal_maximum_counts,
        )
        for index, diagnostic in enumerate(ordinal_diagnostics):
            boundary = int(diagnostic["boundary"])
            expected = targets > boundary
            near_zero = []
            for threshold in np.concatenate(([-1.0], np.unique(ordinal_probabilities[:, index]))):
                effective = (ordinal_probabilities[:, index] > threshold) & (
                    ordinal_maximum_counts > boundary
                )
                false_positive = effective & ~expected
                false_negative = ~effective & expected
                error_count = int(false_positive.sum() + false_negative.sum())
                if error_count <= 5:
                    near_zero.append(
                        {
                            "probability_threshold_exclusive": float(threshold),
                            "false_positive_image_ids": [
                                record["image_id"]
                                for record, failed in zip(records, false_positive)
                                if failed
                            ],
                            "false_negative_image_ids": [
                                record["image_id"]
                                for record, failed in zip(records, false_negative)
                                if failed
                            ],
                        }
                    )
            diagnostic["near_zero_candidates"] = near_zero
        distances = np.min(
            np.abs(
                ordinal_probabilities
                - np.asarray(
                    [row["probability_threshold_exclusive"] for row in ordinal_diagnostics],
                    dtype=np.float32,
                )[None, :]
            ),
            axis=1,
        )
        predicted_confidences[:] = distances
        predicted_margins[:] = distances
        for diagnostic in fold_diagnostics:
            held_out = record_folds == diagnostic["fold"]
            diagnostic["count_accuracy"] = float(
                np.mean(predicted_counts[held_out] == targets[held_out])
            )

    candidates, selected, selected_predictions = _selection_search(
        records,
        ranked,
        predicted_counts,
        score_thresholds=args.score_thresholds,
        nms_thresholds=args.nms_thresholds,
        containment_thresholds=args.containment_thresholds,
        group_minimums=args.group_minimums,
    )
    oracle_candidates, oracle_selected, oracle_predictions = _selection_search(
        records,
        ranked,
        targets,
        score_thresholds=args.score_thresholds,
        nms_thresholds=args.nms_thresholds,
        containment_thresholds=args.containment_thresholds,
        group_minimums=args.group_minimums,
    )
    selective_diagnostic = _selective_diagnostic(
        records,
        selected_predictions,
        {
            "top1_probability": predicted_confidences,
            "top1_margin": predicted_margins,
        },
    )
    disagreement_diagnostic = _disagreement_recapture_diagnostic(
        records,
        ranked,
        selected_predictions,
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
        containment_threshold=selected["containment_threshold"],
        group_minimum=selected["group_minimum"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_count_constrained_proposal_selection",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "model": args.model_kind,
        "feature_view": args.feature_view,
        "feature_count": int(features.shape[1]),
        "full_image_embedding_dimension": image_embedding_dimension,
        "proposal_content_feature_dimension": proposal_feature_dimension,
        "group_count_signals": args.include_group_count_signals,
        "ordinal_boundary_diagnostics": ordinal_diagnostics,
        "fold_diagnostics": fold_diagnostics,
        "count_accuracy": float(np.mean(predicted_counts == targets)),
        "count_error_count": int(np.sum(predicted_counts != targets)),
        "count_errors": [
            {
                "image_id": record["image_id"],
                "fold": record["fold"],
                "difficulty": record["difficulty"],
                "target_count": int(target),
                "predicted_count": int(predicted),
                "prediction_confidence": float(confidence),
                "prediction_margin": float(margin),
            }
            for record, target, predicted, confidence, margin in zip(
                records,
                targets,
                predicted_counts,
                predicted_confidences,
                predicted_margins,
            )
            if target != predicted
        ],
        "candidate_count": len(candidates),
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
        "selective_count_diagnostic": selective_diagnostic,
        "disagreement_recapture_diagnostic": disagreement_diagnostic,
        "oracle_count_diagnostic": {
            "purpose": "diagnostic_upper_bound_not_a_deployable_policy",
            "zero_error_candidate_count": sum(
                row["metrics"]["false_positive_count"] == 0
                and row["metrics"]["false_negative_count"] == 0
                for row in oracle_candidates
            ),
            "selected": oracle_selected,
            "error_images": detection_error_rows(
                records,
                oracle_predictions,
                score_threshold=0.0,
                nms_iou_threshold=1.0,
                match_iou_threshold=0.5,
            ),
        },
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
    parser = argparse.ArgumentParser(description="Cross-fit an image count constrained selector")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument(
        "--allow-prediction-superset",
        action="store_true",
        help="Allow prediction JSONL to include images excluded by the manifest status filter",
    )
    parser.add_argument("--image-embedding-cache", type=Path)
    parser.add_argument("--proposal-content-predictions", type=Path)
    parser.add_argument("--proposal-logits-cache", type=Path)
    parser.add_argument("--include-group-count-signals", action="store_true")
    parser.add_argument(
        "--feature-view",
        choices=[
            "all",
            "candidate",
            "image_raw",
            "image_adapted",
            "image_both",
            "proposal",
            "candidate_proposal",
        ],
        default="all",
    )
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--containment-thresholds", type=float, nargs="+", default=[0.8])
    parser.add_argument("--group-minimums", type=int, nargs="+", default=[0])
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument(
        "--model-kind",
        choices=[
            "extra_trees",
            "random_forest",
            "hist_gradient_boosting",
            "extra_trees_regressor",
            "random_forest_regressor",
            "hist_gradient_boosting_regressor",
            "svc_rbf",
            "svc_linear",
            "logistic_regression",
            "knn",
            "ordinal_extra_trees",
            "ordinal_random_forest",
        ],
        default="extra_trees",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
