from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ...evaluation.recapture_separability import detection_geometry_features
from ...training.data import read_manifest


@dataclass(frozen=True)
class Candidate:
    name: str
    view: str
    kind: str
    parameter: float
    minimum_leaf: int = 1


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _sharpness(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def pixel_quality_features(image: Image.Image, *, maximum_side: int = 256) -> dict[str, float]:
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    rgb_image.thumbnail((maximum_side, maximum_side), Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype(np.float32)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-6,
    )
    horizontal = np.abs(np.diff(gray, axis=1)).ravel()
    vertical = np.abs(np.diff(gray, axis=0)).ravel()
    gradients = np.concatenate((horizontal, vertical))
    height, width = gray.shape
    y_margin = max(1, height // 5)
    x_margin = max(1, width // 5)
    center = gray[y_margin : height - y_margin, x_margin : width - x_margin]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[y_margin : height - y_margin, x_margin : width - x_margin] = False
    border = gray[border_mask]
    histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64) / max(int(histogram.sum()), 1)
    probabilities = probabilities[probabilities > 0.0]
    entropy = -np.sum(probabilities * np.log2(probabilities)) / math.log2(32)
    features: dict[str, float] = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_entropy": float(entropy),
        "dark_fraction_005": float((gray <= 0.05).mean()),
        "dark_fraction_015": float((gray <= 0.15).mean()),
        "dark_fraction_025": float((gray <= 0.25).mean()),
        "bright_fraction_075": float((gray >= 0.75).mean()),
        "bright_fraction_090": float((gray >= 0.90).mean()),
        "bright_fraction_097": float((gray >= 0.97).mean()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "gradient_mean": float(gradients.mean()),
        "gradient_std": float(gradients.std()),
        "gradient_q90": float(np.quantile(gradients, 0.90)),
        "gradient_q99": float(np.quantile(gradients, 0.99)),
        "sharpness": _sharpness(gray),
        "center_gray_mean": float(center.mean()),
        "border_gray_mean": float(border.mean()),
        "center_border_gray_difference": float(center.mean() - border.mean()),
        "aspect_ratio": width / max(float(height), 1.0),
    }
    for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        features[f"gray_q{round(quantile * 100):02d}"] = float(np.quantile(gray, quantile))
    for channel, name in enumerate(("red", "green", "blue")):
        features[f"{name}_mean"] = float(rgb[:, :, channel].mean())
        features[f"{name}_std"] = float(rgb[:, :, channel].std())
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("pixel quality features contain a non-finite value")
    return features


def detector_quality_features(
    record: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, float]:
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    ordered = np.sort(scores)[::-1]
    padded = np.zeros(32, dtype=np.float32)
    padded[: min(len(ordered), len(padded))] = ordered[: len(padded)]
    result = detection_geometry_features(
        prediction,
        width=int(record["width"]),
        height=int(record["height"]),
        nms_iou_threshold=0.5,
        maximum_aspect_ratio=8.0,
    )
    for index, score in enumerate(padded):
        result[f"ranked_score_{index:02d}"] = float(score)
    for threshold in np.linspace(0.01, 0.91, 19):
        result[f"raw_count_{threshold:.2f}"] = float((scores >= threshold).sum())
    result.update(
        {
            "raw_score_mean": float(scores.mean()) if len(scores) else 0.0,
            "raw_score_std": float(scores.std()) if len(scores) else 0.0,
            "raw_score_max": float(scores.max()) if len(scores) else 0.0,
            "raw_score_q95": float(np.quantile(scores, 0.95)) if len(scores) else 0.0,
            "raw_score_q99": float(np.quantile(scores, 0.99)) if len(scores) else 0.0,
        }
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("detector quality features contain a non-finite value")
    return result


def full_recall_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    positive_scores = scores[targets.astype(bool)]
    if not len(positive_scores):
        raise ValueError("at least one recapture target is required")
    return float(positive_scores.min())


def _flag_metrics(
    flags: np.ndarray,
    targets: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    true_positive = int(np.count_nonzero(flags & targets))
    false_positive = int(np.count_nonzero(flags & ~targets))
    positive_count = int(targets.sum())
    normal_count = int((~targets).sum())
    expected_by_reason = Counter(
        reason
        for target, record in zip(targets, records)
        if target
        for reason in record.get("expected_reason_codes", [])
    )
    caught_by_reason = Counter(
        reason
        for flag, target, record in zip(flags, targets, records)
        if flag and target
        for reason in record.get("expected_reason_codes", [])
    )
    return {
        "recapture_sample_count": positive_count,
        "normal_sample_count": normal_count,
        "true_recapture_count": true_positive,
        "recapture_recall": true_positive / positive_count if positive_count else None,
        "false_recapture_count": false_positive,
        "false_recapture_rate": false_positive / normal_count if normal_count else None,
        "total_image_recapture_count": int(flags.sum()),
        "total_image_recapture_rate": float(flags.mean()),
        "by_reason": {
            reason: {
                "caught": caught_by_reason[reason],
                "total": total,
                "recall": caught_by_reason[reason] / total,
            }
            for reason, total in sorted(expected_by_reason.items())
        },
    }


def _reason_masks(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    reasons = sorted(
        {reason for record in records for reason in record.get("expected_reason_codes", [])}
    )
    return {
        reason: np.asarray(
            [reason in record.get("expected_reason_codes", []) for record in records],
            dtype=bool,
        )
        for reason in reasons
    }


def assign_quality_policy_folds(records: list[dict[str, Any]], *, fold_count: int) -> np.ndarray:
    if fold_count < 2:
        raise ValueError("quality policy cross-validation requires at least two folds")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record["perceptual_group_id"]), []).append(index)
    strata: dict[str, list[tuple[str, list[int]]]] = {}
    for group_id, indices in grouped.items():
        labels = {
            reason
            for index in indices
            for reason in records[index].get("expected_reason_codes", [])
        }
        statuses = {records[index]["expected_image_status"] for index in indices}
        if len(statuses) != 1:
            raise ValueError("a quality policy group contains mixed image statuses")
        stratum = "+".join(sorted(labels)) if labels else "ANNOTATED"
        strata.setdefault(stratum, []).append((group_id, indices))
    assignments = np.full(len(records), -1, dtype=np.int64)
    total_counts = [0] * fold_count
    stratum_counts: dict[str, list[int]] = {}
    for stratum in sorted(strata, key=lambda name: (name == "ANNOTATED", name)):
        counts = [0] * fold_count
        stratum_counts[stratum] = counts
        groups = sorted(strata[stratum], key=lambda item: (-len(item[1]), item[0]))
        for _, indices in groups:
            fold = min(
                range(fold_count),
                key=lambda candidate: (
                    counts[candidate],
                    total_counts[candidate],
                    candidate,
                ),
            )
            for index in indices:
                assignments[index] = fold
            counts[fold] += len(indices)
            total_counts[fold] += len(indices)
    if np.any(assignments < 0):
        raise AssertionError("quality policy fold assignment is incomplete")
    return assignments


def _select_reason_conjunction(
    values: np.ndarray,
    feature_names: list[str],
    reason_targets: np.ndarray,
    normal_targets: np.ndarray,
    selection: np.ndarray,
    *,
    maximum_rule_count: int = 2,
) -> tuple[list[dict[str, Any]], np.ndarray] | None:
    selected_positive = selection & reason_targets
    selected_normal = selection & normal_targets
    if not selected_positive.any() or not selected_normal.any():
        return None
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    for column, name in enumerate(feature_names):
        below_threshold = float(values[selected_positive, column].max())
        above_threshold = float(values[selected_positive, column].min())
        candidates.extend(
            (
                (
                    {
                        "feature": name,
                        "direction": "at_or_below",
                        "threshold": below_threshold,
                    },
                    values[:, column] <= below_threshold,
                ),
                (
                    {
                        "feature": name,
                        "direction": "at_or_above",
                        "threshold": above_threshold,
                    },
                    values[:, column] >= above_threshold,
                ),
            )
        )
    candidates.sort(
        key=lambda item: (
            int(np.count_nonzero(item[1] & selected_normal)),
            item[0]["feature"],
            item[0]["direction"],
        )
    )
    best_rules = [candidates[0][0]]
    best_flags = candidates[0][1]
    if maximum_rule_count >= 2:
        best_pair: tuple[list[dict[str, Any]], np.ndarray] | None = None
        best_false_count: int | None = None
        # Rules that already admit almost every normal cannot improve the best pair.
        useful = candidates[: min(len(candidates), 256)]
        for left_index, (left_rule, left_flags) in enumerate(useful):
            for right_rule, right_flags in useful[left_index + 1 :]:
                combined = left_flags & right_flags
                false_count = int(np.count_nonzero(combined & selected_normal))
                if best_false_count is None or false_count < best_false_count:
                    best_pair = ([left_rule, right_rule], combined)
                    best_false_count = false_count
                elif false_count == best_false_count and best_pair is not None:
                    names = (left_rule["feature"], right_rule["feature"])
                    best_names = tuple(rule["feature"] for rule in best_pair[0])
                    if names < best_names:
                        best_pair = ([left_rule, right_rule], combined)
        if best_pair is not None and int(np.count_nonzero(best_pair[1] & selected_normal)) <= int(
            np.count_nonzero(best_flags & selected_normal)
        ):
            best_rules, best_flags = best_pair
    if not np.all(best_flags[selected_positive]):
        raise AssertionError("selected conjunction does not retain every reason target")
    return best_rules, best_flags


def reason_conjunction_policy(
    scalar: np.ndarray,
    scalar_names: list[str],
    records: list[dict[str, Any]],
    folds: np.ndarray,
) -> dict[str, Any]:
    targets = np.asarray(
        [record["expected_image_status"] == "RECAPTURE" for record in records],
        dtype=bool,
    )
    normal = ~targets
    reason_masks = _reason_masks(records)
    pooled_flags = np.zeros(len(records), dtype=bool)
    pooled_specialists = {}
    all_rows = np.ones(len(records), dtype=bool)
    for reason, reason_targets in reason_masks.items():
        selected = _select_reason_conjunction(
            scalar,
            scalar_names,
            reason_targets,
            normal,
            all_rows,
        )
        if selected is None:
            continue
        rules, flags = selected
        pooled_flags |= flags
        pooled_specialists[reason] = {
            "rules": rules,
            "reason_sample_count": int(reason_targets.sum()),
            "false_recapture_count": int(np.count_nonzero(flags & normal)),
            "false_recapture_image_ids": [
                int(record["image_id"])
                for record, flag, is_normal in zip(records, flags, normal)
                if flag and is_normal
            ],
        }
    nested_flags = np.zeros(len(records), dtype=bool)
    nested_diagnostics = []
    for outer_fold in sorted(set(folds.tolist())):
        training = folds != outer_fold
        validation = folds == outer_fold
        fold_flags = np.zeros(len(records), dtype=bool)
        specialists = {}
        for reason, reason_targets in reason_masks.items():
            selected = _select_reason_conjunction(
                scalar,
                scalar_names,
                reason_targets,
                normal,
                training,
            )
            if selected is None:
                specialists[reason] = {"available": False, "rules": []}
                continue
            rules, flags = selected
            fold_flags |= flags
            specialists[reason] = {"available": True, "rules": rules}
        nested_flags[validation] = fold_flags[validation]
        fold_records = [record for record, selected in zip(records, validation) if selected]
        nested_diagnostics.append(
            {
                "fold": int(outer_fold),
                "specialists": specialists,
                **_flag_metrics(fold_flags[validation], targets[validation], fold_records),
            }
        )

    def details(flags: np.ndarray) -> dict[str, Any]:
        return {
            **_flag_metrics(flags, targets, records),
            "missed_recapture_image_ids": [
                int(record["image_id"])
                for record, flag, target in zip(records, flags, targets)
                if target and not flag
            ],
            "false_recapture_image_ids": [
                int(record["image_id"])
                for record, flag, target in zip(records, flags, targets)
                if flag and not target
            ],
        }

    return {
        "selection": "reason-specific conjunctions of at most two scalar conditions",
        "pooled_oof": {
            "specialists": pooled_specialists,
            **details(pooled_flags),
        },
        "nested_outer_fold": {
            "folds": nested_diagnostics,
            **details(nested_flags),
        },
    }


def _records(args: argparse.Namespace) -> list[dict[str, Any]]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("difficulty") == "SCAN_LOG"
        and row.get("expected_image_status") in {"ANNOTATED", "RECAPTURE"}
    ]
    if not records:
        raise ValueError("no scan-log records matched the requested folds")
    return records


def _prediction_map(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _feature_views(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
    embedding_cache: Path,
    predictions: dict[int, dict[str, Any]],
    auxiliary_predictions: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    cache = np.load(embedding_cache)
    expected_ids = np.asarray([int(record["image_id"]) for record in records])
    if "image_ids" not in cache:
        raise ValueError("embedding cache does not contain image_ids")
    if not np.array_equal(cache["image_ids"], expected_ids):
        raise ValueError("embedding cache image order does not match the manifest")
    if not np.array_equal(cache["folds"], np.asarray([int(record["fold"]) for record in records])):
        raise ValueError("embedding cache fold assignments do not match the manifest")
    scalar_rows: list[dict[str, float]] = []
    root = dataset_root.resolve()
    for record in records:
        image_id = int(record["image_id"])
        if image_id not in predictions:
            raise ValueError(f"missing detector prediction for image {image_id}")
        image_path = (root / str(record["image_path"])).resolve()
        image_path.relative_to(root)
        with Image.open(image_path) as image:
            features = pixel_quality_features(image)
        features.update(
            {
                f"production_{name}": value
                for name, value in detector_quality_features(record, predictions[image_id]).items()
            }
        )
        if auxiliary_predictions is not None:
            if image_id not in auxiliary_predictions:
                raise ValueError(f"missing auxiliary detector prediction for image {image_id}")
            features.update(
                {
                    f"outer_fold_dfine_{name}": value
                    for name, value in detector_quality_features(
                        record, auxiliary_predictions[image_id]
                    ).items()
                }
            )
        scalar_rows.append(features)
    scalar_names = sorted(scalar_rows[0])
    if any(sorted(row) != scalar_names for row in scalar_rows):
        raise ValueError("scalar feature schema is inconsistent")
    scalar = np.asarray(
        [[row[name] for name in scalar_names] for row in scalar_rows], dtype=np.float32
    )
    raw = _normalized_rows(cache["raw_embeddings"])
    adapted = _normalized_rows(cache["adapted_embeddings"])
    embedding_summary = np.column_stack(
        (
            np.sum(raw * adapted, axis=1),
            np.linalg.norm(raw - adapted, axis=1),
            raw.mean(axis=1),
            raw.std(axis=1),
            adapted.mean(axis=1),
            adapted.std(axis=1),
        )
    ).astype(np.float32)
    scalar = np.column_stack((scalar, embedding_summary)).astype(np.float32)
    views = {
        "scalar": scalar,
        "raw": raw,
        "adapted": adapted,
        "raw_scalar": np.column_stack((raw, scalar)).astype(np.float32),
        "adapted_scalar": np.column_stack((adapted, scalar)).astype(np.float32),
        "dual_scalar": np.column_stack((raw, adapted, scalar)).astype(np.float32),
    }
    if any(not np.isfinite(values).all() for values in views.values()):
        raise ValueError("model feature views contain a non-finite value")
    return views, scalar_names + [
        "raw_adapted_cosine",
        "raw_adapted_distance",
        "raw_embedding_mean",
        "raw_embedding_std",
        "adapted_embedding_mean",
        "adapted_embedding_std",
    ]


def _candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for view in ("scalar", "raw_scalar", "adapted_scalar", "dual_scalar"):
        for maximum_features in (0.2, 0.5, 0.8):
            for minimum_leaf in (1, 2):
                candidates.extend(
                    (
                        Candidate(
                            f"extra_trees:{view}:mf{maximum_features}:leaf{minimum_leaf}",
                            view,
                            "extra_trees",
                            maximum_features,
                            minimum_leaf,
                        ),
                        Candidate(
                            f"random_forest:{view}:mf{maximum_features}:leaf{minimum_leaf}",
                            view,
                            "random_forest",
                            maximum_features,
                            minimum_leaf,
                        ),
                    )
                )
    for view in ("scalar", "raw", "adapted", "raw_scalar", "adapted_scalar"):
        for regularization in (0.01, 0.1, 1.0, 10.0):
            candidates.extend(
                (
                    Candidate(
                        f"logistic:{view}:c{regularization}",
                        view,
                        "logistic",
                        regularization,
                    ),
                    Candidate(
                        f"linear_svc:{view}:c{regularization}",
                        view,
                        "linear_svc",
                        regularization,
                    ),
                )
            )
    for view in ("raw", "adapted"):
        for regularization in (0.1, 1.0, 10.0):
            candidates.append(
                Candidate(
                    f"rbf_svc:{view}:c{regularization}",
                    view,
                    "rbf_svc",
                    regularization,
                )
            )
    return candidates


def _model(candidate: Candidate, *, seed: int) -> Any:
    if candidate.kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_features=candidate.parameter,
            min_samples_leaf=candidate.minimum_leaf,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if candidate.kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=700,
            max_features=candidate.parameter,
            min_samples_leaf=candidate.minimum_leaf,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if candidate.kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=candidate.parameter,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if candidate.kind == "linear_svc":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=candidate.parameter,
                kernel="linear",
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if candidate.kind == "rbf_svc":
        return SVC(
            C=candidate.parameter,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"unsupported model kind: {candidate.kind}")


def _scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        positive_index = list(model.classes_).index(True)
        return np.asarray(probabilities[:, positive_index], dtype=np.float64)
    return np.asarray(model.decision_function(features), dtype=np.float64)


def _oof_scores(
    candidate: Candidate,
    views: dict[str, np.ndarray],
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.full(len(targets), np.nan, dtype=np.float64)
    diagnostics = []
    for held_out_fold in sorted(set(folds.tolist())):
        training = folds != held_out_fold
        held_out = folds == held_out_fold
        model = _model(candidate, seed=seed + held_out_fold)
        model.fit(views[candidate.view][training], targets[training])
        result[held_out] = _scores(model, views[candidate.view][held_out])
        diagnostics.append(
            {
                "fold": int(held_out_fold),
                "training_count": int(training.sum()),
                "validation_count": int(held_out.sum()),
                "validation_recapture_count": int(targets[held_out].sum()),
            }
        )
    if not np.isfinite(result).all():
        raise ValueError("OOF inference did not cover every scan-log image")
    return result, diagnostics


def _rank_candidate(
    candidate: Candidate,
    scores: np.ndarray,
    targets: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    threshold = full_recall_threshold(scores, targets)
    flags = scores >= threshold
    metrics = _flag_metrics(flags, targets, records)
    return {
        "name": candidate.name,
        "view": candidate.view,
        "kind": candidate.kind,
        "parameter": candidate.parameter,
        "minimum_leaf": candidate.minimum_leaf,
        "pooled_full_recall_threshold": threshold,
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        **metrics,
    }


def _nested_threshold_diagnostic(
    candidate: Candidate,
    views: dict[str, np.ndarray],
    targets: np.ndarray,
    folds: np.ndarray,
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    flags = np.zeros(len(targets), dtype=bool)
    scores = np.full(len(targets), np.nan, dtype=np.float64)
    diagnostics = []
    unique_folds = sorted(set(folds.tolist()))
    for outer_fold in unique_folds:
        outer_training = folds != outer_fold
        outer_validation = folds == outer_fold
        inner_scores = np.full(int(outer_training.sum()), np.nan, dtype=np.float64)
        inner_targets = targets[outer_training]
        inner_folds = folds[outer_training]
        inner_features = views[candidate.view][outer_training]
        for inner_fold in sorted(set(inner_folds.tolist())):
            inner_training = inner_folds != inner_fold
            inner_validation = inner_folds == inner_fold
            inner_model = _model(candidate, seed=seed + outer_fold * 10 + inner_fold)
            inner_model.fit(inner_features[inner_training], inner_targets[inner_training])
            inner_scores[inner_validation] = _scores(inner_model, inner_features[inner_validation])
        threshold = full_recall_threshold(inner_scores, inner_targets)
        outer_model = _model(candidate, seed=seed + outer_fold)
        outer_model.fit(views[candidate.view][outer_training], targets[outer_training])
        outer_scores = _scores(outer_model, views[candidate.view][outer_validation])
        scores[outer_validation] = outer_scores
        flags[outer_validation] = outer_scores >= threshold
        diagnostics.append(
            {
                "fold": int(outer_fold),
                "threshold_selected_without_outer_fold": threshold,
                **_flag_metrics(
                    flags[outer_validation],
                    targets[outer_validation],
                    [record for record, selected in zip(records, outer_validation) if selected],
                ),
            }
        )
    return {
        "selection": "inner-OOF full-recall threshold per outer fold",
        "folds": diagnostics,
        **_flag_metrics(flags, targets, records),
        "missed_recapture_image_ids": [
            int(record["image_id"])
            for record, flag, target in zip(records, flags, targets)
            if target and not flag
        ],
        "false_recapture_image_ids": [
            int(record["image_id"])
            for record, flag, target in zip(records, flags, targets)
            if flag and not target
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = _records(args)
    prediction_map = _prediction_map(args.predictions)
    auxiliary_prediction_map = (
        _prediction_map(args.auxiliary_predictions) if args.auxiliary_predictions else None
    )
    views, scalar_names = _feature_views(
        records,
        dataset_root=args.dataset_root,
        embedding_cache=args.embedding_cache,
        predictions=prediction_map,
        auxiliary_predictions=auxiliary_prediction_map,
    )
    targets = np.asarray(
        [record["expected_image_status"] == "RECAPTURE" for record in records],
        dtype=bool,
    )
    detector_folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    folds = assign_quality_policy_folds(records, fold_count=len(set(args.folds)))
    candidates = _candidates()
    score_cache: dict[str, np.ndarray] = {}
    rankings = []
    fold_coverage: list[dict[str, Any]] | None = None
    for index, candidate in enumerate(candidates):
        scores, diagnostics = _oof_scores(
            candidate,
            views,
            targets,
            folds,
            seed=args.seed + index * 100,
        )
        score_cache[candidate.name] = scores
        rankings.append(_rank_candidate(candidate, scores, targets, records))
        if fold_coverage is None:
            fold_coverage = diagnostics
    rankings.sort(
        key=lambda row: (
            row["false_recapture_count"],
            -row["average_precision"],
            -row["roc_auc"],
            row["name"],
        )
    )
    selected_row = rankings[0]
    selected = next(item for item in candidates if item.name == selected_row["name"])
    selected_scores = score_cache[selected.name]
    threshold = float(selected_row["pooled_full_recall_threshold"])
    selected_flags = selected_scores >= threshold
    nested = _nested_threshold_diagnostic(
        selected,
        views,
        targets,
        folds,
        records,
        seed=args.seed + 1_000_000,
    )
    pooled = {
        **selected_row,
        "missed_recapture_image_ids": [
            int(record["image_id"])
            for record, flag, target in zip(records, selected_flags, targets)
            if target and not flag
        ],
        "false_recapture_image_ids": [
            int(record["image_id"])
            for record, flag, target in zip(records, selected_flags, targets)
            if flag and not target
        ],
    }
    maximum_false_count = math.floor(
        int((~targets).sum()) * args.maximum_false_recapture_rate + 1e-12
    )
    reason_policy = reason_conjunction_policy(views["scalar"], scalar_names, records, folds)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_scan_log_image_recapture_selector",
        "lifecycle_status": "active_experiment",
        "selection_scope": "pooled grouped OOF; no locked test by explicit experiment design",
        "image_count": len(records),
        "normal_image_count": int((~targets).sum()),
        "expected_recapture_count": int(targets.sum()),
        "folds": sorted(set(folds.tolist())),
        "detector_outer_folds": sorted(set(detector_folds.tolist())),
        "policy_fold_assignment": "reason-stratified perceptual groups",
        "policy_fold_reason_counts": {
            str(fold): dict(
                Counter(
                    reason
                    for record, assigned_fold in zip(records, folds)
                    if assigned_fold == fold
                    for reason in (record.get("expected_reason_codes", []) or ["ANNOTATED"])
                )
            )
            for fold in sorted(set(folds.tolist()))
        },
        "fold_coverage": fold_coverage,
        "classifier_source": "single_objects",
        "feature_sources": {
            "pixel_quality": True,
            "production_detector_query_and_geometry": args.predictions.name,
            "outer_fold_dfine_query_and_geometry": (
                args.auxiliary_predictions.name if args.auxiliary_predictions else None
            ),
            "dinov3_full_image_embedding": args.embedding_cache.name,
            "scalar_feature_count": len(scalar_names),
            "feature_view_dimensions": {
                name: int(values.shape[1]) for name, values in views.items()
            },
        },
        "candidate_count": len(candidates),
        "selected_pooled_oof": pooled,
        "nested_threshold_diagnostic": nested,
        "reason_conjunction_policy": reason_policy,
        "target_gate": {
            "required_recapture_recall": 1.0,
            "maximum_false_recapture_rate": args.maximum_false_recapture_rate,
            "maximum_false_recapture_count": maximum_false_count,
            "pooled_oof_pass": bool(
                pooled["recapture_recall"] == 1.0
                and pooled["false_recapture_count"] <= maximum_false_count
            ),
            "nested_diagnostic_pass": bool(
                nested["recapture_recall"] == 1.0
                and nested["false_recapture_count"] <= maximum_false_count
            ),
            "reason_pooled_oof_pass": bool(
                reason_policy["pooled_oof"]["recapture_recall"] == 1.0
                and reason_policy["pooled_oof"]["false_recapture_count"] <= maximum_false_count
            ),
            "reason_nested_diagnostic_pass": bool(
                reason_policy["nested_outer_fold"]["recapture_recall"] == 1.0
                and reason_policy["nested_outer_fold"]["false_recapture_count"]
                <= maximum_false_count
            ),
        },
        "leaderboard": rankings[:20],
        "limitations": [
            "The same grouped OOF predictions are used for candidate and pooled-threshold selection.",
            "The nested diagnostic is less optimistic but has only two training folds per outer fold.",
            "The selected DINOv3 and sklearn policy is not yet exported to ONNX or integrated into the Worker.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as stream:
            for record, score, flag, target, policy_fold in zip(
                records, selected_scores, selected_flags, targets, folds
            ):
                stream.write(
                    json.dumps(
                        {
                            "image_id": int(record["image_id"]),
                            "fold": int(record["fold"]),
                            "quality_policy_fold": int(policy_fold),
                            "expected_image_status": record["expected_image_status"],
                            "expected_reason_codes": record.get("expected_reason_codes", []),
                            "recapture_score": float(score),
                            "recapture_threshold": threshold,
                            "predicted_image_recapture": bool(flag),
                            "correct": bool(flag == target),
                        }
                    )
                    + "\n"
                )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a grouped-OOF scan-log image recapture policy"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--auxiliary-predictions", type=Path)
    parser.add_argument("--maximum-false-recapture-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
