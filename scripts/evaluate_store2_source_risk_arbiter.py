from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bixolon_scanner.training.scene_geometry_quality import extract_scene_geometry_features


def _aligned(
    paths: list[Path], arrays: list[str]
) -> tuple[list[tuple[int, int]], np.ndarray, list[np.ndarray]]:
    bundles = [np.load(path) for path in paths]
    identifiers = [
        list(zip(bundle["image_ids"].tolist(), bundle["annotation_ids"].tolist(), strict=True))
        for bundle in bundles
    ]
    common = sorted(set.intersection(*(set(values) for values in identifiers)))
    if not common:
        raise ValueError("score bundles have no common objects")
    labels = None
    scores = []
    for bundle, bundle_ids, array_name in zip(bundles, identifiers, arrays, strict=True):
        indexes_by_id = {identifier: index for index, identifier in enumerate(bundle_ids)}
        indexes = [indexes_by_id[identifier] for identifier in common]
        bundle_labels = bundle["labels"][indexes].astype(np.int64)
        if labels is None:
            labels = bundle_labels
        elif not np.array_equal(labels, bundle_labels):
            raise ValueError("score bundle labels do not align")
        scores.append(bundle[array_name][indexes].astype(np.float64))
    return common, labels, scores


def _probabilities(scores: np.ndarray) -> np.ndarray:
    standardized = (scores - scores.mean(axis=1, keepdims=True)) / scores.std(
        axis=1, keepdims=True
    ).clip(min=1e-6)
    standardized -= standardized.max(axis=1, keepdims=True)
    values = np.exp(standardized)
    return values / values.sum(axis=1, keepdims=True)


def _risk_features(score_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = [_probabilities(scores) for scores in score_sets]
    predictions = np.stack([values.argmax(axis=1) for values in probabilities], axis=1)
    majority = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=20).argmax(), 1, predictions
    )
    features = []
    for values in probabilities:
        ordered = np.sort(values, axis=1)
        entropy = -(values * np.log(values.clip(min=1e-12))).sum(axis=1) / np.log(20.0)
        majority_probability = values[np.arange(len(values)), majority]
        features.extend(
            [ordered[:, -1], ordered[:, -1] - ordered[:, -2], entropy, majority_probability]
        )
    for left in range(len(probabilities)):
        for right in range(left + 1, len(probabilities)):
            lhs = probabilities[left]
            rhs = probabilities[right]
            cosine = (lhs * rhs).sum(axis=1) / (
                np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1)
            ).clip(min=1e-12)
            midpoint = 0.5 * (lhs + rhs)
            js = 0.5 * (
                (lhs * np.log((lhs / midpoint).clip(min=1e-12))).sum(axis=1)
                + (rhs * np.log((rhs / midpoint).clip(min=1e-12))).sum(axis=1)
            )
            features.extend(
                [cosine, js, (predictions[:, left] == predictions[:, right]).astype(float)]
            )
    vote_count = np.stack(
        [(predictions == class_id).sum(axis=1) for class_id in range(20)], axis=1
    ).max(axis=1)
    features.extend([vote_count / predictions.shape[1], (vote_count == 3).astype(float)])
    return np.stack(features, axis=1).astype(np.float32), majority


def _geometry_features(manifest: Path, identifiers: list[tuple[int, int]]) -> np.ndarray:
    by_identifier = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        for annotation_index, annotation in enumerate(record.get("annotations", [])):
            identifier = (int(record["image_id"]), int(annotation["annotation_id"]))
            by_identifier[identifier] = extract_scene_geometry_features(record, annotation_index)
    missing = [identifier for identifier in identifiers if identifier not in by_identifier]
    if missing:
        raise ValueError(f"geometry manifest is missing {len(missing)} score objects")
    return np.stack([by_identifier[identifier] for identifier in identifiers])


def _threshold_at_correct_recap_rate(
    risk: np.ndarray, errors: np.ndarray, maximum_correct_recap_rate: float
) -> float:
    correct_risk = np.sort(risk[~errors])
    allowed = int(np.floor(maximum_correct_recap_rate * len(correct_risk)))
    return float(correct_risk[-(allowed + 1)])


def _metrics(risk: np.ndarray, errors: np.ndarray, threshold: float) -> dict[str, float | int]:
    recapture = risk > threshold
    return {
        "object_count": int(len(errors)),
        "error_count": int(errors.sum()),
        "recapture_count": int(recapture.sum()),
        "caught_error_count": int((recapture & errors).sum()),
        "remaining_error_count": int((~recapture & errors).sum()),
        "correct_recapture_count": int((recapture & ~errors).sum()),
        "correct_approved_count": int((~recapture & ~errors).sum()),
        "error_recall": float((recapture & errors).sum() / max(1, errors.sum())),
        "correct_recap_rate": float((recapture & ~errors).sum() / max(1, (~errors).sum())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a class-agnostic classifier-error arbiter using source validation only"
    )
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--array", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--maximum-correct-recap-rate", type=float, default=0.005)
    args = parser.parse_args()
    if not len(args.source) == len(args.target) == len(args.array) == 3:
        parser.error("exactly three aligned source, target, and array arguments are required")
    if not 0.0 <= args.maximum_correct_recap_rate < 1.0:
        parser.error("maximum correct recapture rate must be in [0, 1)")

    source_ids, source_labels, source_scores = _aligned(args.source, args.array)
    target_ids, target_labels, target_scores = _aligned(args.target, args.array)
    source_features, source_majority = _risk_features(source_scores)
    target_features, target_majority = _risk_features(target_scores)
    if (args.source_manifest is None) != (args.target_manifest is None):
        parser.error("source and target geometry manifests must be provided together")
    if args.source_manifest is not None:
        source_features = np.concatenate(
            [source_features, _geometry_features(args.source_manifest, source_ids)], axis=1
        )
        target_features = np.concatenate(
            [target_features, _geometry_features(args.target_manifest, target_ids)], axis=1
        )
    source_errors = source_majority != source_labels
    target_errors = target_majority != target_labels
    source_image_ids = np.asarray([identifier[0] for identifier in source_ids])
    calibration = source_image_ids % 5 == 0
    training = ~calibration

    candidates = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20261122),
        ),
        "hist_gradient_balanced": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=20261122,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=20261122,
        ),
    }
    sweep = []
    selected = None
    for name, model in candidates.items():
        model.fit(source_features[training], source_errors[training])
        calibration_risk = model.predict_proba(source_features[calibration])[:, 1]
        threshold = _threshold_at_correct_recap_rate(
            calibration_risk,
            source_errors[calibration],
            args.maximum_correct_recap_rate,
        )
        metrics = _metrics(calibration_risk, source_errors[calibration], threshold)
        row = {"model": name, "threshold": threshold, **metrics}
        sweep.append(row)
        key = (
            int(metrics["caught_error_count"]),
            -int(metrics["correct_recapture_count"]),
            int(metrics["correct_approved_count"]),
        )
        if selected is None or key > selected[0]:
            selected = (key, name, model, threshold)
    assert selected is not None
    _, selected_name, selected_model, threshold = selected
    source_risk = selected_model.predict_proba(source_features)[:, 1]
    target_risk = selected_model.predict_proba(target_features)[:, 1]
    source_metrics = _metrics(source_risk, source_errors, threshold)
    target_metrics = _metrics(target_risk, target_errors, threshold)
    target_recapture = target_risk > threshold
    report = {
        "schema_version": "1.0",
        "experiment": "source_only_class_agnostic_classifier_risk_arbiter",
        "selection": "group-disjoint source calibration only",
        "development_target_used_for_selection": False,
        "source_paths": [str(path) for path in args.source],
        "target_paths": [str(path) for path in args.target],
        "arrays": args.array,
        "feature_contract": "global confidence, entropy, score-distribution similarity, and vote agreement; no class identity",
        "geometry_features": args.source_manifest is not None,
        "source_training_object_count": int(training.sum()),
        "source_calibration_object_count": int(calibration.sum()),
        "candidate_sweep": sweep,
        "selected_model": selected_name,
        "selected_threshold": threshold,
        "source_all_metrics": source_metrics,
        "target_diagnostic_metrics": target_metrics,
        "target_remaining_errors": [
            {
                "image_id": int(target_ids[index][0]),
                "annotation_id": int(target_ids[index][1]),
                "expected": int(target_labels[index] + 1),
                "predicted": int(target_majority[index] + 1),
                "risk": float(target_risk[index]),
            }
            for index in np.flatnonzero(target_errors & ~target_recapture)
        ],
        "target_recaptures": [
            {
                "image_id": int(target_ids[index][0]),
                "annotation_id": int(target_ids[index][1]),
                "was_error": bool(target_errors[index]),
                "risk": float(target_risk[index]),
            }
            for index in np.flatnonzero(target_recapture)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        image_ids=np.asarray([identifier[0] for identifier in target_ids], dtype=np.int64),
        annotation_ids=np.asarray([identifier[1] for identifier in target_ids], dtype=np.int64),
        labels=target_labels,
        majority_predictions=target_majority,
        risk=target_risk.astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"target_remaining_errors", "target_recaptures"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
