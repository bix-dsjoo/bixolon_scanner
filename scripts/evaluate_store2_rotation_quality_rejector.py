from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _identifiers(bundle: np.lib.npyio.NpzFile) -> list[tuple[int, int]]:
    return list(zip(bundle["image_ids"].tolist(), bundle["annotation_ids"].tolist(), strict=True))


def _take(
    bundle: np.lib.npyio.NpzFile,
    identifiers: list[tuple[int, int]],
    array_name: str,
) -> np.ndarray:
    positions = {identifier: index for index, identifier in enumerate(_identifiers(bundle))}
    missing = [identifier for identifier in identifiers if identifier not in positions]
    if missing:
        raise ValueError(f"score bundle is missing {len(missing)} objects")
    return bundle[array_name][[positions[identifier] for identifier in identifiers]]


def _softmax(scores: np.ndarray) -> np.ndarray:
    standardized = (scores - scores.mean(axis=-1, keepdims=True)) / scores.std(
        axis=-1, keepdims=True
    ).clip(min=1e-6)
    standardized -= standardized.max(axis=-1, keepdims=True)
    values = np.exp(standardized)
    return values / values.sum(axis=-1, keepdims=True)


def _rotation_features(scores: np.ndarray) -> np.ndarray:
    probabilities = _softmax(scores.astype(np.float64))
    predictions = probabilities.argmax(axis=2)
    vote_count = np.stack(
        [(predictions == class_id).sum(axis=1) for class_id in range(20)], axis=1
    ).max(axis=1)
    ordered = np.sort(probabilities, axis=2)
    confidence = ordered[:, :, -1]
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    entropy = -(probabilities * np.log(probabilities.clip(min=1e-12))).sum(axis=2) / np.log(20.0)
    js_values = []
    for left in range(probabilities.shape[1]):
        for right in range(left + 1, probabilities.shape[1]):
            lhs = probabilities[:, left]
            rhs = probabilities[:, right]
            midpoint = 0.5 * (lhs + rhs)
            js_values.append(
                0.5
                * (
                    (lhs * np.log((lhs / midpoint).clip(min=1e-12))).sum(axis=1)
                    + (rhs * np.log((rhs / midpoint).clip(min=1e-12))).sum(axis=1)
                )
            )
    js = np.stack(js_values, axis=1)
    mean_probability = probabilities.mean(axis=1)
    mean_ordered = np.sort(mean_probability, axis=1)
    features = np.stack(
        [
            vote_count / predictions.shape[1],
            np.apply_along_axis(lambda row: len(set(row)), 1, predictions),
            confidence.mean(axis=1),
            confidence.min(axis=1),
            confidence.std(axis=1),
            margin.mean(axis=1),
            margin.min(axis=1),
            margin.std(axis=1),
            entropy.mean(axis=1),
            entropy.max(axis=1),
            entropy.std(axis=1),
            js.mean(axis=1),
            js.max(axis=1),
            probabilities.std(axis=1).mean(axis=1),
            probabilities.std(axis=1).max(axis=1),
            mean_ordered[:, -1],
            mean_ordered[:, -1] - mean_ordered[:, -2],
        ],
        axis=1,
    )
    return features.astype(np.float32)


def _visibility(manifest: Path, identifiers: list[tuple[int, int]]) -> np.ndarray:
    by_identifier: dict[tuple[int, int], float] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        for annotation in record.get("annotations", []):
            identifier = (int(record["image_id"]), int(annotation["annotation_id"]))
            by_identifier[identifier] = float(annotation.get("visible_fraction", 1.0))
    missing = [identifier for identifier in identifiers if identifier not in by_identifier]
    if missing:
        raise ValueError(f"manifest is missing {len(missing)} score objects")
    return np.asarray([by_identifier[identifier] for identifier in identifiers])


def _baseline(
    paths: list[Path], arrays: list[str]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    bundles = [np.load(path) for path in paths]
    identifier_sets = [set(_identifiers(bundle)) for bundle in bundles]
    identifiers = sorted(set.intersection(*identifier_sets))
    labels = _take(bundles[0], identifiers, "labels").astype(np.int64)
    predictions = []
    for bundle, array_name in zip(bundles, arrays, strict=True):
        bundle_labels = _take(bundle, identifiers, "labels").astype(np.int64)
        if not np.array_equal(labels, bundle_labels):
            raise ValueError("baseline score labels do not align")
        predictions.append(_take(bundle, identifiers, array_name).argmax(axis=1))
    votes = np.stack(predictions, axis=1)
    majority = np.apply_along_axis(lambda row: np.bincount(row, minlength=20).argmax(), 1, votes)
    return identifiers, labels, majority


def _threshold(risk: np.ndarray, partial: np.ndarray, maximum_complete_recap_rate: float) -> float:
    complete_risk = np.sort(risk[~partial])
    allowed = int(np.floor(maximum_complete_recap_rate * len(complete_risk)))
    return float(complete_risk[-(allowed + 1)])


def _quality_metrics(
    risk: np.ndarray, partial: np.ndarray, threshold: float
) -> dict[str, float | int]:
    recapture = risk > threshold
    return {
        "object_count": int(len(partial)),
        "partial_count": int(partial.sum()),
        "complete_count": int((~partial).sum()),
        "recapture_count": int(recapture.sum()),
        "partial_recapture_count": int((recapture & partial).sum()),
        "complete_recapture_count": int((recapture & ~partial).sum()),
        "partial_recall": float((recapture & partial).sum() / max(1, partial.sum())),
        "complete_recap_rate": float((recapture & ~partial).sum() / max(1, (~partial).sum())),
    }


def _target_metrics(
    risk: np.ndarray, errors: np.ndarray, threshold: float
) -> dict[str, float | int]:
    recapture = risk > threshold
    return {
        "object_count": int(len(errors)),
        "error_count": int(errors.sum()),
        "recapture_count": int(recapture.sum()),
        "caught_error_count": int((recapture & errors).sum()),
        "remaining_error_count": int((~recapture & errors).sum()),
        "correct_recapture_count": int((recapture & ~errors).sum()),
        "correct_approved_count": int((~recapture & ~errors).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a class-agnostic rotation-consistency quality rejector on source data"
    )
    parser.add_argument("--source-rotation", type=Path, required=True)
    parser.add_argument("--target-rotation", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-baseline", type=Path, action="append", required=True)
    parser.add_argument("--target-baseline", type=Path, action="append", required=True)
    parser.add_argument("--baseline-array", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--partial-visible-threshold", type=float, default=0.75)
    parser.add_argument("--maximum-complete-recap-rate", type=float, default=0.005)
    args = parser.parse_args()
    if not (
        len(args.source_baseline) == len(args.target_baseline) == len(args.baseline_array) == 3
    ):
        parser.error("exactly three source/target baseline score bundles are required")

    source_rotation = np.load(args.source_rotation)
    target_rotation = np.load(args.target_rotation)
    source_ids = _identifiers(source_rotation)
    target_ids, target_labels, target_majority = _baseline(
        args.target_baseline, args.baseline_array
    )
    source_baseline_ids, _, _ = _baseline(args.source_baseline, args.baseline_array)
    source_common = sorted(set(source_ids) & set(source_baseline_ids))
    target_common = sorted(set(_identifiers(target_rotation)) & set(target_ids))
    source_features = _rotation_features(
        _take(source_rotation, source_common, "rotation_head_scores")
    )
    target_features = _rotation_features(
        _take(target_rotation, target_common, "rotation_head_scores")
    )
    visibility = _visibility(args.source_manifest, source_common)
    partial = visibility < args.partial_visible_threshold
    target_positions = {identifier: index for index, identifier in enumerate(target_ids)}
    target_indexes = [target_positions[identifier] for identifier in target_common]
    target_labels = target_labels[target_indexes]
    target_majority = target_majority[target_indexes]
    target_errors = target_majority != target_labels

    image_ids = np.asarray([identifier[0] for identifier in source_common])
    calibration = image_ids % 5 == 0
    training = ~calibration
    candidates = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260903),
        ),
        "hist_gradient_balanced": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=20260903,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260903,
        ),
    }
    sweep = []
    selected = None
    for name, model in candidates.items():
        model.fit(source_features[training], partial[training])
        calibration_risk = model.predict_proba(source_features[calibration])[:, 1]
        threshold = _threshold(
            calibration_risk,
            partial[calibration],
            args.maximum_complete_recap_rate,
        )
        metrics = _quality_metrics(calibration_risk, partial[calibration], threshold)
        row = {"model": name, "threshold": threshold, **metrics}
        sweep.append(row)
        key = (
            int(metrics["partial_recapture_count"]),
            -int(metrics["complete_recapture_count"]),
        )
        if selected is None or key > selected[0]:
            selected = (key, name, model, threshold)
    assert selected is not None
    _, selected_name, selected_model, threshold = selected
    source_risk = selected_model.predict_proba(source_features)[:, 1]
    target_risk = selected_model.predict_proba(target_features)[:, 1]
    source_metrics = _quality_metrics(source_risk, partial, threshold)
    target_metrics = _target_metrics(target_risk, target_errors, threshold)
    target_recapture = target_risk > threshold
    report = {
        "schema_version": "1.0",
        "experiment": "source_only_rotation_consistency_quality_rejector",
        "selection": "group-disjoint source calibration only",
        "development_target_used_for_selection": False,
        "feature_contract": "class-agnostic rotation confidence, entropy, vote, and Jensen-Shannon consistency",
        "partial_visible_threshold": args.partial_visible_threshold,
        "maximum_complete_recap_rate": args.maximum_complete_recap_rate,
        "source_training_object_count": int(training.sum()),
        "source_calibration_object_count": int(calibration.sum()),
        "candidate_sweep": sweep,
        "selected_model": selected_name,
        "selected_threshold": threshold,
        "source_all_metrics": source_metrics,
        "target_diagnostic_metrics": target_metrics,
        "target_remaining_errors": [
            {
                "image_id": int(target_common[index][0]),
                "annotation_id": int(target_common[index][1]),
                "expected": int(target_labels[index] + 1),
                "predicted": int(target_majority[index] + 1),
                "risk": float(target_risk[index]),
            }
            for index in np.flatnonzero(target_errors & ~target_recapture)
        ],
        "target_recaptures": [
            {
                "image_id": int(target_common[index][0]),
                "annotation_id": int(target_common[index][1]),
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
        image_ids=np.asarray([identifier[0] for identifier in target_common], dtype=np.int64),
        annotation_ids=np.asarray([identifier[1] for identifier in target_common], dtype=np.int64),
        labels=target_labels,
        majority_predictions=target_majority,
        risk=target_risk.astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
