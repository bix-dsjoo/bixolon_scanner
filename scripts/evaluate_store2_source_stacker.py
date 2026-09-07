from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _ids(bundle: np.lib.npyio.NpzFile) -> list[tuple[int, int]]:
    return list(zip(bundle["image_ids"].tolist(), bundle["annotation_ids"].tolist(), strict=True))


def _aligned(
    paths: list[Path], arrays: list[str]
) -> tuple[list[tuple[int, int]], np.ndarray, list[np.ndarray]]:
    bundles = [np.load(path) for path in paths]
    identifiers = sorted(set.intersection(*(set(_ids(bundle)) for bundle in bundles)))
    labels = None
    scores = []
    for bundle, array_name in zip(bundles, arrays, strict=True):
        positions = {identifier: index for index, identifier in enumerate(_ids(bundle))}
        indexes = [positions[identifier] for identifier in identifiers]
        bundle_labels = bundle["labels"][indexes].astype(np.int64)
        if labels is None:
            labels = bundle_labels
        elif not np.array_equal(labels, bundle_labels):
            raise ValueError("score bundle labels do not align")
        scores.append(bundle[array_name][indexes].astype(np.float64))
    assert labels is not None
    return identifiers, labels, scores


def _probabilities(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    standardized = (scores - scores.mean(axis=1, keepdims=True)) / scores.std(
        axis=1, keepdims=True
    ).clip(min=1e-6)
    shifted = standardized - standardized.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return standardized, values / values.sum(axis=1, keepdims=True)


def _features(score_sets: list[np.ndarray]) -> np.ndarray:
    standardized = []
    probabilities = []
    for scores in score_sets:
        current_standardized, current_probabilities = _probabilities(scores)
        standardized.append(current_standardized)
        probabilities.append(current_probabilities)
    mean_probability = np.mean(probabilities, axis=0)
    minimum_probability = np.min(probabilities, axis=0)
    maximum_probability = np.max(probabilities, axis=0)
    return np.concatenate(
        [
            *standardized,
            *probabilities,
            mean_probability,
            minimum_probability,
            maximum_probability,
        ],
        axis=1,
    ).astype(np.float32)


def _visibility(manifest: Path, identifiers: list[tuple[int, int]]) -> np.ndarray:
    values = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        for annotation in record.get("annotations", []):
            values[(int(record["image_id"]), int(annotation["annotation_id"]))] = float(
                annotation.get("visible_fraction", 1.0)
            )
    return np.asarray([values[identifier] for identifier in identifiers])


def _model(name: str):
    if name.startswith("logistic_c"):
        c_value = float(name.removeprefix("logistic_c"))
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=3000, random_state=20260904),
        )
    if name == "hist_gradient":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=20260904,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=-1,
            random_state=20260904,
        )
    if name == "mlp_128":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=0.001,
                batch_size=256,
                early_stopping=True,
                max_iter=300,
                random_state=20260904,
            ),
        )
    raise ValueError(f"unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a global three-model classifier stacker on source holdout only"
    )
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--array", action="append", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()
    if not len(args.source) == len(args.target) == len(args.array) == 3:
        parser.error("exactly three source, target, and array arguments are required")

    source_ids, source_labels, source_scores = _aligned(args.source, args.array)
    target_ids, target_labels, target_scores = _aligned(args.target, args.array)
    source_features = _features(source_scores)
    target_features = _features(target_scores)
    visibility = _visibility(args.source_manifest, source_ids)
    image_ids = np.asarray([identifier[0] for identifier in source_ids])
    calibration = image_ids % 5 == 0
    training = ~calibration
    training_modes = {
        "all": training,
        "complete": training & (visibility >= 0.75),
        "mostly_visible": training & (visibility >= 0.55),
    }
    model_names = (
        "logistic_c0.01",
        "logistic_c0.1",
        "logistic_c1",
        "logistic_c10",
        "hist_gradient",
        "extra_trees",
        "mlp_128",
    )
    sweep = []
    selected = None
    for training_mode, selected_training in training_modes.items():
        for model_name in model_names:
            model = _model(model_name)
            model.fit(source_features[selected_training], source_labels[selected_training])
            predictions = model.predict(source_features[calibration])
            correct = predictions == source_labels[calibration]
            complete = visibility[calibration] >= 0.75
            partial = visibility[calibration] <= 0.45
            row = {
                "training_mode": training_mode,
                "model": model_name,
                "training_object_count": int(selected_training.sum()),
                "calibration_accuracy": float(correct.mean()),
                "calibration_complete_accuracy": float(correct[complete].mean()),
                "calibration_partial_accuracy": float(correct[partial].mean()),
            }
            sweep.append(row)
            key = (
                row["calibration_accuracy"],
                row["calibration_complete_accuracy"],
                row["calibration_partial_accuracy"],
            )
            if selected is None or key > selected[0]:
                selected = (key, training_mode, model_name)
    assert selected is not None
    _, selected_training_mode, selected_model_name = selected
    if selected_training_mode == "all":
        final_training = np.ones(len(source_labels), dtype=bool)
    elif selected_training_mode == "complete":
        final_training = visibility >= 0.75
    else:
        final_training = visibility >= 0.55
    final_model = _model(selected_model_name)
    final_model.fit(source_features[final_training], source_labels[final_training])
    source_predictions = final_model.predict(source_features)
    target_predictions = final_model.predict(target_features)
    source_correct = source_predictions == source_labels
    target_correct = target_predictions == target_labels
    report = {
        "schema_version": "1.0",
        "experiment": "source_only_global_classifier_stacker",
        "selection": "group-disjoint source calibration only",
        "development_target_used_for_selection": False,
        "feature_contract": "global normalized logits and probabilities from primary, detail, and verifier",
        "candidate_sweep": sweep,
        "selected_training_mode": selected_training_mode,
        "selected_model": selected_model_name,
        "source_all_accuracy": float(source_correct.mean()),
        "target_diagnostic": {
            "object_count": int(len(target_labels)),
            "correct_count": int(target_correct.sum()),
            "error_count": int((~target_correct).sum()),
            "accuracy": float(target_correct.mean()),
        },
        "target_errors": [
            {
                "image_id": int(target_ids[index][0]),
                "annotation_id": int(target_ids[index][1]),
                "expected": int(target_labels[index] + 1),
                "predicted": int(target_predictions[index] + 1),
            }
            for index in np.flatnonzero(~target_correct)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    target_probabilities = final_model.predict_proba(target_features)
    np.savez_compressed(
        args.scores_output,
        image_ids=np.asarray([identifier[0] for identifier in target_ids], dtype=np.int64),
        annotation_ids=np.asarray([identifier[1] for identifier in target_ids], dtype=np.int64),
        labels=target_labels,
        predictions=target_predictions,
        probabilities=target_probabilities.astype(np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
