from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
        current_labels = bundle["labels"][indexes].astype(np.int64)
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError("score bundle labels do not align")
        current_scores = bundle[array_name][indexes].astype(np.float64)
        standardized = (current_scores - current_scores.mean(axis=1, keepdims=True)) / (
            current_scores.std(axis=1, keepdims=True).clip(min=1e-6)
        )
        shifted = standardized - standardized.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        scores.append(probabilities)
    assert labels is not None
    return identifiers, labels, scores


def _candidate_rows(
    scores: list[np.ndarray], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    top1 = np.stack([values.argmax(axis=1) for values in scores], axis=1)
    rows = []
    targets = []
    mapping = []
    disagreement = np.asarray([len(set(row)) > 1 for row in top1])
    for object_index in np.flatnonzero(disagreement):
        candidates = sorted(set(top1[object_index].tolist()))
        for candidate in candidates:
            features = []
            for values in scores:
                probability = values[object_index]
                order = np.argsort(probability)[::-1]
                rank = int(np.flatnonzero(order == candidate)[0])
                features.extend(
                    [
                        probability[candidate],
                        probability[candidate] - probability[order[0]],
                        float(rank == 0),
                        float(rank == 1),
                        float(rank) / 19.0,
                        probability[order[0]],
                        probability[order[0]] - probability[order[1]],
                    ]
                )
            vote_count = int((top1[object_index] == candidate).sum())
            features.extend([vote_count / len(scores), float(vote_count == 1)])
            rows.append(features)
            targets.append(candidate == labels[object_index])
            mapping.append((int(object_index), int(candidate)))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(targets, dtype=bool),
        mapping,
        disagreement,
    )


def _predict(
    model,
    scores: list[np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    top1 = np.stack([values.argmax(axis=1) for values in scores], axis=1)
    predictions = np.apply_along_axis(lambda row: np.bincount(row, minlength=20).argmax(), 1, top1)
    features, _, mapping, disagreement = _candidate_rows(scores, labels)
    correctness = model.predict_proba(features)[:, 1]
    best: dict[int, tuple[float, int]] = {}
    for score, (object_index, candidate) in zip(correctness, mapping, strict=True):
        if object_index not in best or score > best[object_index][0]:
            best[object_index] = (float(score), candidate)
    for object_index, (_, candidate) in best.items():
        predictions[object_index] = candidate
    return predictions, disagreement


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a global disagreement candidate arbiter on source holdout only"
    )
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--array", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()
    if not len(args.source) == len(args.target) == len(args.array) == 3:
        parser.error("exactly three source, target, and array arguments are required")

    source_ids, source_labels, source_scores = _aligned(args.source, args.array)
    target_ids, target_labels, target_scores = _aligned(args.target, args.array)
    source_features, source_targets, source_mapping, _ = _candidate_rows(
        source_scores, source_labels
    )
    source_candidate_object_ids = np.asarray(
        [source_ids[object_index][0] for object_index, _ in source_mapping]
    )
    training = source_candidate_object_ids % 5 != 0
    candidates = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=20260904),
        ),
        "hist_gradient": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=20260904,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260904,
        ),
    }
    source_calibration_objects = np.asarray([identifier[0] % 5 == 0 for identifier in source_ids])
    sweep = []
    selected = None
    for name, model in candidates.items():
        model.fit(source_features[training], source_targets[training])
        predictions, disagreement = _predict(model, source_scores, source_labels)
        correct = predictions == source_labels
        selected_objects = source_calibration_objects
        row = {
            "model": name,
            "calibration_accuracy": float(correct[selected_objects].mean()),
            "calibration_disagreement_count": int((selected_objects & disagreement).sum()),
            "calibration_disagreement_accuracy": float(
                correct[selected_objects & disagreement].mean()
            ),
        }
        sweep.append(row)
        key = (row["calibration_accuracy"], row["calibration_disagreement_accuracy"])
        if selected is None or key > selected[0]:
            selected = (key, name)
    assert selected is not None
    _, selected_name = selected
    final_model = candidates[selected_name]
    final_model.fit(source_features, source_targets)
    source_predictions, source_disagreement = _predict(final_model, source_scores, source_labels)
    target_predictions, target_disagreement = _predict(final_model, target_scores, target_labels)
    source_correct = source_predictions == source_labels
    target_correct = target_predictions == target_labels
    report = {
        "schema_version": "1.0",
        "experiment": "source_only_global_disagreement_candidate_arbiter",
        "selection": "group-disjoint source calibration only",
        "development_target_used_for_selection": False,
        "feature_contract": "candidate-relative probabilities, ranks, margins, and vote count; no class identity",
        "candidate_sweep": sweep,
        "selected_model": selected_name,
        "source_all": {
            "accuracy": float(source_correct.mean()),
            "disagreement_count": int(source_disagreement.sum()),
            "disagreement_accuracy": float(source_correct[source_disagreement].mean()),
        },
        "target_diagnostic": {
            "object_count": int(len(target_labels)),
            "correct_count": int(target_correct.sum()),
            "error_count": int((~target_correct).sum()),
            "accuracy": float(target_correct.mean()),
            "disagreement_count": int(target_disagreement.sum()),
            "disagreement_accuracy": float(target_correct[target_disagreement].mean()),
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
    np.savez_compressed(
        args.scores_output,
        image_ids=np.asarray([identifier[0] for identifier in target_ids], dtype=np.int64),
        annotation_ids=np.asarray([identifier[1] for identifier in target_ids], dtype=np.int64),
        labels=target_labels,
        predictions=target_predictions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
