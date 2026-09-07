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


def _visibility(path: Path) -> dict[tuple[int, int], float]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        for annotation in record.get("annotations", []):
            output[(int(record["image_id"]), int(annotation["annotation_id"]))] = float(
                annotation["visible_fraction"]
            )
    return output


def _threshold(scores: np.ndarray, labels: np.ndarray, rate: float) -> float:
    complete = np.sort(scores[labels == 0])
    allowed = int(np.floor(rate * len(complete)))
    return float(complete[-(allowed + 1)])


def _metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int | float]:
    recapture = scores > threshold
    return {
        "sample_count": int(len(labels)),
        "partial_count": int(labels.sum()),
        "recapture_count": int(recapture.sum()),
        "partial_recall": float((recapture & (labels == 1)).sum() / max(1, labels.sum())),
        "complete_recap_rate": float(
            (recapture & (labels == 0)).sum() / max(1, (labels == 0).sum())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a nonlinear completeness rejector on frozen source-only features"
    )
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--partial-maximum", type=float, default=0.45)
    parser.add_argument("--complete-minimum", type=float, default=0.75)
    parser.add_argument("--maximum-complete-recap-rate", type=float, default=0.005)
    args = parser.parse_args()

    source = np.load(args.source_features)
    source_ids = list(
        zip(source["image_ids"].tolist(), source["annotation_ids"].tolist(), strict=True)
    )
    visibility = _visibility(args.source_manifest)
    visible = np.asarray([visibility[identifier] for identifier in source_ids])
    selected = (visible <= args.partial_maximum) | (visible >= args.complete_minimum)
    features = source["evaluation_features"][selected].astype(np.float32)
    labels = (visible[selected] <= args.partial_maximum).astype(np.int64)
    image_ids = source["image_ids"][selected]
    training = image_ids % 5 != 0
    calibration = ~training

    candidates = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20261126),
        ),
        "mlp_64_16": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 16),
                alpha=0.01,
                batch_size=128,
                learning_rate_init=0.0005,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=20261126,
            ),
        ),
        "hist_gradient": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=20261126,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.5,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=20261126,
        ),
    }
    sweep = []
    chosen = None
    for name, model in candidates.items():
        model.fit(features[training], labels[training])
        risk = model.predict_proba(features[calibration])[:, 1]
        threshold = _threshold(risk, labels[calibration], args.maximum_complete_recap_rate)
        metrics = _metrics(risk, labels[calibration], threshold)
        row = {"model": name, "threshold": threshold, **metrics}
        sweep.append(row)
        key = (float(metrics["partial_recall"]), -float(metrics["complete_recap_rate"]))
        if chosen is None or key > chosen[0]:
            chosen = (key, name, model, threshold)
    assert chosen is not None
    _, name, model, threshold = chosen

    source_risk = model.predict_proba(features)[:, 1]
    target = np.load(args.target_features)
    target_risk = model.predict_proba(target["evaluation_features"].astype(np.float32))[:, 1]
    decisions = np.load(args.target_decisions)
    decision_by_id = {
        (int(decisions["image_ids"][index]), int(decisions["annotation_ids"][index])): index
        for index in range(len(decisions["labels"]))
    }
    target_indexes = [
        decision_by_id[(int(image_id), int(annotation_id))]
        for image_id, annotation_id in zip(
            target["image_ids"], target["annotation_ids"], strict=True
        )
    ]
    expected = decisions["labels"][target_indexes]
    predicted = decisions["majority_predictions"][target_indexes]
    errors = predicted != expected
    recapture = target_risk > threshold
    target_metrics = {
        "object_count": int(len(errors)),
        "error_count": int(errors.sum()),
        "recapture_count": int(recapture.sum()),
        "caught_error_count": int((recapture & errors).sum()),
        "remaining_error_count": int((~recapture & errors).sum()),
        "correct_recapture_count": int((recapture & ~errors).sum()),
        "correct_approved_count": int((~recapture & ~errors).sum()),
    }
    report = {
        "schema_version": "1.0",
        "experiment": "frozen_feature_nonlinear_completeness_rejector",
        "selection": "group-disjoint source-only calibration",
        "development_target_used_for_selection": False,
        "partial_maximum": args.partial_maximum,
        "complete_minimum": args.complete_minimum,
        "maximum_complete_recap_rate": args.maximum_complete_recap_rate,
        "candidate_sweep": sweep,
        "selected_model": name,
        "selected_threshold": threshold,
        "source_all_metrics": _metrics(source_risk, labels, threshold),
        "target_diagnostic_metrics": target_metrics,
        "target_remaining_errors": [
            {
                "image_id": int(target["image_ids"][index]),
                "annotation_id": int(target["annotation_ids"][index]),
                "risk": float(target_risk[index]),
            }
            for index in np.flatnonzero(errors & ~recapture)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        image_ids=target["image_ids"],
        annotation_ids=target["annotation_ids"],
        risk=target_risk.astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "target_remaining_errors"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
