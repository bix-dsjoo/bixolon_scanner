from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bixolon_scanner.training.scene_geometry_quality import (
    extract_scene_geometry_features,
)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rows(records: list[dict], *, partial_maximum: float, complete_minimum: float):
    rows = []
    features = []
    labels = []
    for record in records:
        for annotation_index, annotation in enumerate(record.get("annotations", [])):
            visible = float(annotation.get("visible_fraction", 1.0))
            if visible <= partial_maximum:
                complete = 0
            elif visible >= complete_minimum:
                complete = 1
            else:
                continue
            rows.append(
                {
                    "image_id": int(record["image_id"]),
                    "annotation_id": int(annotation.get("annotation_id", 0)),
                    "category_id": int(annotation["category_id"]),
                    "visible_fraction": visible,
                    "complete": complete,
                }
            )
            features.append(extract_scene_geometry_features(record, annotation_index))
            labels.append(complete)
    return rows, np.stack(features), np.asarray(labels, dtype=np.int64)


def _evaluation_rows(records: list[dict]):
    rows = []
    features = []
    for record in records:
        for annotation_index, annotation in enumerate(record.get("annotations", [])):
            rows.append(
                {
                    "image_id": int(record["image_id"]),
                    "annotation_id": int(annotation.get("annotation_id", 0)),
                    "category_id": int(annotation["category_id"]),
                }
            )
            features.append(extract_scene_geometry_features(record, annotation_index))
    return rows, np.stack(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BIX-only scene geometry quality")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial-maximum", type=float, default=0.45)
    parser.add_argument("--complete-minimum", type=float, default=0.75)
    parser.add_argument("--complete-recapture-rate", type=float, default=0.005)
    parser.add_argument(
        "--relative-only",
        action="store_true",
        help="Discard absolute frame scale and position features for cross-camera transfer",
    )
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_features, train_labels = _rows(
        _records(args.train_manifest),
        partial_maximum=args.partial_maximum,
        complete_minimum=args.complete_minimum,
    )
    validation_rows, validation_features, validation_labels = _rows(
        _records(args.validation_manifest),
        partial_maximum=args.partial_maximum,
        complete_minimum=args.complete_minimum,
    )
    evaluation_rows, evaluation_features = _evaluation_rows(_records(args.evaluation_manifest))
    if args.relative_only:
        relative_indices = np.asarray([3, 4, 5, 6, 7, 10, 11], dtype=np.int64)
        train_features = train_features[:, relative_indices]
        validation_features = validation_features[:, relative_indices]
        evaluation_features = evaluation_features[:, relative_indices]
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(train_features, train_labels)
    validation_scores = model.predict_proba(validation_features)[:, 1]
    complete_scores = np.sort(validation_scores[validation_labels == 1])
    threshold_index = min(
        len(complete_scores) - 1,
        int(np.floor(args.complete_recapture_rate * len(complete_scores))),
    )
    threshold = float(complete_scores[threshold_index])
    evaluation_scores = model.predict_proba(evaluation_features)[:, 1]
    joblib.dump(model, args.output_dir / "scene-geometry-quality.joblib")
    with (args.output_dir / "evaluation-scores.jsonl").open("w", encoding="utf-8") as stream:
        for row, score in zip(evaluation_rows, evaluation_scores, strict=True):
            stream.write(
                json.dumps({**row, "geometry_complete_score": float(score)}, separators=(",", ":"))
                + "\n"
            )
    partial = validation_labels == 0
    complete = validation_labels == 1
    report = {
        "schema_version": "1.0",
        "experiment": "source_only_scene_geometry_quality",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "feature_count": int(train_features.shape[1]),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "selected_complete_score_threshold": threshold,
        "validation_complete_recap_rate": float(np.mean(validation_scores[complete] < threshold)),
        "validation_partial_recap_rate": float(np.mean(validation_scores[partial] < threshold)),
        "validation_accuracy_at_0_5": float(
            np.mean((validation_scores >= 0.5) == validation_labels)
        ),
        "evaluation_count": len(evaluation_rows),
        "evaluation_recap_count": int(np.sum(evaluation_scores < threshold)),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
