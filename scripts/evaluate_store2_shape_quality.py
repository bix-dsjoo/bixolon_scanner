from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import HistGradientBoostingClassifier

from bixolon_scanner.training.shape_quality import extract_shape_quality_features


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _samples(records: list[dict], *, low: float, high: float) -> list[dict]:
    rows = []
    for record in records:
        for annotation in record.get("annotations", []):
            visible = float(annotation.get("visible_fraction", 1.0))
            if visible <= low:
                complete = 0
            elif visible >= high:
                complete = 1
            else:
                continue
            rows.append(
                {
                    "image_id": int(record["image_id"]),
                    "image_path": str(record["image_path"]),
                    "annotation_id": int(annotation.get("annotation_id", 0)),
                    "category_id": int(annotation["category_id"]),
                    "bbox_xywh": annotation["bbox_xywh"],
                    "visible_fraction": visible,
                    "complete": complete,
                }
            )
    return rows


def _balanced(rows: list[dict], maximum_per_label: int, seed: int) -> list[dict]:
    generator = np.random.default_rng(seed)
    output = []
    for label in (0, 1):
        options = [row for row in rows if row["complete"] == label]
        count = min(maximum_per_label, len(options))
        indexes = generator.choice(len(options), count, replace=False)
        output.extend(options[int(index)] for index in indexes)
    generator.shuffle(output)
    return output


def _fragment_rows(
    rows: list[dict],
    seed: int,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
) -> list[dict]:
    generator = np.random.default_rng(seed)
    output = []
    for row in rows:
        if row["complete"] != 1:
            continue
        x, y, width, height = (float(value) for value in row["bbox_xywh"])
        fraction = float(generator.uniform(minimum_fraction, maximum_fraction))
        edge = int(generator.integers(0, 4))
        if edge == 0:
            width *= fraction
        elif edge == 1:
            x += width * (1.0 - fraction)
            width *= fraction
        elif edge == 2:
            height *= fraction
        else:
            y += height * (1.0 - fraction)
            height *= fraction
        output.append(
            {
                **row,
                "bbox_xywh": [x, y, width, height],
                "visible_fraction": fraction,
                "complete": 0,
                "synthetic_fragment": True,
            }
        )
    return output


def _features(root: Path, rows: list[dict], margin: float) -> np.ndarray:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["image_path"]].append((index, row))
    values: list[np.ndarray | None] = [None] * len(rows)
    for image_path, indexed_rows in grouped.items():
        with Image.open(root / image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for index, row in indexed_rows:
                x, y, width, height = (float(value) for value in row["bbox_xywh"])
                crop = image.crop(
                    (
                        max(0, math.floor(x - width * margin)),
                        max(0, math.floor(y - height * margin)),
                        min(image.width, math.ceil(x + width * (1.0 + margin))),
                        min(image.height, math.ceil(y + height * (1.0 + margin))),
                    )
                )
                values[index] = extract_shape_quality_features(crop)
    if any(value is None for value in values):
        raise RuntimeError("shape feature extraction lost a sample")
    return np.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a source-only global shape rejector")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial-maximum", type=float, default=0.45)
    parser.add_argument("--complete-minimum", type=float, default=0.75)
    parser.add_argument("--train-per-label", type=int, default=12000)
    parser.add_argument("--complete-recapture-rate", type=float, default=0.005)
    parser.add_argument("--fragment-minimum", type=float, default=0.25)
    parser.add_argument("--fragment-maximum", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if not 0.0 < args.fragment_minimum < args.fragment_maximum <= 1.0:
        parser.error("fragment range must satisfy 0 < minimum < maximum <= 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_rows = _samples(
        _records(args.train_manifest),
        low=args.partial_maximum,
        high=args.complete_minimum,
    )
    natural_partial = _balanced(
        [row for row in raw_train_rows if row["complete"] == 0],
        args.train_per_label // 2,
        args.seed,
    )
    fragments = _balanced(
        _fragment_rows(
            raw_train_rows,
            args.seed + 1,
            minimum_fraction=args.fragment_minimum,
            maximum_fraction=args.fragment_maximum,
        ),
        args.train_per_label // 2,
        args.seed + 2,
    )
    complete_rows = _balanced(
        [row for row in raw_train_rows if row["complete"] == 1],
        args.train_per_label,
        args.seed + 3,
    )
    train_rows = [
        *[row for row in natural_partial if row["complete"] == 0],
        *[row for row in fragments if row["complete"] == 0],
        *[row for row in complete_rows if row["complete"] == 1],
    ]
    validation_rows = _samples(
        _records(args.validation_manifest),
        low=args.partial_maximum,
        high=args.complete_minimum,
    )
    evaluation_rows = _samples(
        _records(args.evaluation_manifest),
        low=args.partial_maximum,
        high=args.complete_minimum,
    )
    train_features = _features(args.train_root, train_rows, args.margin)
    validation_features = _features(args.validation_root, validation_rows, args.margin)
    evaluation_features = _features(args.evaluation_root, evaluation_rows, args.margin)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    model.fit(train_features, np.asarray([row["complete"] for row in train_rows]))
    validation_scores = model.predict_proba(validation_features)[:, 1]
    complete = np.asarray([row["complete"] == 1 for row in validation_rows])
    complete_scores = np.sort(validation_scores[complete])
    threshold_index = min(
        len(complete_scores) - 1,
        int(np.floor(args.complete_recapture_rate * len(complete_scores))),
    )
    threshold = float(complete_scores[threshold_index])
    evaluation_scores = model.predict_proba(evaluation_features)[:, 1]
    joblib.dump(model, args.output_dir / "shape-quality.joblib")
    rows_output = []
    for row, score in zip(evaluation_rows, evaluation_scores, strict=True):
        rows_output.append({**row, "shape_complete_score": float(score)})
    (args.output_dir / "evaluation-scores.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_output),
        encoding="utf-8",
    )
    partial = ~complete
    report = {
        "schema_version": "1.0",
        "experiment": "source_only_global_shape_rejector",
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
            np.mean((validation_scores >= 0.5) == complete.astype(np.int64))
        ),
        "evaluation_count": len(evaluation_rows),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
