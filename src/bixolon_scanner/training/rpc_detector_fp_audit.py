from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import softmax
from .rpc_worker_gate import _iou, postprocess_worker_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.output_dir / "runs" / "full" / f"seed{config['experiment']['seeds'][0]}"
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    archive = np.load(run_dir / "selection_predictions.npz")
    classifier_probabilities = softmax(
        archive["logits"], float(calibration["temperature"])
    )
    approved_unmatched_ids = set(
        archive["sample_ids"][(archive["targets"] < 0) & (
            classifier_probabilities.max(axis=1) >= float(calibration["approval_threshold"])
        )].astype(str).tolist()
    )
    detector_dir = args.output_dir / "detector"
    threshold = json.loads(
        (detector_dir / "threshold.json").read_text(encoding="utf-8")
    )["selected_score_threshold"]
    options = dict(config["detector"], score_threshold=threshold)
    records = [
        json.loads(line)
        for line in (detector_dir / "manifest" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions = {
        row["sample_key"]: row
        for row in (
            json.loads(line)
            for line in (detector_dir / "predictions" / "val_oof.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    rows = []
    for record in records:
        if record["role"] != "selection":
            continue
        result = postprocess_worker_gate(
            record,
            predictions[f"{record['source']}:{record['image_id']}"],
            options,
        )
        detections = result["detections"]
        for index in result["unmatched_detection_indices"]:
            box = detections[int(index)]["bbox_xyxy"]
            overlaps = [
                _iou(box, detection["bbox_xyxy"])
                for other, detection in enumerate(detections)
                if other != int(index)
            ]
            rows.append(
                {
                    "level": record["level"],
                    "score": float(detections[int(index)]["score"]),
                    "max_detection_iou": max(overlaps, default=0.0),
                    "detection_count": len(detections),
                    "classifier_approved": (
                        f"val:{record['image_id']}:det{index}" in approved_unmatched_ids
                    ),
                }
            )
    overlap = np.asarray([row["max_detection_iou"] for row in rows])
    score = np.asarray([row["score"] for row in rows])
    report = {
        "count": len(rows),
        "score_quantiles": np.quantile(score, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "overlap_quantiles": np.quantile(
            overlap, [0, 0.1, 0.5, 0.9, 0.95, 0.99, 1]
        ).tolist(),
        "overlap_threshold_counts": {
            str(value): int((overlap >= value).sum())
            for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        },
        "by_level": {
            level: sum(row["level"] == level for row in rows)
            for level in ("easy", "medium", "hard")
        },
        "classifier_approved": {},
    }
    approved = [row for row in rows if row["classifier_approved"]]
    approved_overlap = np.asarray([row["max_detection_iou"] for row in approved])
    report["classifier_approved"] = {
        "count": len(approved),
        "overlap_quantiles": np.quantile(
            approved_overlap, [0, 0.1, 0.5, 0.9, 0.95, 0.99, 1]
        ).tolist(),
        "overlap_threshold_counts": {
            str(value): int((approved_overlap >= value).sum())
            for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        },
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
