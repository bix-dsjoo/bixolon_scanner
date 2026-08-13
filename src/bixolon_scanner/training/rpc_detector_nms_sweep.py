from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_detector import _metrics
from .rpc_data_scale import LEVELS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = config["detector"]
    detector_dir = args.output_dir / "detector"
    score_threshold = float(
        json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))[
            "selected_score_threshold"
        ]
    )
    records = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    predictions = {
        str(row["sample_key"]): row
        for row in _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    }
    rows: list[dict[str, Any]] = []
    for nms_threshold in np.linspace(0.30, 0.70, 17):
        row: dict[str, Any] = {"nms_iou_threshold": float(nms_threshold)}
        for role in ("calibration", "selection"):
            role_records = [record for record in records if record["role"] == role]
            role_predictions = [
                predictions[f"{record['source']}:{record['image_id']}"]
                for record in role_records
            ]
            role_report: dict[str, Any] = {
                "overall": _metrics(
                    role_records,
                    role_predictions,
                    score_threshold=score_threshold,
                    nms_iou_threshold=float(nms_threshold),
                    match_iou_threshold=float(options["match_iou_threshold"]),
                    max_queries=int(options["max_queries"]),
                )
            }
            role_report["difficulty"] = {}
            for level in LEVELS:
                subset = [record for record in role_records if record["level"] == level]
                subset_predictions = [
                    predictions[f"{record['source']}:{record['image_id']}"]
                    for record in subset
                ]
                role_report["difficulty"][level] = _metrics(
                    subset,
                    subset_predictions,
                    score_threshold=score_threshold,
                    nms_iou_threshold=float(nms_threshold),
                    match_iou_threshold=float(options["match_iou_threshold"]),
                    max_queries=int(options["max_queries"]),
                )
            row[role] = role_report
        rows.append(row)
    result = {"score_threshold": score_threshold, "rows": rows}
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
