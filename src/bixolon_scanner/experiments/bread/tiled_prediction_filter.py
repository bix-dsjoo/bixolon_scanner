from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def touches_internal_tile_boundary(
    box: list[float],
    window: list[int],
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> bool:
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("internal tile margin ratio must be in [0, 0.5)")
    x1, y1, x2, y2 = box
    tile_x1, tile_y1, tile_x2, tile_y2 = window
    margin_x = (tile_x2 - tile_x1) * margin_ratio
    margin_y = (tile_y2 - tile_y1) * margin_ratio
    return bool(
        (tile_x1 > 0 and x1 <= tile_x1 + margin_x)
        or (tile_x2 < image_width and x2 >= tile_x2 - margin_x)
        or (tile_y1 > 0 and y1 <= tile_y1 + margin_y)
        or (tile_y2 < image_height and y2 >= tile_y2 - margin_y)
    )


def filter_tiled_prediction_row(row: dict[str, Any], *, margin_ratio: float) -> dict[str, Any]:
    windows = row["tile_windows"]
    count = len(row["boxes_xyxy"])
    if not windows or count % len(windows):
        raise ValueError("tiled prediction candidates are not aligned with tile windows")
    queries_per_tile = count // len(windows)
    image_width = max(window[2] for window in windows)
    image_height = max(window[3] for window in windows)
    kept = []
    for tile_index, window in enumerate(windows):
        start = tile_index * queries_per_tile
        end = start + queries_per_tile
        kept.extend(
            index
            for index in range(start, end)
            if not touches_internal_tile_boundary(
                row["boxes_xyxy"][index],
                window,
                image_width=image_width,
                image_height=image_height,
                margin_ratio=margin_ratio,
            )
        )
    output = {**row}
    for field in ("boxes_xyxy", "scores", "class_ids", "top3_class_ids"):
        if field in row:
            if len(row[field]) != count:
                raise ValueError(f"tiled prediction field {field} is not candidate-aligned")
            output[field] = [row[field][index] for index in kept]
    output["raw_tile_candidate_count"] = count
    output["internal_boundary_filtered_count"] = count - len(kept)
    output["internal_boundary_margin_ratio"] = margin_ratio
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line
    ]
    filtered = [filter_tiled_prediction_row(row, margin_ratio=args.margin_ratio) for row in rows]
    input_count = sum(len(row["boxes_xyxy"]) for row in rows)
    output_count = sum(len(row["boxes_xyxy"]) for row in filtered)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_tiled_internal_boundary_filter",
        "selection_scope": "fixed normalized internal-boundary rule without labels or image ids",
        "image_count": len(rows),
        "margin_ratio": args.margin_ratio,
        "input_candidate_count": input_count,
        "output_candidate_count": output_count,
        "filtered_candidate_count": input_count - output_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in filtered), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove tile detections that touch an artificial internal crop boundary"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--margin-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
