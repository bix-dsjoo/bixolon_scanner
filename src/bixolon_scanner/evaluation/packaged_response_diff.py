from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.catalog import sha256_file

COMPONENTS = (
    "status",
    "reason_codes",
    "segmentation_count",
    "bbox",
    "item_status",
    "prediction",
    "top3",
    "confidence",
    "version_null_pattern",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("packaged response evidence contains no rows")
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in indexed or not isinstance(row.get("response"), dict):
            raise ValueError("packaged response evidence has duplicate IDs or missing responses")
        indexed[image_id] = row
    return indexed


def _latency(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"sample_count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": round(float(array.mean()), 3),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "p99_ms": round(float(np.percentile(array, 99)), 3),
    }


def compare_packaged_responses(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _index(_rows(baseline_path))
    candidate = _index(_rows(candidate_path))
    if set(baseline) != set(candidate):
        raise ValueError("packaged response evidence image IDs differ")

    differences = {component: set() for component in COMPONENTS}
    maximum_confidence_delta = 0.0
    baseline_full_path_latency: list[float] = []
    candidate_full_path_latency: list[float] = []
    for image_id in sorted(baseline):
        left = baseline[image_id]["response"]
        right = candidate[image_id]["response"]
        if left["status"] != right["status"]:
            differences["status"].add(image_id)
        if left["reason_codes"] != right["reason_codes"]:
            differences["reason_codes"].add(image_id)
        left_nulls = {key: value is None for key, value in left.items() if key.endswith("_version")}
        right_nulls = {
            key: value is None for key, value in right.items() if key.endswith("_version")
        }
        if left_nulls != right_nulls:
            differences["version_null_pattern"].add(image_id)
        if left["status"] == "SEGMENTATION":
            baseline_full_path_latency.append(float(left["processing_time_ms"]))
        if right["status"] == "SEGMENTATION":
            candidate_full_path_latency.append(float(right["processing_time_ms"]))

        left_items = left["segmentations"]
        right_items = right["segmentations"]
        if len(left_items) != len(right_items):
            differences["segmentation_count"].add(image_id)
            continue
        for left_item, right_item in zip(left_items, right_items, strict=True):
            if left_item["bbox"] != right_item["bbox"]:
                differences["bbox"].add(image_id)
            if (left_item["status"], left_item["reason_codes"]) != (
                right_item["status"],
                right_item["reason_codes"],
            ):
                differences["item_status"].add(image_id)
            if left_item["prediction"] != right_item["prediction"]:
                differences["prediction"].add(image_id)
            left_top3 = [(item["class_id"], item["class_name"]) for item in left_item["top3"]]
            right_top3 = [(item["class_id"], item["class_name"]) for item in right_item["top3"]]
            if left_top3 != right_top3:
                differences["top3"].add(image_id)
            deltas = [
                abs(float(left_item["confidence"]) - float(right_item["confidence"])),
                *(
                    abs(float(left_value["confidence"]) - float(right_value["confidence"]))
                    for left_value, right_value in zip(left_item["top3"], right_item["top3"])
                ),
            ]
            maximum_confidence_delta = max(maximum_confidence_delta, *deltas)
            if any(delta != 0.0 for delta in deltas):
                differences["confidence"].add(image_id)

    component_counts = {key: len(value) for key, value in differences.items()}
    baseline_latency = _latency(baseline_full_path_latency)
    candidate_latency = _latency(candidate_full_path_latency)
    latency_delta = {
        key: (
            None
            if baseline_latency[key] is None or candidate_latency[key] is None
            else round(float(candidate_latency[key]) - float(baseline_latency[key]), 3)
        )
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
    }
    return {
        "schema_version": "1.0",
        "comparison": "packaged_worker_semantic_response_diff",
        "baseline": {
            "path": baseline_path.resolve().as_posix(),
            "sha256": sha256_file(baseline_path),
        },
        "candidate": {
            "path": candidate_path.resolve().as_posix(),
            "sha256": sha256_file(candidate_path),
        },
        "image_count": len(baseline),
        "semantic_diff_count": len(set().union(*differences.values())),
        "component_diff_counts": component_counts,
        "component_diff_image_ids": {
            key: sorted(value) for key, value in differences.items() if value
        },
        "maximum_confidence_delta": maximum_confidence_delta,
        "full_path_processing_time_ms": {
            "baseline": baseline_latency,
            "candidate": candidate_latency,
            "delta": latency_delta,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two packaged Worker response traces")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_packaged_responses(args.baseline, args.candidate)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
