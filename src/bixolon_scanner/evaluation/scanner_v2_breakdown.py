from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.catalog import sha256_file

_ITEM_STATUSES = ("APPROVED", "UNKNOWN", "SEGMENT_RECAPTURE")
_DIFFICULTY_ORDER = {"EASY": 0, "MEDIUM": 1, "HARD": 2}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in: {path}")
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _latency(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "sample_count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _new_bucket() -> dict[str, Any]:
    return {"counts": Counter(), "latencies_ms": []}


def _section(bucket: dict[str, Any]) -> dict[str, Any]:
    counts: Counter = bucket["counts"]
    segmentation_outputs = sum(counts[f"status_{status}"] for status in _ITEM_STATUSES)
    segmentation_images = counts["segmentation_image_count"]
    gt = counts["ground_truth_count"]
    approved = counts["status_APPROVED"]
    unknown = counts["status_UNKNOWN"]
    segment_recapture = counts["status_SEGMENT_RECAPTURE"]
    wrong_approved = counts["approved_misrecognition_count"]
    candidate_out = counts["unknown_candidate_out_count"]
    return {
        "counts": {
            "image_count": counts["image_count"],
            "segmentation_image_count": segmentation_images,
            "image_recapture_count": counts["image_recapture_count"],
            "ground_truth_count": gt,
            "segmentation_output_count": segmentation_outputs,
            "matched_count": counts["matched_count"],
            "false_negative_count": counts["false_negative_count"],
            "false_positive_count": counts["false_positive_count"],
            "false_negative_image_count": counts["false_negative_image_count"],
            "false_positive_image_count": counts["false_positive_image_count"],
            "approved_count": approved,
            "unknown_top3_count": unknown,
            "segment_recapture_count": segment_recapture,
            "matched_approved_count": counts["matched_status_APPROVED"],
            "approved_misrecognition_count": wrong_approved,
            "unknown_candidate_out_count": candidate_out,
        },
        "metrics": {
            "segmentation_rate": _rate(segmentation_images, counts["image_count"]),
            "image_recapture_rate": _rate(counts["image_recapture_count"], counts["image_count"]),
            "approved_over_segmentation_rate": _rate(approved, segmentation_outputs),
            "unknown_top3_over_segmentation_rate": _rate(unknown, segmentation_outputs),
            "segment_recapture_over_segmentation_rate": _rate(
                segment_recapture, segmentation_outputs
            ),
            "segmentation_image_false_negative_rate": _rate(
                counts["false_negative_image_count"], segmentation_images
            ),
            "segmentation_image_false_positive_rate": _rate(
                counts["false_positive_image_count"], segmentation_images
            ),
            "approved_object_misrecognition_rate_over_all_gt": _rate(wrong_approved, gt),
            "approved_output_misrecognition_rate": _rate(wrong_approved, approved),
            "correct_approved_coverage_over_all_gt": _rate(
                counts["matched_status_APPROVED"] - wrong_approved, gt
            ),
            "unknown_top3_candidate_out_rate_over_all_gt": _rate(candidate_out, gt),
            "unknown_top3_candidate_out_rate_over_unknown": _rate(candidate_out, unknown),
        },
        "performance": _latency(bucket["latencies_ms"]),
    }


def _add_trace(bucket: dict[str, Any], trace: dict[str, Any]) -> None:
    counts: Counter = bucket["counts"]
    counts["image_count"] += 1
    counts["ground_truth_count"] += int(trace["ground_truth_count"])
    bucket["latencies_ms"].append(float(trace["latency_ms"]))
    status = str(trace["status"])
    if status == "IMAGE_RECAPTURE":
        counts["image_recapture_count"] += 1
        if trace.get("decision", {}).get("segmentations"):
            raise ValueError("IMAGE_RECAPTURE trace contains segmentations")
        return
    if status != "SEGMENTATION":
        raise ValueError(f"unsupported trace status: {status}")
    counts["segmentation_image_count"] += 1
    segmentations = trace.get("decision", {}).get("segmentations")
    if not isinstance(segmentations, list) or not segmentations:
        raise ValueError("SEGMENTATION trace must contain segmentation outputs")
    for segmentation in segmentations:
        item_status = str(segmentation.get("status"))
        if item_status not in _ITEM_STATUSES:
            raise ValueError(f"unsupported segmentation status: {item_status}")
        counts[f"status_{item_status}"] += 1
    if int(trace.get("prediction_count", -1)) != len(segmentations):
        raise ValueError("trace prediction count differs from segmentation outputs")
    counts["matched_count"] += int(trace["matched_count"])
    counts["false_negative_count"] += int(trace["false_negative_count"])
    counts["false_positive_count"] += int(trace["false_positive_count"])
    counts["false_negative_image_count"] += int(trace["false_negative_count"]) > 0
    counts["false_positive_image_count"] += int(trace["false_positive_count"]) > 0
    for diagnostic in trace.get("matched_classifier_diagnostics", []):
        final_status = str(diagnostic["final_status"])
        counts[f"matched_status_{final_status}"] += 1
        if final_status == "APPROVED" and not diagnostic["classifier_top1_correct"]:
            counts["approved_misrecognition_count"] += 1
        if final_status == "UNKNOWN" and not diagnostic["classifier_top3_hit"]:
            counts["unknown_candidate_out_count"] += 1


def build_difficulty_breakdown(
    *,
    development_report_path: Path,
    trace_path: Path,
    manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    development = _read_json(development_report_path)
    if (
        development.get("evaluation") != "scanner_2_0_development_300"
        or development.get("promotion_evidence") is not False
    ):
        raise ValueError("difficulty breakdown requires a Scanner 2.0 development report")
    if sha256_file(trace_path) != development.get("trace", {}).get("sha256"):
        raise ValueError("trace differs from the locked development report")
    if sha256_file(manifest_path) != development.get("dataset", {}).get("manifest_sha256"):
        raise ValueError("manifest differs from the locked development report")

    manifest = _read_jsonl(manifest_path)
    traces = _read_jsonl(trace_path)
    manifest_by_id: dict[int, dict[str, Any]] = {}
    for row in manifest:
        image_id = int(row["image_id"])
        if image_id in manifest_by_id:
            raise ValueError("manifest contains duplicate image IDs")
        difficulty = str(row.get("difficulty", "")).upper()
        if not difficulty:
            raise ValueError("manifest image is missing difficulty")
        manifest_by_id[image_id] = row
    trace_by_id = {int(row["image_id"]): row for row in traces}
    if len(trace_by_id) != len(traces):
        raise ValueError("trace contains duplicate image IDs")
    if set(trace_by_id) != set(manifest_by_id):
        raise ValueError("trace and manifest image IDs differ")

    overall = _new_bucket()
    by_difficulty: dict[str, dict[str, Any]] = {}
    for image_id, trace in trace_by_id.items():
        difficulty = str(manifest_by_id[image_id]["difficulty"]).upper()
        bucket = by_difficulty.setdefault(difficulty, _new_bucket())
        _add_trace(overall, trace)
        _add_trace(bucket, trace)

    overall_section = _section(overall)
    locked_counts = development.get("counts", {})
    comparisons = {
        "image_count": overall_section["counts"]["image_count"],
        "segmentation_image_count": overall_section["counts"]["segmentation_image_count"],
        "image_recapture_count": overall_section["counts"]["image_recapture_count"],
        "ground_truth_count": overall_section["counts"]["ground_truth_count"],
        "prediction_count": overall_section["counts"]["segmentation_output_count"],
        "matched_count": overall_section["counts"]["matched_count"],
        "false_negative_count": overall_section["counts"]["false_negative_count"],
        "false_positive_count": overall_section["counts"]["false_positive_count"],
        "false_negative_image_count": overall_section["counts"]["false_negative_image_count"],
        "false_positive_image_count": overall_section["counts"]["false_positive_image_count"],
        "approved_count": overall_section["counts"]["matched_approved_count"],
        "approved_misrecognition_count": overall_section["counts"]["approved_misrecognition_count"],
        "unknown_count": overall["counts"]["matched_status_UNKNOWN"],
        "segment_recapture_count": overall["counts"]["matched_status_SEGMENT_RECAPTURE"],
        "unknown_candidate_out_count": overall_section["counts"]["unknown_candidate_out_count"],
    }
    mismatches = {
        name: {"locked": locked_counts.get(name), "derived": value}
        for name, value in comparisons.items()
        if locked_counts.get(name) != value
    }
    if mismatches:
        raise ValueError(f"derived counts differ from locked development report: {mismatches}")

    ordered_difficulties = sorted(
        by_difficulty,
        key=lambda value: (_DIFFICULTY_ORDER.get(value, len(_DIFFICULTY_ORDER)), value),
    )
    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_development_300_difficulty_breakdown",
        "promotion_evidence": False,
        "evidence_role": "derived_from_locked_development_trace",
        "denominators": {
            "image_status_rates": "all images in the requested difficulty scope",
            "segmentation_status_rates": (
                "all segmentations[] emitted by SEGMENTATION images in the requested scope"
            ),
            "segmentation_image_fn_fp_rates": "SEGMENTATION images in the requested scope",
            "approved_misrecognition_and_candidate_out_rates": (
                "all judgeable ground-truth objects in the requested scope"
            ),
        },
        "source": {
            "development_report_sha256": sha256_file(development_report_path),
            "trace_sha256": sha256_file(trace_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "overall": overall_section,
        "by_difficulty": {
            difficulty: _section(by_difficulty[difficulty]) for difficulty in ordered_difficulties
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive Scanner 2.0 difficulty metrics from a locked development trace"
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_difficulty_breakdown(
        development_report_path=args.development_report,
        trace_path=args.trace,
        manifest_path=args.manifest,
        output=args.output,
    )


if __name__ == "__main__":
    main()
