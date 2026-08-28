from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.catalog import sha256_file
from ..pipeline.ports import Detection
from ..runtime.onnx import box_iou


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL evidence: {path}")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latency(values: list[float]) -> dict[str, float | int | None]:
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
        "mean_ms": round(float(array.mean()), 3),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "p99_ms": round(float(np.percentile(array, 99)), 3),
    }


def _box(values: list[float] | dict[str, float]) -> Detection:
    if isinstance(values, dict):
        x, y = float(values["x"]), float(values["y"])
        width, height = float(values["width"]), float(values["height"])
    else:
        x, y, width, height = (float(value) for value in values)
    return Detection(x, y, x + width, y + height, 1.0)


def _match(
    predictions: list[Detection], targets: list[Detection], threshold: float
) -> dict[int, int]:
    candidates = sorted(
        (
            (box_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
        ),
        reverse=True,
    )
    matches: dict[int, int] = {}
    used_targets: set[int] = set()
    for overlap, prediction_index, target_index in candidates:
        if overlap < threshold:
            break
        if prediction_index not in matches and target_index not in used_targets:
            matches[prediction_index] = target_index
            used_targets.add(target_index)
    return matches


@dataclass
class Outcome:
    image_count: int = 0
    segmentation_image_count: int = 0
    image_recapture_count: int = 0
    error_count: int = 0
    ground_truth_count: int = 0
    recaptured_ground_truth_count: int = 0
    prediction_count: int = 0
    matched_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0
    approved_count: int = 0
    approved_wrong_count: int = 0
    unknown_count: int = 0
    unknown_candidate_out_count: int = 0
    segment_recapture_count: int = 0
    client_ms: list[float] = field(default_factory=list)
    worker_ms: list[float] = field(default_factory=list)
    segmentation_worker_ms: list[float] = field(default_factory=list)
    image_recapture_worker_ms: list[float] = field(default_factory=list)

    def add(self, record: dict[str, Any], trace: dict[str, Any], threshold: float) -> None:
        response = trace["response"]
        annotations = record.get("annotations", [])
        self.image_count += 1
        self.ground_truth_count += len(annotations)
        self.client_ms.append(float(trace["client_elapsed_ms"]))
        if response.get("processing_time_ms") is not None:
            processing_ms = float(response["processing_time_ms"])
            self.worker_ms.append(processing_ms)
        else:
            processing_ms = None

        status = response.get("status")
        if status == "IMAGE_RECAPTURE":
            self.image_recapture_count += 1
            self.recaptured_ground_truth_count += len(annotations)
            if processing_ms is not None:
                self.image_recapture_worker_ms.append(processing_ms)
            return
        if status != "SEGMENTATION":
            self.error_count += 1
            return

        self.segmentation_image_count += 1
        if processing_ms is not None:
            self.segmentation_worker_ms.append(processing_ms)
        segmentations = response["segmentations"]
        predictions = [_box(segmentation["bbox"]) for segmentation in segmentations]
        targets = [
            _box(annotation.get("bbox_xywh", annotation.get("bbox"))) for annotation in annotations
        ]
        matches = _match(predictions, targets, threshold)
        self.prediction_count += len(predictions)
        self.matched_count += len(matches)
        self.false_positive_count += len(predictions) - len(matches)
        self.false_negative_count += len(targets) - len(matches)

        for prediction_index, segmentation in enumerate(segmentations):
            item_status = segmentation["status"]
            if item_status == "APPROVED":
                self.approved_count += 1
            elif item_status == "UNKNOWN":
                self.unknown_count += 1
            elif item_status == "SEGMENT_RECAPTURE":
                self.segment_recapture_count += 1
            else:
                raise ValueError(f"unsupported segmentation status: {item_status}")

            target_index = matches.get(prediction_index)
            if target_index is None:
                continue
            expected = f"bread_{int(annotations[target_index]['category_id']):02d}"
            if item_status == "APPROVED":
                prediction = segmentation.get("prediction")
                self.approved_wrong_count += int(
                    prediction is None or prediction.get("class_id") != expected
                )
            elif item_status == "UNKNOWN":
                candidates = {candidate["class_id"] for candidate in segmentation["top3"]}
                self.unknown_candidate_out_count += int(expected not in candidates)

    def render(self) -> dict[str, Any]:
        return {
            "counts": {key: value for key, value in vars(self).items() if not key.endswith("_ms")},
            "rates": {
                "segmentation_image_rate": _rate(self.segmentation_image_count, self.image_count),
                "image_recapture_rate": _rate(self.image_recapture_count, self.image_count),
                "approved_output_rate": _rate(self.approved_count, self.prediction_count),
                "unknown_output_rate": _rate(self.unknown_count, self.prediction_count),
                "segment_recapture_output_rate": _rate(
                    self.segment_recapture_count, self.prediction_count
                ),
                "false_positive_per_ground_truth": _rate(
                    self.false_positive_count, self.ground_truth_count
                ),
                "false_negative_per_ground_truth": _rate(
                    self.false_negative_count, self.ground_truth_count
                ),
                "approved_wrong_per_approved": _rate(
                    self.approved_wrong_count, self.approved_count
                ),
                "unknown_candidate_out_per_unknown": _rate(
                    self.unknown_candidate_out_count, self.unknown_count
                ),
            },
            "performance": {
                "worker_processing_all": _latency(self.worker_ms),
                "worker_processing_segmentation": _latency(self.segmentation_worker_ms),
                "worker_processing_image_recapture": _latency(self.image_recapture_worker_ms),
                "multipart_http_round_trip_all": _latency(self.client_ms),
            },
        }


def _rollup_name(group: str) -> str:
    if group.startswith("multi_object_scenes"):
        return "multi_object_scenes"
    if group.startswith("operational_collections/"):
        return "operational_collections"
    raise ValueError(f"unsupported evaluation group: {group}")


def summarize(
    pairs: list[tuple[str, Path, Path, Path]], *, match_iou_threshold: float = 0.5
) -> dict[str, Any]:
    outcomes: defaultdict[str, Outcome] = defaultdict(Outcome)
    providers: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for default_group, manifest_path, trace_path, report_path in pairs:
        records = _jsonl(manifest_path)
        traces = _jsonl(trace_path)
        trace_by_id = {int(row["image_id"]): row for row in traces}
        if len(trace_by_id) != len(traces) or len(records) != len(traces):
            raise ValueError("manifest and trace cardinality differ")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        providers.append(report["provider"])
        evidence.append(
            {
                "manifest_sha256": sha256_file(manifest_path),
                "trace_sha256": sha256_file(trace_path),
                "http_report_sha256": sha256_file(report_path),
                "image_count": len(records),
            }
        )
        for record in records:
            image_id = int(record["image_id"])
            trace = trace_by_id.get(image_id)
            if trace is None or trace.get("image_sha256") != record.get("image_sha256"):
                raise ValueError("manifest and trace identity differ")
            group = str(record.get("evaluation_set", default_group))
            rollup = _rollup_name(group)
            for name in dict.fromkeys((group, rollup, "all")):
                outcomes[name].add(record, trace, match_iou_threshold)

    if not providers or any(provider != providers[0] for provider in providers[1:]):
        raise ValueError("profile provider contracts differ across evidence pairs")
    return {
        "schema_version": "1.0",
        "evaluation": "packaged_worker_current_pc_profile_summary",
        "product_version": "0.1.8",
        "provider": providers[0],
        "match_iou_threshold": match_iou_threshold,
        "metric_contract": {
            "status_rate_denominator": "all_images",
            "segmentation_status_rate_denominator": "all_output_segmentations",
            "fp_fn_scope": "SEGMENTATION images only; IMAGE_RECAPTURE ground truth is reported separately",
            "approved_wrong_scope": "IoU-matched APPROVED outputs",
            "candidate_out_scope": "IoU-matched UNKNOWN outputs whose Top-3 omits ground truth",
            "primary_speed": "worker processing_time_ms including all images; warmups excluded",
        },
        "datasets": {name: outcome.render() for name, outcome in sorted(outcomes.items())},
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize packaged Worker profile traces")
    parser.add_argument(
        "--pair",
        nargs=4,
        action="append",
        metavar=("DEFAULT_GROUP", "MANIFEST", "TRACE", "HTTP_REPORT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    pairs = [
        (label, Path(manifest), Path(trace), Path(report))
        for label, manifest, trace, report in args.pair
    ]
    result = summarize(pairs, match_iou_threshold=args.match_iou_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
