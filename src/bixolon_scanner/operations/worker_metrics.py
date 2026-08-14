from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _latency(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def aggregate_worker_logs(paths: list[Path]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    segment_statuses: Counter[str] = Counter()
    versions: Counter[tuple[str, str, str | None]] = Counter()
    latencies: list[float] = []
    error_count = 0
    completed_count = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("level") == "ERROR":
                error_count += 1
            if row.get("message") != "scan_request_complete":
                continue
            completed_count += 1
            statuses[str(row["status"])] += 1
            segment_statuses["APPROVED"] += int(row.get("approved_count", 0))
            segment_statuses["UNKNOWN"] += int(row.get("unknown_count", 0))
            segment_statuses["SEGMENT_RECAPTURE"] += int(row.get("segment_recapture_count", 0))
            latencies.append(float(row["processing_time_ms"]))
            versions[
                (
                    str(row["worker_version"]),
                    str(row["detector_version"]),
                    None
                    if row.get("classifier_version") is None
                    else str(row["classifier_version"]),
                )
            ] += 1
    total_segments = sum(segment_statuses.values())
    return {
        "schema_version": "1.0",
        "evaluation": "worker_operational_metrics",
        "request_count": completed_count,
        "error_log_count": error_count,
        "image_status_counts": dict(sorted(statuses.items())),
        "segment_status_counts": dict(sorted(segment_statuses.items())),
        "recognition_proxy": {
            "definition": "APPROVED segments / all returned segments; not ground-truth accuracy",
            "approved_rate": (
                segment_statuses["APPROVED"] / total_segments if total_segments else None
            ),
        },
        "latency_ms": _latency(latencies),
        "version_compositions": [
            {
                "worker_version": key[0],
                "detector_version": key[1],
                "classifier_version": key[2],
                "request_count": count,
            }
            for key, count in sorted(
                versions.items(), key=lambda item: tuple(str(v) for v in item[0])
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate structured Worker operational metrics")
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_worker_logs(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
