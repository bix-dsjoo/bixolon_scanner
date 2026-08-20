from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
from fastapi.testclient import TestClient

from ..contracts.catalog import sha256_file
from ..worker.api import create_app
from ..worker.settings import WorkerSettings


def _latency(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "maximum_ms": float(array.max()),
    }


def _records(manifest: Path, dataset_root: Path, trace: Path) -> list[dict]:
    source = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    expected = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    if len(source) != 300 or len(expected) != len(source):
        raise ValueError("reliability gate requires the locked 300-image trace")
    by_id = {int(row["image_id"]): row["decision"] for row in expected}
    records = []
    for row in source:
        image_id = int(row["image_id"])
        if image_id not in by_id:
            raise ValueError("locked trace does not match the reliability manifest")
        records.append(
            {
                "image_id": image_id,
                "bytes": (dataset_root / row["image_path"]).read_bytes(),
                "expected": by_id[image_id],
            }
        )
    return records


def _decision(body: dict) -> dict:
    return {
        key: value for key, value in body.items() if key not in {"request_id", "processing_time_ms"}
    }


def evaluate(args: argparse.Namespace) -> dict:
    signing_key = os.environ.get(args.signing_key_env, "")
    if len(signing_key.encode()) < 16:
        raise ValueError("Catalog signing key must contain at least 16 bytes")
    records = _records(args.manifest, args.dataset_root, args.expected_trace)
    settings = WorkerSettings(
        package_dir=args.runtime,
        catalog_dir=args.catalog,
        catalog_store_id=args.store_id,
        catalog_key_id=args.key_id,
        catalog_signing_key=signing_key,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    app = create_app(settings=settings)
    process = psutil.Process()
    latencies = []
    rss_samples = []
    status_counts: Counter[str] = Counter()
    mismatch_ids: set[int] = set()
    non_200_count = 0
    readiness_failure_count = 0
    started = time.perf_counter()
    with TestClient(app) as client:
        for warmup_index in range(args.warmup_count):
            row = records[warmup_index % len(records)]
            response = client.post(
                "/v1/scan",
                files={"image": ("warmup.jpg", row["bytes"], "image/jpeg")},
            )
            if response.status_code != 200:
                raise RuntimeError("reliability warm-up request failed")
        measurement_started = time.perf_counter()
        for index in range(args.request_count):
            row = records[index % len(records)]
            request_started = time.perf_counter()
            response = client.post(
                "/v1/scan",
                files={"image": ("scan.jpg", row["bytes"], "image/jpeg")},
            )
            latencies.append((time.perf_counter() - request_started) * 1000.0)
            if response.status_code != 200:
                non_200_count += 1
            else:
                body = response.json()
                status_counts[str(body.get("status"))] += 1
                if _decision(body) != row["expected"]:
                    mismatch_ids.add(row["image_id"])
            ordinal = index + 1
            if ordinal % args.sample_interval == 0:
                rss_samples.append(process.memory_info().rss)
            if ordinal % args.health_interval == 0:
                ready = client.get("/health/ready")
                if ready.status_code != 200 or ready.json().get("status") != "ready":
                    readiness_failure_count += 1
            if ordinal % args.progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "completed": ordinal,
                            "mean_ms": float(np.mean(latencies[-args.progress_interval :])),
                            "rss_mib": process.memory_info().rss / (1024 * 1024),
                            "mismatch_image_count": len(mismatch_ids),
                            "non_200_count": non_200_count,
                        }
                    ),
                    flush=True,
                )
        measurement_seconds = time.perf_counter() - measurement_started
    total_seconds = time.perf_counter() - started
    window = min(args.rss_window_samples, len(rss_samples) // 2)
    if window < 1:
        raise ValueError("reliability gate did not collect enough RSS samples")
    first_rss_p95 = float(np.percentile(rss_samples[:window], 95))
    last_rss_p95 = float(np.percentile(rss_samples[-window:], 95))
    rss_growth = (last_rss_p95 - first_rss_p95) / first_rss_p95
    passes = (
        len(latencies) == args.request_count
        and non_200_count == 0
        and not mismatch_ids
        and readiness_failure_count == 0
        and rss_growth <= args.maximum_rss_growth
    )
    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_accelerated_reliability_gate",
        "evidence_role": "pre_private_operational_gate",
        "runtime_metadata_sha256": sha256_file(args.runtime / "metadata.json"),
        "catalog_metadata_sha256": sha256_file(args.catalog / "catalog.json"),
        "manifest_sha256": sha256_file(args.manifest),
        "expected_trace_sha256": sha256_file(args.expected_trace),
        "provider": args.provider,
        "request_count": args.request_count,
        "warmup_count": args.warmup_count,
        "measurement_seconds": measurement_seconds,
        "total_seconds": total_seconds,
        "latency": _latency(latencies),
        "status_counts": dict(sorted(status_counts.items())),
        "non_200_count": non_200_count,
        "decision_mismatch_image_count": len(mismatch_ids),
        "decision_mismatch_image_ids": sorted(mismatch_ids),
        "readiness_failure_count": readiness_failure_count,
        "rss": {
            "sample_interval_requests": args.sample_interval,
            "sample_count": len(rss_samples),
            "window_sample_count": window,
            "first_window_p95_bytes": first_rss_p95,
            "last_window_p95_bytes": last_rss_p95,
            "peak_bytes": max(rss_samples),
            "growth_rate": rss_growth,
            "maximum_growth_rate": args.maximum_rss_growth,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "passes": passes,
        "limitations": {
            "accelerated_sequential_gate_is_not_wall_clock_shadow_soak": True,
            "owner_private_accuracy_evidence": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passes:
        raise RuntimeError("accelerated reliability gate failed")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Scanner 2.0 reliability gate")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--request-count", type=int, default=10_000)
    parser.add_argument("--warmup-count", type=int, default=300)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--health-interval", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--rss-window-samples", type=int, default=10)
    parser.add_argument("--maximum-rss-growth", type=float, default=0.05)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
