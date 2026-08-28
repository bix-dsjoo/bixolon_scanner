from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), probability * 100.0)), 3)


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean_ms": None if not values else float(np.mean(values)),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _request(
    url: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], float]:
    headers = {} if content_type is None else {"Content-Type": content_type}
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status, payload, (time.perf_counter_ns() - started) / 1_000_000.0
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read())
        return exc.code, payload, (time.perf_counter_ns() - started) / 1_000_000.0


def _multipart_image(content: bytes, filename: str, media_type: str) -> tuple[bytes, str]:
    boundary = f"bixolon-{uuid.uuid4().hex}"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("packaged HTTP evaluation manifest is empty or invalid")
    return records


def _version_contract(body: dict[str, Any], expected_version: str) -> bool:
    versions = [
        value for key, value in body.items() if key.endswith("_version") and value is not None
    ]
    return bool(versions) and all(value == expected_version for value in versions)


def _public_response_contract(body: dict[str, Any]) -> bool:
    status = body.get("status")
    if status == "IMAGE_RECAPTURE":
        return body.get("reason_codes") == ["IMAGE_RECAPTURE_REQUIRED"] and not body.get(
            "segmentations"
        )
    if status != "SEGMENTATION" or not body.get("segmentations"):
        return False
    for segmentation in body["segmentations"]:
        if segmentation.get("status") == "SEGMENT_RECAPTURE" and segmentation.get(
            "reason_codes"
        ) != ["SEGMENT_RECAPTURE_REQUIRED"]:
            return False
    return True


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = _records(args.manifest)
    if len(records) != args.expected_image_count:
        raise ValueError("packaged HTTP evaluation image count does not match the lock")
    executable = args.executable.resolve()
    worker_artifact = args.worker_artifact.resolve()
    executable.relative_to(worker_artifact)
    port = _free_local_port()
    environment = os.environ.copy()
    environment.update(
        {
            "BIXOLON_PACKAGE_DIR": str(args.runtime.resolve()),
            "BIXOLON_CATALOG_DIR": str(args.catalog.resolve()),
            "BIXOLON_CATALOG_STORE_ID": args.store_id,
            "BIXOLON_PROVIDER": args.provider,
            "BIXOLON_HOST": "127.0.0.1",
            "BIXOLON_PORT": str(port),
            "BIXOLON_REQUEST_TIMEOUT_SECONDS": "60",
            "BIXOLON_CPU_DETECTOR_WORKERS": "1",
            "BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS": "0",
            "BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS": "0",
        }
    )
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(executable)],
        cwd=executable.parent,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    base_url = f"http://127.0.0.1:{port}"
    http_latencies: list[float] = []
    worker_latencies: list[float] = []
    status_counts: dict[str, int] = {}
    segmentation_status_counts: dict[str, int] = {}
    error_count = 0
    response_contract_error_count = 0
    trace_rows: list[dict[str, Any]] = []
    try:
        deadline = time.monotonic() + args.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("packaged Worker exited before readiness")
            try:
                ready_status, ready, _ = _request(f"{base_url}/health/ready")
                if ready_status == 200 and ready.get("status") == "ready":
                    if ready.get("provider") != args.provider or not _version_contract(
                        ready, args.expected_version
                    ):
                        raise RuntimeError("packaged Worker readiness contract mismatch")
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError("packaged Worker readiness timed out")

        warmup_record = next(record for record in records if record.get("annotations"))
        warmup_path = args.dataset_root / warmup_record["image_path"]
        warmup_type = "image/png" if warmup_path.suffix.lower() == ".png" else "image/jpeg"
        warmup_body, warmup_content_type = _multipart_image(
            warmup_path.read_bytes(), warmup_path.name, warmup_type
        )
        for _ in range(args.warmup_count):
            warmup_status, warmup, _ = _request(
                f"{base_url}/v1/scan",
                body=warmup_body,
                content_type=warmup_content_type,
                timeout=args.request_timeout_seconds,
            )
            if warmup_status != 200 or warmup.get("status") != "SEGMENTATION":
                raise RuntimeError("packaged Worker full-path warmup failed")

        for record in records:
            image_path = args.dataset_root / record["image_path"]
            if sha256_file(image_path) != record["image_sha256"]:
                raise ValueError("packaged HTTP input checksum mismatch")
            media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            body, content_type = _multipart_image(
                image_path.read_bytes(), image_path.name, media_type
            )
            http_status, response, elapsed_ms = _request(
                f"{base_url}/v1/scan",
                body=body,
                content_type=content_type,
                timeout=args.request_timeout_seconds,
            )
            trace_rows.append(
                {
                    "image_id": record.get("image_id"),
                    "image_sha256": record["image_sha256"],
                    "http_status": http_status,
                    "client_elapsed_ms": elapsed_ms,
                    "response": response,
                }
            )
            status = str(response.get("status"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if (
                http_status != 200
                or status == "ERROR"
                or not _version_contract(response, args.expected_version)
            ):
                error_count += 1
                continue
            if not _public_response_contract(response):
                response_contract_error_count += 1
            if status == "SEGMENTATION":
                http_latencies.append(elapsed_ms)
                worker_latencies.append(float(response["processing_time_ms"]))
                for segmentation in response["segmentations"]:
                    item_status = str(segmentation["status"])
                    segmentation_status_counts[item_status] = (
                        segmentation_status_counts.get(item_status, 0) + 1
                    )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    http_summary = latency_summary(http_latencies)
    worker_summary = latency_summary(worker_latencies)
    passes = (
        error_count == 0
        and response_contract_error_count == 0
        and len(http_latencies) == args.expected_full_path_count
        and http_summary["mean_ms"] is not None
        and http_summary["p95_ms"] is not None
        and http_summary["mean_ms"] <= args.maximum_mean_ms
        and http_summary["p95_ms"] <= args.maximum_p95_ms
    )
    trace_evidence = None
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace_rows),
            encoding="utf-8",
        )
        trace_evidence = {
            "path": args.trace_output.resolve().as_posix(),
            "sha256": sha256_file(args.trace_output),
            "row_count": len(trace_rows),
        }
    report = {
        "schema_version": "1.0",
        "evaluation": "scanner_0_1_3_packaged_worker_full_valid_http",
        "product_version": args.expected_version,
        "provider": args.provider,
        "dataset": {
            "manifest_sha256": sha256_file(args.manifest),
            "image_count": len(records),
            "held_out_test_set": False,
        },
        "worker_artifact": {
            "content_manifest_sha256": directory_content_manifest(worker_artifact)[
                "manifest_sha256"
            ],
            "executable_sha256": sha256_file(executable),
        },
        "counts": {
            "error_count": error_count,
            "response_contract_error_count": response_contract_error_count,
            "status_counts": dict(sorted(status_counts.items())),
            "segmentation_status_counts": dict(sorted(segmentation_status_counts.items())),
        },
        "performance": {
            "scope": "multipart HTTP round trip; IMAGE_RECAPTURE early exits excluded",
            "warmup_count": args.warmup_count,
            "http_round_trip": http_summary,
            "worker_processing_time": worker_summary,
        },
        "targets": {
            "maximum_mean_ms": args.maximum_mean_ms,
            "maximum_p95_ms": args.maximum_p95_ms,
            "expected_full_path_count": args.expected_full_path_count,
        },
        "passes": passes,
        "trace": trace_evidence,
        "privacy": {"image_paths_recorded": False, "image_bytes_recorded": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passes:
        raise RuntimeError("packaged Worker HTTP evaluation did not meet the target")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the packaged Worker over full-path HTTP")
    parser.add_argument("--worker-artifact", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--provider", choices=("cpu", "openvino"), default="openvino")
    parser.add_argument("--expected-version", default="0.1.7")
    parser.add_argument("--expected-image-count", type=int, default=415)
    parser.add_argument("--expected-full-path-count", type=int, default=411)
    parser.add_argument("--warmup-count", type=int, default=10)
    parser.add_argument("--maximum-mean-ms", type=float, default=300.0)
    parser.add_argument("--maximum-p95-ms", type=float, default=300.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
