"""Repeat locked-source Worker measurements, compare providers, and record review."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

from ...configuration import load_json_config
from ...contracts.catalog import sha256_file
from ...evaluation.bread_runtime_parity import compare_runtime_traces
from ...evaluation.limited_source_review import record_review
from ...training.limited_source import read_jsonl, verify_sources, write_json


def summarize_repetitions(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("at least one measurement is required")
    first = reports[0]
    if any(
        row["counts"] != first["counts"]
        or row["dataset"] != first["dataset"]
        or row["artifacts"] != first["artifacts"]
        or row["environment"] != first["environment"]
        for row in reports
    ):
        raise ValueError("repetitions changed decisions, dataset, artifacts, or environment")
    latencies = [row["performance"]["full_path"]["p95_ms"] for row in reports]
    available = [value for value in latencies if value is not None]
    median = statistics.median(available) if available else None
    representative = (
        min(range(len(reports)), key=lambda i: abs(latencies[i] - median)) if available else 0
    )
    return {
        "repetitions": len(reports),
        "unique_original_images": first["dataset"]["image_count"],
        "full_path_measurement_count": sum(
            r["performance"]["full_path"]["sample_count"] for r in reports
        ),
        "p95_ms_per_run": latencies,
        "median_run_p95_ms": median,
        "representative_index": representative,
        "counts": first["counts"],
        "independent_validation": False,
    }


def run_diagnostics(
    work: Path,
    *,
    repetitions: int,
    cpu_threads: int,
    cuda_dll_dir: Path,
    history_directory: Path | None = None,
) -> dict:
    if repetitions < 1 or cpu_threads < 1:
        raise ValueError("repetitions and CPU threads must be positive")
    source = verify_sources(work / "sources")
    output = work / f"diagnostics-cpu{cpu_threads}"
    if output.exists():
        raise FileExistsError("diagnostics already exist; preserve the previous measurements")
    output.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    selected = {}
    summary = {"source_manifest_sha256": source["source_manifest_sha256"], "profiles": {}}
    for provider in ("cpu", "cuda"):
        reports, paths, traces = [], [], []
        for index in range(repetitions):
            stem = output / f"{provider}-{index + 1:02d}"
            report_path, trace_path = stem.with_suffix(".json"), stem.with_suffix(".jsonl")
            argv = [
                sys.executable,
                "-m",
                "bixolon_scanner.evaluation.scanner_v2",
                "--runtime",
                str(work / "candidate/runtime"),
                "--catalog",
                str(work / "candidate/catalog"),
                "--store-id",
                "limited220",
                "--dataset-root",
                source["dataset_root"],
                "--manifest",
                str(work / "sources/detector.jsonl"),
                "--provider",
                provider,
                "--cpu-detector-threads",
                str(cpu_threads),
                "--cpu-embedder-threads",
                str(cpu_threads),
                "--cuda-dll-dir",
                str(cuda_dll_dir),
                "--expected-image-count",
                str(source["multi_count"]),
                "--evidence-role",
                "stress_regression",
                "--output",
                str(report_path),
                "--trace-output",
                str(trace_path),
            ]
            with stem.with_suffix(".log").open("w", encoding="utf-8") as stream:
                subprocess.run(
                    argv,
                    cwd=repository,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report = load_json_config(report_path)
            if report["dataset"]["manifest_sha256"] != source["detector_manifest_sha256"]:
                raise ValueError("measurement used different originals")
            if sha256_file(trace_path) != report["trace"]["sha256"]:
                raise ValueError("measurement trace changed")
            reports.append(report)
            paths.append(report_path)
            traces.append(trace_path)
            print(f"completed: {provider} repetition {index + 1}/{repetitions}", flush=True)
        profile = summarize_repetitions(reports)
        selected_index = profile["representative_index"]
        profile["representative_report"] = str(paths[selected_index])
        profile["review"] = record_review(work, paths[selected_index], history_directory)
        summary["profiles"][provider] = profile
        selected[provider] = [
            {**row["decision"], "image_id": row["image_id"]}
            for row in read_jsonl(traces[selected_index])
        ]
    summary["parity"] = compare_runtime_traces(
        selected["cpu"], selected["cuda"], minimum_bbox_iou=0.999, maximum_confidence_error=0.005
    )
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cuda-dll-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path)
    args = parser.parse_args()
    print(
        run_diagnostics(
            args.work_dir.resolve(),
            repetitions=args.repetitions,
            cpu_threads=args.cpu_threads,
            cuda_dll_dir=args.cuda_dll_dir.resolve(),
            history_directory=args.history_dir,
        )
    )


if __name__ == "__main__":
    main()
