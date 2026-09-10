"""Interleaved HTTP confirmation of CPU thread-pool sleep, including external Worker activity."""

from __future__ import annotations

import argparse
from pathlib import Path

import psutil

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..training.three_bakery_data import read_jsonl, write_json
from .three_bakery_http import measure, provider_parity


def external_workers() -> dict:
    result = {}
    for process in psutil.process_iter(["pid", "name", "cpu_times", "create_time"]):
        if "bixolon-worker" in (process.info["name"] or "").lower():
            result[str(process.pid)] = {
                "created": process.info["create_time"],
                "cpu_seconds": sum(process.info["cpu_times"][:2]),
            }
    return result


def confirm(config_path: Path) -> dict:
    settings = load_json_config(config_path)
    config = load_json_config(Path(settings["study_config"]))
    work = Path(config["work"])
    records = read_jsonl(work / "log-inputs.jsonl")
    outputs = []
    for repetition in range(config["repetitions"]):
        order = ("baseline", "sleep") if repetition % 2 == 0 else ("sleep", "baseline")
        reports = {}
        pair = work / "cpu-interleaved" / f"repeat-{repetition + 1}"
        retry = 0
        while any(
            (pair / mode / "external-worker-activity.json").exists()
            and load_json_config(pair / mode / "external-worker-activity.json")[
                "possible_interference"
            ]
            for mode in order
        ):
            retry += 1
            pair = work / "cpu-interleaved" / f"repeat-{repetition + 1}-retry-{retry}"
        for mode in order:
            before = external_workers()
            report = measure(
                Path(config["baseline_candidate"]),
                records,
                pair / mode,
                python=Path(config["cpu_python"]),
                provider="cpu",
                warmup=config["warmup"],
                repetitions=1,
                input_identity={
                    "config_sha256": sha256_file(config_path),
                    "manifest_sha256": sha256_file(work / "log-inputs.jsonl"),
                },
                cpu_profile=tuple(config["cpu_profile"]),
                worker_executable=Path(config["baseline_worker"]) if mode == "baseline" else None,
            )
            after = external_workers()
            # A second app may remain idle; record actual CPU use rather than terminating it.
            active = before.keys() != after.keys() or any(
                after[pid]["created"] != old["created"]
                or after[pid]["cpu_seconds"] - old["cpu_seconds"] > 0.5
                for pid, old in before.items()
                if pid in after
            )
            activity = {"before": before, "after": after, "possible_interference": active}
            receipt = pair / mode / "external-worker-activity.json"
            if not receipt.exists():
                write_json(receipt, activity)
            reports[mode] = report
        ratios = {
            group: reports["sleep"]["summary"]["latency"][group]["p95_ms"]
            / reports["baseline"]["summary"]["latency"][group]["p95_ms"]
            for group in ("all", "full_path")
        }
        parity = provider_parity(pair / "baseline/responses.jsonl", pair / "sleep/responses.jsonl")
        interference = any(
            load_json_config(pair / mode / "external-worker-activity.json")["possible_interference"]
            for mode in order
        )
        outputs.append(
            {
                "repetition": repetition + 1,
                "measurement_directory": str(pair),
                "p95_ratios": ratios,
                "parity": parity,
                "possible_external_worker_interference": interference,
                "accepted": parity["status_rank_parity"]
                and not interference
                and all(
                    ratio <= 1 - settings["acceptance"]["minimum_p95_reduction_ratio"]
                    for ratio in ratios.values()
                ),
            }
        )
        print(outputs[-1], flush=True)
    result = {"repetitions": outputs, "accepted": all(r["accepted"] for r in outputs)}
    write_json(work / "cpu-interleaved/report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    confirm(args.config)


if __name__ == "__main__":
    main()
