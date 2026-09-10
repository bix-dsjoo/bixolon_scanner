"""Evaluate fixed global proposal settings; classifier policy stays unchanged."""

import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2
from bixolon_scanner.evaluation.three_bakery_http import measure
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    config = load_json_config(Path("configs/experiments/bread/n100_020.json"))
    work = Path(config["work"])
    source = Path(
        "artifacts/retraining/three-bakery-improvement-0.1.18/candidates/dfine640-dense-detector"
    )
    candidate = work / "candidates/dfine-dense-sensitive"
    policy = {
        "score_threshold": 0.025,
        "uncertainty_score_threshold": 0.0125,
        "nms_iou_threshold": 0.5,
        "nms_containment_threshold": None,
    }
    if not candidate.exists():
        shutil.copytree(source, candidate)
        metadata = load_json_config(candidate / "runtime/metadata.json")
        metadata["detector"].update(policy)
        write_json(candidate / "runtime/metadata.json", metadata)
        write_json(
            candidate / "proposal-policy.json",
            {
                "source_metadata_sha256": sha256_file(source / "runtime/metadata.json"),
                "policy": policy,
                "role": "log132 exposed development comparison; no learned weight changes",
            },
        )
    load_runtime_package_v2(candidate / "runtime")
    report = measure(
        candidate,
        read_jsonl(Path(config["log_manifest"])),
        work / "measurements/dfine-dense-sensitive",
        python=Path("artifacts/build-envs/worker-cpu-0.1.17-py311/Scripts/python.exe"),
        provider="cpu",
        cpu_profile=(8, 12),
        warmup=10,
        repetitions=1,
        maximum_p95_ms=200,
    )
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
