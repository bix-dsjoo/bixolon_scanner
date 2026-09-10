"""Freeze and evaluate the 0.2.0 source-only model with exposed log132 diagnostics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ...configuration import load_json_config
from ...contracts.artifact import directory_content_manifest
from ...contracts.catalog import sha256_file
from ...evaluation.n100_objective import assess
from ...evaluation.three_bakery_http import measure
from ...operations.version_bundle import load_version_config, prepare_version_bundle
from ...training.three_bakery_data import read_jsonl, write_json


def freeze(root: Path, config_path: Path) -> dict:
    config = load_json_config(config_path)
    work = root / config["work"]
    candidate = root / config["selected_candidate"]
    output = root / "artifacts/versions/0.2.0/source"
    reports = {
        "cpu-12-8": work / "confirmation/cpu-12-8/report.json",
        "cpu-8-8": work / "confirmation/cpu-8-8/report.json",
        "cuda": work / "confirmation/cuda-8-8/report.json",
        "local-recapture": work / "measurements/dfine-dense-local-recapture/report.json",
        "selective": work / "measurements/dfine-dense-selective/report.json",
        "occlusion": work / "measurements/ssdlite-occlusion/report.json",
        "dfine-recall": work / "measurements/dfine-dense-recall/report.json",
        "dfine-sensitive": work / "measurements/dfine-dense-sensitive/report.json",
        "openvino-fallback": work / "measurements/build-host-gpu-tiny-cpu-vit/report.json",
    }
    selection = {
        "selected_candidate": str(candidate),
        "reason": "Matches every log GT with zero wrong approvals and >=95% correct approvals; prefer the simpler full normal-ROI batch because selective skip did not reduce measured CPU latency.",
        "cpu_profile": [8, 12],
        "n100_profile": config["execution"],
        "data_policy": config["data_policy"],
        "targets": config["targets"],
        "metrics": {
            name: [assess(s, config["targets"]) for s in load_json_config(path)["repetitions"]]
            for name, path in reports.items()
        },
        "report_sha256": {name: sha256_file(path) for name, path in reports.items()},
        "runtime_manifest": directory_content_manifest(candidate / "runtime"),
        "catalog_manifest": directory_content_manifest(candidate / "catalog"),
        "n100_hardware_result": None,
        "n100_limitation": "No Intel N100 target is connected. Build-host OpenVINO GPU warmup fails on the current driver; explicit CPU fallback is measured and is not GPU/N100 performance.",
    }
    for name in ["runtime", "catalog"]:
        target = output / name
        if not target.exists():
            shutil.copytree(candidate / name, target)
        if directory_content_manifest(target) != directory_content_manifest(candidate / name):
            raise ValueError("frozen source payload changed")
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    sources = {f"{name}.json": path for name, path in reports.items()}
    sources.update(
        {
            "study-config.json": config_path,
            "log-inputs.jsonl": root / config["log_manifest"],
            "detector-training.json": root
            / "artifacts/retraining/three-bakery-revised300/models/dfine-dense-20260908/report.json",
            "classifier-training.json": root
            / "artifacts/retraining/three-bakery-revised300/models/margin-dense-20260908/report.json",
            "occlusion-training.json": work / "training/ssdlite-occlusion/report.json",
        }
    )
    for name, path in sources.items():
        target = evidence / name
        if target.exists() and sha256_file(target) != sha256_file(path):
            raise ValueError("frozen evidence changed")
        shutil.copy2(path, target)
    write_json(evidence / "selection.json", selection)
    version = load_json_config(root / "configs/archive/versions/0.1.18.json")
    version.update(
        version="0.2.0",
        app_build=23,
        source_candidate="dfine-margin-dense-20260908-local-recapture",
    )
    for name in ["runtime", "catalog"]:
        version[name] = {
            "path": (output / name).relative_to(root).as_posix(),
            "manifest_sha256": directory_content_manifest(output / name)["manifest_sha256"],
        }
    version["catalog"]["store_id"] = "three_bakery"
    version["evaluation_evidence"] = [
        {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
        for p in sorted(evidence.iterdir())
        if p.is_file()
    ]
    version_path = root / "configs/versions/0.2.0.json"
    if version_path.exists() and load_json_config(version_path) != version:
        raise ValueError("frozen version configuration changed")
    write_json(version_path, version)
    write_json(work / "selection.json", selection)
    prepare_version_bundle(load_version_config(version_path), repository_root=root)
    return selection


def evaluate(root: Path, config_path: Path, profile: str, packaged: bool) -> dict:
    config = load_json_config(config_path)
    work = root / config["work"]
    version = root / "artifacts/versions/0.2.0"
    load_json_config(work / "selection.json")
    n100 = profile == "n100"
    python = (
        root
        / "artifacts/build-envs"
        / (
            {
                "cpu": "worker-cpu-0.1.17-py311",
                "cuda": "worker-cuda-py311",
                "n100": "worker-openvino-gpu-py311",
            }[profile]
        )
        / "Scripts/python.exe"
    )
    executable = (
        (
            root / "artifacts/n100/0.2.0/worker-payload/worker/bixolon-worker.exe"
            if n100
            else root / "artifacts/installers/0.2.0/windows-payload/worker/bixolon-worker.exe"
            if profile == "cpu"
            else version / "bixolon-bakery-ai-scanner-0.2.0/worker/bixolon-worker.exe"
        )
        if packaged
        else None
    )
    result = measure(
        version / "staging",
        read_jsonl(root / config["log_manifest"]),
        version / f"log132/{profile}-{'packaged' if packaged else 'source'}",
        python=python,
        provider="cuda" if profile == "cuda" else "cpu",
        cpu_profile=(2, 4) if n100 else (8, 12),
        embedder_provider="openvino_gpu" if n100 else "same",
        embedder_fallback_provider="same" if n100 else "none",
        execution_cpu_fallback=n100,
        verifier_provider="cpu" if n100 else "same",
        worker_executable=executable,
        cuda_dll_dir=version / "staging/cuda-runtime" if profile == "cuda" else None,
        repetitions=3,
        warmup=10,
        maximum_p95_ms=200,
        input_identity={
            "selection_sha256": sha256_file(work / "selection.json"),
            "role": config["data_policy"],
        },
    )
    write_json(
        version / f"log132/{profile}-{'packaged' if packaged else 'source'}/objective.json",
        {
            "repetitions": [assess(s, config["targets"]) for s in result["repetitions"]],
            "n100_hardware_measured": False,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "evaluate"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bread/n100_020.json")
    )
    parser.add_argument("--profile", choices=["cpu", "cuda", "n100"], default="cpu")
    parser.add_argument("--packaged", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    result = (
        freeze(root, args.config)
        if args.command == "freeze"
        else evaluate(root, args.config, args.profile, args.packaged)
    )
    print(result.get("summary", {"selected_candidate": result.get("selected_candidate")}))


if __name__ == "__main__":
    main()
