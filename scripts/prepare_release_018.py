"""Freeze the retained model and verified CPU runtime change for product 0.1.18."""

from __future__ import annotations

import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.three_bakery_data import write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work = root / "artifacts/retraining/three-bakery-improvement-0.1.18"
    selection = load_json_config(work / "model-selection.json")
    cpu = load_json_config(work / "cpu-interleaved/report.json")
    source_parity = load_json_config(work / "cpu-sleep-comparison.json")["source"]["parity"]
    if selection["selected_model"] != "0.1.17_baseline" or not cpu["accepted"]:
        raise ValueError("release requires the documented retained model and verified CPU change")
    if not source_parity["status_rank_parity"]:
        raise ValueError("CPU change regressed source predictions")
    baseline = root / "artifacts/versions/0.1.17/staging"
    target = root / "artifacts/versions/0.1.18/source"
    for name in ("runtime", "catalog"):
        destination = target / name
        if not destination.exists():
            shutil.copytree(baseline / name, destination)
        if directory_content_manifest(destination) != directory_content_manifest(baseline / name):
            raise ValueError(f"retained payload changed: {name}")
    evidence = target / "evidence"
    evidence.mkdir(exist_ok=True)
    sources = {
        "data-audit.json": work / "data-audit.json",
        "model-selection.json": work / "model-selection.json",
        "cpu-interleaved.json": work / "cpu-interleaved/report.json",
        "cpu-sleep-comparison.json": work / "cpu-sleep-comparison.json",
        "baseline-log.json": work / "baseline/log-cpu/report.json",
        "optimized-log.json": work / "measurements/cpu-sleep/log-cpu/report.json",
        "baseline-source.json": work / "baseline/source-cpu/report.json",
        "optimized-source.json": work / "measurements/cpu-sleep/source-cpu/report.json",
        "rejected-replay-training.json": work / "training/ssdlite-real-replay/report.json",
        "study-config.json": root / "configs/experiments/bread/log_improvement_018.json",
        "cpu-study-config.json": root / "configs/experiments/bread/log_cpu_sleep_018.json",
        "cpu-profile.json": root / "artifacts/versions/0.1.17/source/evidence/cpu-profile.json",
    }
    for name, source in sources.items():
        destination = evidence / name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise ValueError("release evidence changed")
        shutil.copy2(source, destination)
    previous_config = root / "configs/versions/0.1.17.json"
    if not previous_config.exists():
        previous_config = root / "configs/archive/versions/0.1.17.json"
    config = load_json_config(previous_config)
    config.update(
        version="0.1.18",
        app_build=21,
        source_date_epoch=1788912000,
        source_candidate="three_bakery/ssdlite-margin-dense-20260908-retained-cpu-sleep",
    )
    for name in ("runtime", "catalog"):
        config[name] = {
            "path": (target / name).relative_to(root).as_posix(),
            "manifest_sha256": directory_content_manifest(target / name)["manifest_sha256"],
        }
    config["catalog"]["store_id"] = "three_bakery"
    config["evaluation_evidence"] = [
        {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
        for p in sorted(evidence.iterdir())
        if p.is_file()
    ]
    config_path = root / "configs/versions/0.1.18.json"
    if config_path.exists() and load_json_config(config_path) != config:
        raise ValueError("release version inputs changed")
    write_json(config_path, config)
    freeze = {
        "product_version": "0.1.18",
        "model_changed": False,
        "catalog_changed": False,
        "policy_changed": False,
        "cpu_runtime_change": "disable per-session idle spinning",
        "runtime_code_sha256": sha256_file(root / "src/bixolon_scanner/runtime/onnx_session.py"),
        "version_config_sha256": sha256_file(config_path),
        "model_selection_sha256": sha256_file(work / "model-selection.json"),
        "cpu_confirmation_sha256": sha256_file(work / "cpu-interleaved/report.json"),
        "regression_rule": "No change in CPU/CUDA status/class ranks or correct/false approvals on historical final300; CPU full/all p95 must not regress. Do not retune after access.",
        "benchmark_role": "historically exposed fixed regression set; not independent generalization proof",
    }
    freeze_path = work / "release-selection.json"
    if freeze_path.exists() and load_json_config(freeze_path) != freeze:
        raise ValueError("release selection already frozen with different inputs")
    write_json(freeze_path, freeze)
    print(f"Frozen {config_path}")


if __name__ == "__main__":
    main()
