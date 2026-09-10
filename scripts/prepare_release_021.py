"""Freeze the improved proposal policy and advance active product surfaces."""

import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.three_bakery_data import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / "artifacts/retraining/recapture-0.2.1"
    target = root / "artifacts/versions/0.2.1/source"
    source = work / "candidates/score040"
    for relative in ("original/cpu", "log/cpu", "log/cuda"):
        if not load_json_config(work / f"confirmation/{relative}/comparison.json")["accepted"]:
            raise ValueError(f"candidate comparison did not pass: {relative}")
    if target.exists():
        raise ValueError("0.2.1 source already frozen")
    for name in ("runtime", "catalog"):
        shutil.copytree(source / name, target / name)
    evidence = target / "evidence"
    evidence.mkdir()
    evidence_sources = {
        "log-inputs.jsonl": root
        / "artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl",
        "recapture-causes.json": work / "recapture-causes.json",
        "proposal-pruning-diagnosis.json": work / "proposal-pruning-diagnosis.json",
        "original-comparison.json": work / "confirmation/original/cpu/comparison.json",
        "source-baseline.json": work / "source-baseline.json",
        "cpu-comparison.json": work / "confirmation/log/cpu/comparison.json",
        "cuda-comparison.json": work / "confirmation/log/cuda/comparison.json",
        "cpu-cuda-parity.json": work / "confirmation/log/cuda/cpu-cuda-parity.json",
    }
    for name in ("baseline", "score030", "score040"):
        evidence_sources[f"{name}.json"] = work / f"measurements/development/{name}-cpu/report.json"
    for name in ("score040-maskoff", "score040-bias0"):
        evidence_sources[f"rejected-{name}.json"] = (
            work / f"measurements/mask-study/{name}-cpu/report.json"
        )
    for name, path in evidence_sources.items():
        shutil.copy2(path, evidence / name)
    config = load_json_config(root / "configs/versions/0.2.0.json")
    config.update(
        version="0.2.1", app_build=24, source_candidate="dfine-margin-dense-20260908-proposal040"
    )
    for name in ("runtime", "catalog"):
        config[name] = {
            "path": f"artifacts/versions/0.2.1/source/{name}",
            "manifest_sha256": directory_content_manifest(target / name)["manifest_sha256"],
        }
    config["catalog"]["store_id"] = "three_bakery"
    config["evaluation_evidence"] = [
        {"path": f"artifacts/versions/0.2.1/source/evidence/{p.name}", "sha256": sha256_file(p)}
        for p in sorted(evidence.iterdir())
    ]
    write_json(root / "configs/versions/0.2.1.json", config)
    write_json(
        work / "selection.json",
        {
            "version": "0.2.1",
            "candidate": "score040",
            "runtime_sha256": sha256_file(source / "runtime/metadata.json"),
            "changes": {"detector.score_threshold": {"before": 0.025, "after": 0.04}},
            "weights_changed": False,
            "approval_and_recapture_policy_changed": False,
            "rejected": [
                "containment NMS loses real GT",
                "maskoff loses 167 correct approvals",
                "bias0 creates one wrong approval",
            ],
            "data_role": "Log132 exposed development policy selection, excluded from weight training. Final300 is regression-only after this freeze; no tuning from its outcomes.",
            "n100_measurement": "After development completion; no connected N100 target on the build host.",
        },
    )
    # Preserve prior contracts verbatim before creating the new active copies.
    for relative, archive in (
        ("configs/versions/0.2.0.json", "configs/archive/versions/0.2.0.json"),
        (
            "docs/contracts/worker-integration-0.2.0.md",
            "docs/archive/contracts/worker-integration-0.2.0.md",
        ),
        ("docs/contracts/examples/0.2.0", "docs/archive/contracts/examples/0.2.0"),
    ):
        a, b = root / relative, root / archive
        b.parent.mkdir(parents=True, exist_ok=True)
        a.rename(b)
    shutil.copy2(root / "docs/status/current.md", root / "docs/archive/status/0.2.0.md")
    shutil.copytree(
        root / "docs/archive/contracts/examples/0.2.0", root / "docs/contracts/examples/0.2.1"
    )
    shutil.copy2(
        root / "docs/archive/contracts/worker-integration-0.2.0.md",
        root / "docs/contracts/worker-integration-0.2.1.md",
    )
    paths = [
        "pyproject.toml",
        "AGENTS.md",
        "README.md",
        "configs/README.md",
        "docs/README.md",
        "docs/contracts/api.md",
        "docs/contracts/flutter-worker-client-example.md",
        "docs/contracts/worker-integration-0.2.1.md",
        "src/bixolon_scanner/__init__.py",
        "src/bixolon_scanner/operations/lite_bundle.py",
        "scripts/build_app.ps1",
        "scripts/build_windows_installer.ps1",
        "scripts/build_lite.ps1",
        "scripts/build_n100.ps1",
        "scripts/build_external_sdk.ps1",
        "installer/windows/INSTALL-KO.txt",
        "installer/windows/WORKER-KO.txt",
        "apps/product_scanner/pubspec.yaml",
        "apps/product_scanner/README.md",
        "apps/product_scanner/windows/CMakeLists.txt",
        "apps/bakery_scanner_lite/pubspec.yaml",
        "apps/bakery_scanner_lite/README.md",
        "sdk/external/README-KO.md",
        "sdk/external/deployment/active-bundle.example.json",
        "sdk/flutter/bixolon_scanner_sdk/pubspec.yaml",
        "tests/test_documentation.py",
        "tests/test_windows_installer.py",
        "tests/test_lite_bundle.py",
        "tests/test_external_sdk_packaging.py",
        "tests/test_handoff_artifacts.py",
    ]
    selected = {root / p for p in paths}
    selected.update((root / "docs/contracts/examples/0.2.1").glob("*.json"))
    for app in ("product_scanner", "bakery_scanner_lite"):
        for directory in ("lib", "test", "integration_test"):
            selected.update((root / "apps" / app / directory).rglob("*.dart"))
    for path in sorted(selected):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("0.2.0+23", "0.2.1+24").replace("0.2.0", "0.2.1")
        if path.name in {
            "README.md",
            "README-KO.md",
            "pubspec.yaml",
            "build_external_sdk.ps1",
            "test_external_sdk_packaging.py",
        }:
            text = text.replace("1.2.0", "1.2.1")
        if path.name == "test_documentation.py":
            text = text.replace("test_only_020_", "test_only_021_").replace(
                '"app_build": 23', '"app_build": 24'
            )
        if path.name == "test_lite_bundle.py":
            text = text.replace("app_build=23", "app_build=24")
        if path == root / "apps/bakery_scanner_lite/README.md":
            text = text.replace("내부 build는 20다", "내부 build는 24다")
        path.write_text(text, encoding="utf-8")
    print("0.2.1 source frozen; 0.2.0 contracts archived", flush=True)


if __name__ == "__main__":
    main()
