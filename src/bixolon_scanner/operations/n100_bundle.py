"""Assemble a self-contained Intel UHD Worker without changing model payloads."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..training.three_bakery_data import read_jsonl, write_json
from .lite_bundle import verify as verify_manifest
from .version_bundle import load_version_config


def prepare(root: Path, version: str) -> Path:
    config = load_version_config(root / f"configs/versions/{version}.json")
    staging = root / f"artifacts/versions/{version}/staging"
    binary = root / f"artifacts/versions/{version}/openvino-worker-build/bixolon-worker"
    runtime = load_runtime_package_v2(staging / "runtime")
    catalog = load_store_catalog_package(
        staging / "catalog", expected_store_id=config.catalog.store_id
    )
    if runtime.metadata.worker_version != version or catalog.metadata.catalog_version != version:
        raise ValueError("N100 payload version mismatch")
    for filename in (
        "openvino.dll",
        "openvino_intel_gpu_plugin.dll",
        "openvino_intel_cpu_plugin.dll",
    ):
        if not (binary / "_internal" / filename).is_file():
            raise ValueError("N100 OpenVINO dependency missing")
    forbidden = [
        p
        for p in binary.rglob("*")
        if p.is_file() and (p.name.startswith("torch") or "providers_cuda" in p.name)
    ]
    if forbidden:
        raise ValueError("N100 Worker contains a training or CUDA dependency")
    output = root / f"artifacts/n100/{version}/worker-payload"
    if output.exists():
        raise ValueError("N100 payload exists; inspect it before rebuilding")
    shutil.copytree(binary, output / "worker")
    for name, target in [("runtime", "model-package"), ("catalog", "store-catalog")]:
        shutil.copytree(staging / name, output / "worker" / target)
        if directory_content_manifest(staging / name) != directory_content_manifest(
            output / "worker" / target
        ):
            raise ValueError("N100 model copy changed bytes")
    shutil.copytree(root / "licenses", output / "licenses")
    openvino_license = (
        root
        / "artifacts/build-envs/worker-openvino-gpu-py311/Lib/site-packages"
        / "openvino-2025.4.1.dist-info/licenses/LICENSE"
    )
    shutil.copy2(openvino_license, output / "licenses/OPENVINO-LICENSE.txt")
    shutil.copy2(root / "docs/experiments/n100-0.2.0.md", output / "N100-KO.md")
    shutil.copy2(root / "installer/windows/start-n100-worker.ps1", output / "start-n100-worker.ps1")
    shutil.copy2(root / "installer/windows/measure-n100.ps1", output / "measure-n100.ps1")
    write_json(
        output / "log132-image-sha256.json",
        {
            "image_sha256": sorted(
                row["image_sha256"]
                for row in read_jsonl(
                    root / f"artifacts/versions/{version}/source/evidence/log-inputs.jsonl"
                )
            )
        },
    )
    (output / "RUN-N100-WORKER.cmd").write_text(
        '@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-n100-worker.ps1" %*\r\n',
        encoding="ascii",
    )
    profile = {
        "provider": "cpu",
        "embedder_provider": "openvino_gpu",
        "embedder_fallback_provider": "same",
        "provider_execution_cpu_fallback": True,
        "verifier_provider": "cpu",
        "cpu_detector_intra_op_threads": 2,
        "cpu_embedder_intra_op_threads": 4,
    }
    write_json(output / "version.json", {"product_version": version, "app_build": config.app_build})
    write_json(
        output / "provenance.json",
        {
            "version": version,
            "target": "Intel N100 / integrated Intel UHD",
            "execution_profile": profile,
            "source_worker": directory_content_manifest(binary),
            "source_runtime": directory_content_manifest(staging / "runtime"),
            "source_catalog": directory_content_manifest(staging / "catalog"),
            "dependency_lock_sha256": sha256_file(
                root / "configs/runtime/requirements-windows-openvino.lock"
            ),
            "runtime_and_catalog_payload_changed": False,
            "publisher_authentication": "UNSIGNED",
            "n100_hardware_measurement": "not_performed_on_build_host",
        },
    )
    write_json(output / "bundle-manifest.json", directory_content_manifest(output))
    return output


def prepare_lite(root: Path, version: str) -> Path:
    source = root / f"artifacts/n100/{version}/worker-payload"
    verify_manifest(source)
    app = root / "apps/bakery_scanner_lite/build/windows/x64/runner/Release"
    if not (app / "bakery_scanner_lite.exe").is_file():
        raise ValueError("N100 Lite executable missing")
    output = root / f"artifacts/n100/{version}/lite-payload"
    if output.exists():
        raise ValueError("N100 Lite output exists; inspect before rebuilding")
    shutil.copytree(app, output)
    for path in source.iterdir():
        if path.name == "bundle-manifest.json":
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    if directory_content_manifest(source / "worker") != directory_content_manifest(
        output / "worker"
    ):
        raise ValueError("N100 Lite changed the Worker payload")
    write_json(output / "bundle-manifest.json", directory_content_manifest(output))
    verify_manifest(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--lite", action="store_true")
    args = parser.parse_args()
    action = prepare_lite if args.lite else prepare
    print(action(args.repository_root.resolve(), args.version))


if __name__ == "__main__":
    main()
