"""Copy the Lite app and the byte-identical verified Worker of the requested version."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import canonical_sha256, directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2
from bixolon_scanner.operations.version_bundle import load_version_config


def prepare(root: Path, version: str = "0.2.0") -> dict:
    root = root.resolve()
    config = load_version_config(root / f"configs/versions/{version}.json")
    if config.version != version:
        raise ValueError("Lite version config identity mismatch")
    worker = root / f"artifacts/installers/{version}/windows-payload/worker"
    evidence_path = root / f"artifacts/versions/{version}/packaged-cpu-smoke.json"
    evidence = load_json_config(evidence_path)
    deployment = load_json_config(worker.parent / "deployment-provenance.json")
    source = directory_content_manifest(worker)
    if (
        not evidence["passes"]
        or source["manifest_sha256"] != evidence["worker_artifact_content_manifest_sha256"]
    ):
        raise ValueError("the source Worker differs from its successful packaged smoke")
    runtime = load_runtime_package_v2(worker / "model-package")
    if runtime.metadata.worker_version != version:
        raise ValueError(f"Lite requires the unchanged {version} model")
    app = root / "apps/bakery_scanner_lite/build/windows/x64/runner/Release"
    if not (app / "bakery_scanner_lite.exe").is_file():
        raise FileNotFoundError("Lite Windows executable is missing")
    output = root / f"artifacts/lite/{version}"
    output.mkdir(parents=True, exist_ok=True)
    payload = (output / "payload").resolve()
    payload.relative_to(output.resolve())
    if payload.exists():
        shutil.rmtree(payload)
    shutil.copytree(app, payload)
    shutil.copytree(worker, payload / "worker")
    if directory_content_manifest(payload / "worker") != source:
        raise ValueError("Worker payload changed during copy")
    shutil.copy2(root / "apps/bakery_scanner_lite/README.md", payload / "LITE-KO.md")
    shutil.copytree(root / "licenses", payload / "licenses")
    evidence_target = payload / "evidence/worker-source-smoke.json"
    evidence_target.parent.mkdir()
    shutil.copy2(evidence_path, evidence_target)
    version_metadata = {
        "product_version": version,
        "app_build": config.app_build,
        "worker": {"path": "worker", "manifest_sha256": source["manifest_sha256"]},
        "evaluation_evidence": {
            "path": "evidence/worker-source-smoke.json",
            "sha256": sha256_file(evidence_target),
        },
    }
    (payload / "version.json").write_text(
        json.dumps(version_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "product_name": "BIXOLON Bakery AI Scanner Lite",
        "version": version,
        "app_build": config.app_build,
        "worker_source_manifest_sha256": source["manifest_sha256"],
        "worker_executable_sha256": sha256_file(worker / "bixolon-worker.exe"),
        "worker_modified": False,
        "model_modified": False,
        "provider": "cpu",
        "detector_threads": deployment["default_profile"]["detector_intra_op_threads"],
        "embedder_threads": deployment["default_profile"]["embedder_intra_op_threads"],
        "image_retention": "30_days_cleanup_on_start_or_log_access",
        "lite_application_source": directory_content_manifest(
            root / "apps/bakery_scanner_lite/lib"
        ),
    }
    (payload / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = directory_content_manifest(payload)
    (payload / "bundle-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verify(payload)
    return {"payload": str(payload), "manifest_sha256": manifest["manifest_sha256"]}


def verify(payload: Path) -> None:
    expected = load_json_config(payload / "bundle-manifest.json")
    actual = [
        row
        for row in directory_content_manifest(payload)["files"]
        if row["path"] != "bundle-manifest.json"
    ]
    if actual != expected["files"] or canonical_sha256(actual) != expected["manifest_sha256"]:
        raise ValueError("Lite bundle checksum mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--version", default="0.2.0")
    args = parser.parse_args()
    print(json.dumps(prepare(args.repository_root, args.version)))
