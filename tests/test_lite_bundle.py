"""Lite packaging keeps the verified Worker byte-identical and fails before replacing output."""

import json
from types import SimpleNamespace

import pytest

from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.operations import lite_bundle


def fixture_tree(tmp_path, monkeypatch):
    worker = tmp_path / "artifacts/installers/0.2.0/windows-payload/worker"
    worker.mkdir(parents=True)
    (worker / "bixolon-worker.exe").write_bytes(b"verified Worker")
    evidence = tmp_path / "artifacts/versions/0.2.0/packaged-cpu-smoke.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "passes": True,
                "worker_artifact_content_manifest_sha256": directory_content_manifest(worker)[
                    "manifest_sha256"
                ],
            }
        )
    )
    app = tmp_path / "apps/bakery_scanner_lite"
    release = app / "build/windows/x64/runner/Release"
    release.mkdir(parents=True)
    (release / "bakery_scanner_lite.exe").write_bytes(b"Lite")
    (app / "README.md").write_text("Lite usage")
    (app / "lib").mkdir()
    (app / "lib/main.dart").write_text("source")
    (tmp_path / "licenses").mkdir()
    (tmp_path / "licenses/APACHE-2.0.txt").write_text("license fixture")
    monkeypatch.setattr(
        lite_bundle,
        "load_runtime_package_v2",
        lambda _: SimpleNamespace(
            metadata=SimpleNamespace(worker_version="0.2.0"),
        ),
    )
    monkeypatch.setattr(
        lite_bundle,
        "load_version_config",
        lambda _: SimpleNamespace(version="0.2.0", app_build=23),
    )
    (worker.parent / "deployment-provenance.json").write_text(
        json.dumps(
            {"default_profile": {"detector_intra_op_threads": 8, "embedder_intra_op_threads": 12}}
        )
    )
    return worker


def test_lite_payload_preserves_source_and_records_manifest(tmp_path, monkeypatch):
    worker = fixture_tree(tmp_path, monkeypatch)
    result = lite_bundle.prepare(tmp_path)
    payload = tmp_path / "artifacts/lite/0.2.0/payload"
    assert directory_content_manifest(payload / "worker") == directory_content_manifest(worker)
    provenance = json.loads((payload / "provenance.json").read_text())
    assert provenance["worker_modified"] is False
    assert provenance["version"] == "0.2.0"
    assert (payload / "bakery_scanner_lite.exe").read_bytes() == b"Lite"
    assert (
        json.loads((payload / "bundle-manifest.json").read_text())["manifest_sha256"]
        == result["manifest_sha256"]
    )
    lite_bundle.verify(payload)
    (payload / "bakery_scanner_lite.exe").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        lite_bundle.verify(payload)


def test_lite_source_tampering_preserves_existing_output(tmp_path, monkeypatch):
    worker = fixture_tree(tmp_path, monkeypatch)
    output = tmp_path / "artifacts/lite/0.2.0/payload"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("previous build")
    (worker / "bixolon-worker.exe").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        lite_bundle.prepare(tmp_path)
    assert (output / "keep.txt").read_text() == "previous build"


def test_lite_rejects_wrong_model_version(tmp_path, monkeypatch):
    fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        lite_bundle,
        "load_runtime_package_v2",
        lambda _: SimpleNamespace(
            metadata=SimpleNamespace(worker_version="0.1.15"),
        ),
    )
    with pytest.raises(ValueError, match="unchanged 0.2.0"):
        lite_bundle.prepare(tmp_path)
