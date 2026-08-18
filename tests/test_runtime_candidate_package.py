import hashlib
import json
from pathlib import Path

from bixolon_scanner.experiments.bread import runtime_candidate_package


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_runtime_candidate_package_is_checksum_verified_and_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path
    manifest_dir = root / "manifests" / "candidate"
    manifest_dir.mkdir(parents=True)
    source = root / "artifacts" / "model.onnx"
    source.parent.mkdir()
    source.write_bytes(b"model")
    metadata_bytes = json.dumps(
        {"checksums": {"model.onnx": _sha(b"model")}}, separators=(",", ":")
    ).encode()
    template = manifest_dir / "metadata.json"
    template.write_bytes(metadata_bytes + b"\n")
    manifest = {
        "candidate_id": "candidate-v1",
        "package": {
            "path": "artifacts/package",
            "metadata_template": "manifests/candidate/metadata.json",
            "metadata_template_sha256": _sha(metadata_bytes + b"\n"),
            "metadata_sha256": _sha(metadata_bytes + b"\n"),
        },
        "package_sources": {"model.onnx": "artifacts/model.onnx"},
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Metadata:
        worker_version = "1.1.0"
        detector = type("Detector", (), {"version": "1.1.0"})()
        classifier = type("Classifier", (), {"version": "1.1.0"})()
        promotion_status = "development"

    package = type("Package", (), {"metadata": Metadata()})()
    monkeypatch.setattr(runtime_candidate_package, "load_model_package", lambda _: package)

    first = runtime_candidate_package.assemble_runtime_candidate(manifest_path)
    second = runtime_candidate_package.assemble_runtime_candidate(manifest_path)

    assert first["files"][0]["action"] == "copied"
    assert second["files"][0]["action"] == "reused"
    assert (root / "artifacts" / "package" / "model.onnx").read_bytes() == b"model"
    assert (root / "artifacts" / "package" / "metadata.json").read_bytes() == metadata_bytes + b"\n"
