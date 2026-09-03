from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from bixolon_scanner.experiments.bread import build_fixed224_classifier_runtime as builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_promotes_224_fallback_and_enables_any_verifier_safety(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "embedder.onnx").write_bytes(b"primary-192")
    (source / "embedder-fallback-224.onnx").write_bytes(b"primary-224")
    (source / "detector.onnx").write_bytes(b"detector")
    (source / "classifier-verifier.onnx").write_bytes(b"verifier")
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "embedder": {"filename": "embedder.onnx"},
                "classifier_resolution_fallback": {
                    "embedder": {
                        "filename": "embedder-fallback-224.onnx",
                        "input_size": [224, 224],
                    }
                },
                "classifier_verification": {
                    "ambiguity_maximum_approval_score": 0.5,
                    "unknown_recapture_on_dual_verifier_rejection": True,
                },
                "sources": {"embedder": {"architecture": "cascade"}},
                "checksums": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    def fake_loader(path: Path):
        if path == source:
            return SimpleNamespace(
                metadata=SimpleNamespace(
                    detector_class_mode="class_agnostic",
                    classifier_resolution_fallback=SimpleNamespace(
                        embedder=SimpleNamespace(input_size=(224, 224))
                    ),
                    classifier_verification=SimpleNamespace(),
                ),
                classifier_fallback_embedder_path=(source / "embedder-fallback-224.onnx"),
            )
        assert path == output
        return SimpleNamespace(
            metadata=SimpleNamespace(
                detector_class_mode="class_agnostic",
                embedder=SimpleNamespace(input_size=(224, 224)),
                classifier_resolution_fallback=None,
                classifier_verification=SimpleNamespace(
                    ambiguity_maximum_approval_score=0.55,
                    unknown_recapture_on_any_verifier_rejection=True,
                    independent_embedder=SimpleNamespace(input_size=(160, 160)),
                ),
            ),
            embedder_path=output / "embedder.onnx",
        )

    monkeypatch.setattr(builder, "load_runtime_package_v2", fake_loader)
    report = builder.build(
        argparse.Namespace(
            source_runtime=source,
            output_dir=output,
            ambiguity_maximum_approval_score=0.55,
        )
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert (output / "embedder.onnx").read_bytes() == b"primary-224"
    assert not (output / "embedder-fallback-224.onnx").exists()
    assert metadata["embedder"]["input_size"] == [224, 224]
    assert metadata["classifier_resolution_fallback"] is None
    assert metadata["classifier_verification"] == {
        "ambiguity_maximum_approval_score": 0.55,
        "unknown_recapture_on_dual_verifier_rejection": False,
        "unknown_recapture_on_any_verifier_rejection": True,
    }
    assert metadata["checksums"]["embedder.onnx"] == _sha256(output / "embedder.onnx")
    assert report["candidate_primary_embedder_sha256"] == _sha256(output / "embedder.onnx")
    assert report["model_graph_or_weight_changed"] is False
