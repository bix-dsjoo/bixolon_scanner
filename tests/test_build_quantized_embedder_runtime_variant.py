import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from bixolon_scanner.experiments.bread import (
    build_quantized_embedder_runtime_variant as runtime_builder,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_replaces_only_primary_embedder_and_refreshes_checksums(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "embedder.onnx").write_bytes(b"fp32-primary")
    (source / "embedder-fallback-224.onnx").write_bytes(b"fp32-fallback")
    (source / "detector.onnx").write_bytes(b"detector")
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "embedder": {"filename": "embedder.onnx"},
                "sources": {"embedder": {"architecture": "DINOv3 ConvNeXt-Tiny"}},
                "checksums": {},
            }
        ),
        encoding="utf-8",
    )
    quantized = tmp_path / "embedder-int8.onnx"
    quantized.write_bytes(b"int8-primary")
    output = tmp_path / "output"

    def fake_loader(path: Path):
        assert path in {source, output}
        return SimpleNamespace(
            metadata=SimpleNamespace(embedder=SimpleNamespace(filename="embedder.onnx"))
        )

    monkeypatch.setattr(runtime_builder, "load_runtime_package_v2", fake_loader)
    report = runtime_builder.build(
        argparse.Namespace(
            source_runtime=source,
            quantized_embedder=quantized,
            quantization_description="test quantization",
            output_dir=output,
        )
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert (output / "embedder.onnx").read_bytes() == b"int8-primary"
    assert (output / "embedder-fallback-224.onnx").read_bytes() == b"fp32-fallback"
    assert metadata["checksums"]["embedder.onnx"] == _sha256(output / "embedder.onnx")
    assert metadata["checksums"]["embedder-fallback-224.onnx"] == _sha256(
        output / "embedder-fallback-224.onnx"
    )
    assert "NNCF INT8 PTQ (test quantization)" in metadata["sources"]["embedder"]["architecture"]
    assert report["weights_modified"] is True
    assert report["quantized_embedder_sha256"] == _sha256(quantized)
