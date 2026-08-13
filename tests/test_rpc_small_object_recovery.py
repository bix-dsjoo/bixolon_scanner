from __future__ import annotations

import json

import numpy as np

from bixolon_scanner.training import rpc_small_object_recovery as recovery


def _archive(sample_ids: list[str], values: list[int]) -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray(sample_ids),
        "targets": np.asarray(values, dtype=np.int64),
    }


def test_merge_archive_handles_empty_additions_and_deduplicates() -> None:
    existing = _archive(["b", "a"], [2, 1])
    empty = _archive([], [])

    merged_empty = recovery._merge_archive(existing, empty)
    merged = recovery._merge_archive(existing, _archive(["b", "c"], [20, 3]))

    assert merged_empty["sample_ids"].tolist() == ["a", "b"]
    assert merged_empty["targets"].tolist() == [1, 2]
    assert merged["sample_ids"].tolist() == ["a", "b", "c"]
    assert merged["targets"].tolist() == [1, 2, 3]


def test_validation_package_changes_only_experimental_policy(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "validation-candidate-package"
    source.mkdir()
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "package_version": "0.0.0-rpc-v3",
                "quality": {"min_object_area_ratio": 0.005},
            }
        ),
        encoding="utf-8",
    )
    (source / "detector.onnx").write_bytes(b"detector")
    (source / "classifier.onnx").write_bytes(b"classifier")
    validated: list[object] = []
    monkeypatch.setattr(recovery, "load_model_package", validated.append)

    destination = recovery._build_validation_package(tmp_path, 0.002)
    metadata = json.loads(
        (destination / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["package_version"] == "0.0.0-rpc-small-object-v5"
    assert metadata["quality"]["min_object_area_ratio"] == 0.002
    assert (destination / "detector.onnx").read_bytes() == b"detector"
    assert (destination / "classifier.onnx").read_bytes() == b"classifier"
    assert validated == [destination]
