from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.operations.local_artifacts import (
    build_cleanup_plan,
    execute_cleanup,
)


def _write(path: Path, value: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    _write(root / "pyproject.toml", b"[project]\nname='test'\n")
    runtime = _write(root / "artifacts/packages/runtime/model.onnx")
    evaluation = _write(root / "artifacts/evaluations/current/result.json")
    config = {
        "version": "0.1.8",
        "runtime": {
            "path": runtime.parent.relative_to(root).as_posix(),
            "manifest_sha256": "a" * 64,
        },
        "evaluation_evidence": [
            {
                "path": evaluation.relative_to(root).as_posix(),
                "sha256": "b" * 64,
            }
        ],
    }
    config_path = root / "configs/versions/0.1.8.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (root / "configs/archive").mkdir(parents=True)
    return root


def test_cleanup_plan_preserves_data_references_and_evidence(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    archive_evidence = _write(root / "artifacts/experiments/archive/metadata.json")
    (root / "configs/archive/versions").mkdir(parents=True)
    (root / "configs/archive/versions/0.1.7.json").write_text(
        json.dumps(
            {
                "evaluation_evidence": [
                    {
                        "path": archive_evidence.relative_to(root).as_posix(),
                        "sha256": "c" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write(root / "datasets/source/image.jpg")
    _write(root / "artifacts/experiments/run/report.json")
    _write(root / "artifacts/experiments/run/model.onnx")
    _write(root / "artifacts/experiments/openvino-venv/Lib/module.pyd")
    _write(root / "artifacts/experiments/openvino-venv/Lib/module.py")
    _write(root / "artifacts/versions/0.1.7/old.bin")
    _write(root / "artifacts/versions/0.1.8/final.bin")
    _write(root / "apps/product_scanner/build/intermediate.dll")
    _write(root / "runs/run-1/cache.bin")

    plan = build_cleanup_plan(root)
    candidates = {entry.path for entry in plan.candidates}
    preserved = {entry.path for entry in plan.preserved}

    assert "datasets" in preserved
    assert "artifacts/packages/runtime" in preserved
    assert "artifacts/evaluations/current/result.json" in preserved
    assert "artifacts/experiments/archive/metadata.json" in preserved
    assert "artifacts/versions/0.1.8" in preserved
    assert "artifacts/versions/0.1.7" in candidates
    assert "apps/product_scanner/build" in candidates
    assert "runs" in candidates
    assert "artifacts/experiments/run/model.onnx" in candidates
    assert "artifacts/experiments/openvino-venv/Lib/module.py" in candidates
    assert "artifacts/experiments/run/report.json" not in candidates


def test_cleanup_apply_deletes_only_planned_workspace_paths(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    dataset = _write(root / "datasets/source/image.jpg")
    report = _write(root / "artifacts/experiments/run/training-report.json")
    model = _write(root / "artifacts/experiments/run/model.onnx")
    cache = _write(root / ".ruff_cache/index")
    outside = _write(tmp_path / "outside/keep.bin")
    manifest = root / "artifacts/reports/maintenance/cleanup.json"

    payload = execute_cleanup(build_cleanup_plan(root), manifest)

    assert payload["status"] == "completed"
    assert payload["summary"]["actual_freed_bytes"] >= 0
    assert manifest.is_file()
    assert dataset.is_file()
    assert report.is_file()
    assert outside.is_file()
    assert not model.exists()
    assert not cache.exists()


def test_cleanup_rejects_non_repository_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="저장소 루트 표식"):
        build_cleanup_plan(tmp_path)


def test_cleanup_rejects_manifest_outside_reports(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    with pytest.raises(ValueError, match="artifacts/reports"):
        execute_cleanup(build_cleanup_plan(root), root / "cleanup.json")
