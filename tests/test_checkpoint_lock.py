from __future__ import annotations

from pathlib import Path

import pytest

from bixolon_scanner.cli import COMMANDS
from bixolon_scanner.training.checkpoint_lock import lock_detector_checkpoint


def test_detector_checkpoint_lock_embeds_native_provenance_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pth"
    torch.save({"model": {"weight": torch.ones(1)}}, source)
    artifacts = {}
    for name in ("manifest", "synthetic", "coco", "config"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = path
    output = tmp_path / "locked.pth"

    result = lock_detector_checkpoint(
        source,
        output,
        dataset_version="bread-test",
        manifest=artifacts["manifest"],
        source_revision="a" * 40,
        source_weight_sha256="b" * 64,
        synthetic_manifest=artifacts["synthetic"],
        coco_provenance=artifacts["coco"],
        training_config=artifacts["config"],
        epochs=40,
        batch_size=8,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-4,
        weight_decay=1e-4,
        synthetic_seed=7,
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["model"]["weight"].item() == 1
    assert payload["bixolon_training_provenance"]["dataset_version"] == "bread-test"
    assert result["sha256"]
    assert ("train", "lock-detector-checkpoint") in COMMANDS
    with pytest.raises(FileExistsError):
        lock_detector_checkpoint(
            source,
            output,
            dataset_version="bread-test",
            manifest=artifacts["manifest"],
            source_revision="a" * 40,
            source_weight_sha256="b" * 64,
            synthetic_manifest=artifacts["synthetic"],
            coco_provenance=artifacts["coco"],
            training_config=artifacts["config"],
            epochs=40,
            batch_size=8,
            backbone_learning_rate=1e-5,
            head_learning_rate=1e-4,
            weight_decay=1e-4,
            synthetic_seed=7,
        )
