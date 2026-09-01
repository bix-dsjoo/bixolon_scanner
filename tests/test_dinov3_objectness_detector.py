from pathlib import Path

import pytest

from bixolon_scanner.experiments.bread.dinov3_objectness_detector import (
    checkpoint_rank,
    load_config,
)
from bixolon_scanner.training.dinov3_objectness_detector import target_to_tensors


def test_dinov3_detector_config_excludes_existing_and_yolo_detectors() -> None:
    config = load_config(Path("configs/experiments/bread/dinov3_objectness_fcos_0.1.json"))

    assert config["prohibited_inputs"] == {
        "existing_detector_weights": True,
        "existing_detector_predictions": True,
        "yolo_family": True,
        "rfdetr": True,
    }
    assert config["backbone"]["kind"] == "dinov3_convnext_tiny"


def test_dinov3_detector_rejects_incomplete_exclusions(tmp_path: Path) -> None:
    source = Path("configs/experiments/bread/dinov3_objectness_fcos_0.1.json")
    value = source.read_text(encoding="utf-8").replace(
        '"yolo_family": true', '"yolo_family": false'
    )
    path = tmp_path / "invalid.json"
    path.write_text(value, encoding="utf-8")

    with pytest.raises(ValueError, match="exclusions"):
        load_config(path)


def test_checkpoint_rank_never_trades_a_miss_for_fewer_proposals() -> None:
    broad = {
        "false_negative_count": 0,
        "missed_image_count": 0,
        "surplus_proposal_count": 1000,
    }
    narrow = {
        "false_negative_count": 1,
        "missed_image_count": 1,
        "surplus_proposal_count": 0,
    }

    assert checkpoint_rank(broad, 1) > checkpoint_rank(narrow, 99)


def test_target_to_tensors_builds_one_class_xyxy_targets() -> None:
    target = target_to_tensors(
        {
            "image_id": 7,
            "annotations": [
                {"bbox": [10.0, 20.0, 30.0, 40.0]},
                {"bbox": [0.0, 1.0, 2.0, 3.0]},
            ],
        }
    )

    assert target["boxes"].tolist() == [[10.0, 20.0, 40.0, 60.0], [0.0, 1.0, 2.0, 4.0]]
    assert target["labels"].tolist() == [0, 0]
    assert target["image_id"].item() == 7
