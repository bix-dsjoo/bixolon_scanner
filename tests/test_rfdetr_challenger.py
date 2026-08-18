import json
from pathlib import Path

import pytest

from bixolon_scanner.experiments.bread.rfdetr_challenger import load_config, training_kwargs

CONFIG = Path("configs/experiments/bread/rfdetr_large_bread_1.1.0.json")


def test_rfdetr_challenger_locks_group_aware_class_aware_three_fold_contract():
    config = load_config(CONFIG)

    assert config["dataset"]["folds"] == [0, 1, 2]
    assert config["dataset"]["group_fold_overlap_allowed"] is False
    assert config["dataset"]["class_mode"] == "class_aware_20"
    assert config["model"]["num_classes"] == 20
    assert config["training"]["run_test"] is False


def test_rfdetr_challenger_builds_reproducible_training_arguments(tmp_path):
    config = load_config(CONFIG)
    kwargs = training_kwargs(
        config,
        {
            "dataset": tmp_path / "dataset",
            "output": tmp_path / "output",
            "checkpoint": tmp_path / "checkpoint.pth",
        },
        epochs=1,
    )

    assert kwargs["epochs"] == 1
    assert kwargs["resolution"] == 704
    assert kwargs["batch_size"] * kwargs["grad_accum_steps"] == 16
    assert kwargs["seed"] == 20260818
    assert kwargs["square_resize_div_64"] is True
    assert kwargs["expanded_scales"] is False
    assert kwargs["run_test"] is False


def test_rfdetr_challenger_rejects_group_overlap_configuration(tmp_path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["group_fold_overlap_allowed"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="group-fold overlap"):
        load_config(path)
