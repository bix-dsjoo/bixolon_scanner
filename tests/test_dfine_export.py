from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.training.dfine_export import (
    apply_export_resolution,
    checkpoint_model_state,
    compatible_checkpoint_state,
)


def test_checkpoint_model_state_prefers_ema_weights():
    ema = {"weight": object()}
    model = {"weight": object()}

    assert checkpoint_model_state({"ema": {"module": ema}, "model": model}) is ema


def test_checkpoint_model_state_falls_back_to_model_weights():
    model = {"weight": object()}

    assert checkpoint_model_state({"model": model}) is model


def test_checkpoint_model_state_rejects_missing_weights():
    with pytest.raises(ValueError, match="neither EMA nor model"):
        checkpoint_model_state({"ema": {}})


def test_compatible_checkpoint_state_drops_resolution_derived_buffers_only():
    current = {
        "decoder.anchors": np.zeros((1, 20, 4)),
        "decoder.valid_mask": np.zeros((1, 20, 1)),
        "weight": np.zeros((2, 2)),
    }
    checkpoint = {
        "decoder.anchors": np.zeros((1, 10, 4)),
        "decoder.valid_mask": np.zeros((1, 10, 1)),
        "weight": np.ones((2, 2)),
    }

    result = compatible_checkpoint_state(current, checkpoint)

    assert set(result) == {"weight"}


def test_apply_export_resolution_updates_global_shared_setting():
    config = {"eval_spatial_size": [640, 640]}

    apply_export_resolution(config, (576, 608))

    assert config["eval_spatial_size"] == [576, 608]
