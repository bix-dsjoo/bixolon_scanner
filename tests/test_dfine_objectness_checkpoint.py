from __future__ import annotations

import math

import pytest

from bixolon_scanner.training.dfine_export import checkpoint_model_state
from bixolon_scanner.training.dfine_objectness_checkpoint import (
    collapse_dfine_state_to_objectness,
    create_dfine_objectness_tuning_checkpoint,
)

torch = pytest.importorskip("torch")


def _state():
    return {
        "backbone.weight": torch.tensor([9.0]),
        "decoder.enc_score_head.weight": torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
        "decoder.enc_score_head.bias": torch.tensor([-2.0, -1.0]),
        "decoder.dec_score_head.0.weight": torch.tensor([[2.0, 4.0], [4.0, 6.0]]),
        "decoder.dec_score_head.0.bias": torch.tensor([-3.0, -2.0]),
        "decoder.denoising_class_embed.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]]),
    }


def test_collapse_dfine_state_removes_product_class_outputs():
    converted, report = collapse_dfine_state_to_objectness(_state())

    assert converted["backbone.weight"].tolist() == [9.0]
    assert converted["decoder.enc_score_head.weight"].tolist() == [[2.0, 4.0]]
    assert converted["decoder.enc_score_head.bias"].item() == pytest.approx(
        math.log(math.exp(-2.0) + math.exp(-1.0))
    )
    assert converted["decoder.denoising_class_embed.weight"].tolist() == [
        [2.0, 3.0],
        [8.0, 9.0],
    ]
    assert report["source_class_count"] == 2
    assert report["target_class_count"] == 1
    assert report["requires_one_class_tuning"] is True


def test_collapse_dfine_state_rejects_inconsistent_class_count():
    state = _state()
    state["decoder.dec_score_head.0.bias"] = torch.tensor([-3.0, -2.0, -1.0])

    with pytest.raises(ValueError, match="disagree on class count"):
        collapse_dfine_state_to_objectness(state)


def test_create_objectness_checkpoint_records_non_release_provenance(tmp_path):
    source = tmp_path / "source.pth"
    output = tmp_path / "objectness.pth"
    torch.save({"ema": {"module": _state()}}, source)

    provenance = create_dfine_objectness_tuning_checkpoint(source, output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    state = checkpoint_model_state(payload)

    assert state["decoder.enc_score_head.weight"].shape == (1, 2)
    assert provenance["product_class_outputs_removed"] is True
    assert provenance["release_model"] is False
