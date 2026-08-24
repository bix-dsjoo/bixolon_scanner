from __future__ import annotations

import pytest

from bixolon_scanner.training.dfine_export import checkpoint_model_state
from bixolon_scanner.training.dfine_score_head_merge import (
    create_dfine_score_head_checkpoint,
    merge_dfine_score_head_states,
)

torch = pytest.importorskip("torch")


def test_merge_dfine_score_head_states_preserves_non_score_state():
    reference = {
        "backbone.bn.running_mean": torch.tensor([1.0]),
        "decoder.dec_score_head.0.weight": torch.tensor([2.0]),
        "decoder.denoising_class_embed.weight": torch.tensor([3.0]),
    }
    donor = {
        "backbone.bn.running_mean": torch.tensor([9.0]),
        "decoder.dec_score_head.0.weight": torch.tensor([4.0]),
        "decoder.denoising_class_embed.weight": torch.tensor([5.0]),
    }

    merged, transferred = merge_dfine_score_head_states(reference, donor)

    assert merged["backbone.bn.running_mean"].item() == 1.0
    assert merged["decoder.dec_score_head.0.weight"].item() == 4.0
    assert merged["decoder.denoising_class_embed.weight"].item() == 5.0
    assert len(transferred) == 2


def test_merge_dfine_score_head_states_rejects_missing_score_parameter():
    with pytest.raises(ValueError, match="missing parameter"):
        merge_dfine_score_head_states(
            {"decoder.enc_score_head.weight": torch.tensor([1.0])},
            {},
        )


def test_create_dfine_score_head_checkpoint_records_provenance(tmp_path):
    reference = tmp_path / "reference.pth"
    donor = tmp_path / "donor.pth"
    output = tmp_path / "merged.pth"
    torch.save(
        {
            "ema": {
                "module": {
                    "backbone.weight": torch.tensor([1.0]),
                    "decoder.enc_score_head.weight": torch.tensor([2.0]),
                }
            }
        },
        reference,
    )
    torch.save(
        {
            "model": {
                "backbone.weight": torch.tensor([9.0]),
                "decoder.enc_score_head.weight": torch.tensor([4.0]),
            }
        },
        donor,
    )

    provenance = create_dfine_score_head_checkpoint(reference, donor, output)
    merged = checkpoint_model_state(torch.load(output, map_location="cpu", weights_only=False))

    assert merged["backbone.weight"].item() == 1.0
    assert merged["decoder.enc_score_head.weight"].item() == 4.0
    assert provenance["transferred_parameter_count"] == 1
    assert provenance["independent_test_claimed"] is False
