from __future__ import annotations

import pytest

from bixolon_scanner.training.dfine_checkpoint_soup import (
    average_dfine_states,
    create_dfine_checkpoint_soup,
)
from bixolon_scanner.training.dfine_export import checkpoint_model_state

torch = pytest.importorskip("torch")


def test_average_dfine_states_averages_float_and_preserves_integer():
    result = average_dfine_states(
        [
            {"weight": torch.tensor([1.0, 3.0]), "step": torch.tensor(2)},
            {"weight": torch.tensor([3.0, 5.0]), "step": torch.tensor(2)},
        ]
    )

    assert result["weight"].tolist() == [2.0, 4.0]
    assert result["step"].item() == 2


def test_average_dfine_states_rejects_changed_integer_buffer():
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        average_dfine_states(
            [
                {"step": torch.tensor(1)},
                {"step": torch.tensor(2)},
            ]
        )


def test_create_dfine_checkpoint_soup_records_members(tmp_path):
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    output = tmp_path / "soup.pth"
    torch.save({"last_epoch": 4, "ema": {"module": {"weight": torch.tensor([1.0])}}}, first)
    torch.save({"last_epoch": 5, "ema": {"module": {"weight": torch.tensor([3.0])}}}, second)

    provenance = create_dfine_checkpoint_soup([first, second], output)
    result = torch.load(output, map_location="cpu", weights_only=False)

    assert checkpoint_model_state(result)["weight"].item() == 2.0
    assert provenance["member_count"] == 2
    assert provenance["independent_test_claimed"] is False
    assert all(len(member["sha256"]) == 64 for member in provenance["members"])
