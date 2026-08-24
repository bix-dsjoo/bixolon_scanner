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


def test_average_dfine_states_supports_weighted_interpolation():
    result = average_dfine_states(
        [
            {"weight": torch.tensor([1.0, 3.0])},
            {"weight": torch.tensor([5.0, 7.0])},
        ],
        [0.75, 0.25],
    )

    assert result["weight"].tolist() == [2.0, 4.0]


@pytest.mark.parametrize("weights", [[1.0], [-1.0, 2.0], [0.0, 0.0]])
def test_average_dfine_states_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weight"):
        average_dfine_states(
            [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([3.0])}],
            weights,
        )


def test_average_dfine_states_rejects_changed_integer_buffer():
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        average_dfine_states(
            [
                {"step": torch.tensor(1)},
                {"step": torch.tensor(2)},
            ]
        )


def test_average_dfine_states_preserves_reference_only_integer_buffer():
    result = average_dfine_states(
        [
            {"weight": torch.tensor([1.0]), "step": torch.tensor(2)},
            {"weight": torch.tensor([3.0])},
        ]
    )

    assert result["weight"].item() == 2.0
    assert result["step"].item() == 2


def test_average_dfine_states_rejects_reference_only_float_tensor():
    with pytest.raises(ValueError, match="floating tensor is missing"):
        average_dfine_states(
            [
                {"weight": torch.tensor([1.0]), "optional": torch.tensor([2.0])},
                {"weight": torch.tensor([3.0])},
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


def test_create_dfine_checkpoint_soup_records_normalized_weights(tmp_path):
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    output = tmp_path / "soup.pth"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, first)
    torch.save({"model": {"weight": torch.tensor([5.0])}}, second)

    provenance = create_dfine_checkpoint_soup([first, second], output, weights=[9.0, 1.0])
    result = torch.load(output, map_location="cpu", weights_only=False)

    assert checkpoint_model_state(result)["weight"].item() == pytest.approx(1.4)
    assert provenance["recipe"] == "weighted_inference_parameter_interpolation"
    assert [member["weight"] for member in provenance["members"]] == [0.9, 0.1]
    assert provenance["selection_scope"] == "development_dataset_only"
