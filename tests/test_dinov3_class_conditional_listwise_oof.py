import numpy as np
import pytest
import torch

from bixolon_scanner.experiments.bread.dinov3_class_conditional_listwise_oof import (
    _listwise_loss,
    best_positive_pair_indices,
    fixed_random_projection,
)


def test_best_positive_pair_indices_select_one_target_per_group() -> None:
    indices, groups = best_positive_pair_indices(
        np.asarray([0, 0, 1, 1, 2]),
        np.asarray([0.2, 0.8, 0.4, 0.3, 0.1]),
    )

    assert indices.tolist() == [1]
    assert groups.tolist() == [0]


def test_listwise_loss_rewards_the_best_pair_in_each_positive_group() -> None:
    good = _listwise_loss(
        torch.tensor([0.0, 3.0, 2.0, 0.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        2,
    )
    bad = _listwise_loss(
        torch.tensor([3.0, 0.0, 0.0, 2.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        2,
    )

    assert good < bad


def test_fixed_random_projection_is_deterministic_and_normalized() -> None:
    features = np.arange(24, dtype=np.float32).reshape(4, 6)
    first = fixed_random_projection(features, output_dimension=3, seed=7)
    second = fixed_random_projection(features, output_dimension=3, seed=7)

    assert np.array_equal(first, second)
    assert np.linalg.norm(first, axis=1).tolist() == pytest.approx([1.0] * 4)
