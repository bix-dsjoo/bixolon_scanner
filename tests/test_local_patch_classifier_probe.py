from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.experiments.archive.bread_1_0_0.local_patch_classifier_probe import (
    aggregate_patch_matches,
)


def test_aggregate_patch_matches_preserves_class_axis() -> None:
    values = np.asarray(
        [
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ]
        ],
        dtype=np.float32,
    )

    result = aggregate_patch_matches(values)

    np.testing.assert_allclose(result["mean_all"], [[0.25, 0.75]])
    np.testing.assert_allclose(result["top_half"], [[0.35, 0.85]])
    np.testing.assert_allclose(result["maximum"], [[0.4, 0.9]])


def test_aggregate_patch_matches_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="patch similarities"):
        aggregate_patch_matches(np.zeros((2, 3)))
