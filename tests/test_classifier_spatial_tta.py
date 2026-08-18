from __future__ import annotations

import pytest

from bixolon_scanner.experiments.bread.classifier_spatial_tta import spatial_crop_bounds


def test_spatial_crop_bounds_cover_opposite_corners() -> None:
    assert spatial_crop_bounds(100, 200, scale=0.5, x_position=-1, y_position=-1) == (
        0,
        0,
        50,
        100,
    )
    assert spatial_crop_bounds(100, 200, scale=0.5, x_position=1, y_position=1) == (
        50,
        100,
        50,
        100,
    )


def test_spatial_crop_bounds_reject_unknown_position() -> None:
    with pytest.raises(ValueError, match="positions"):
        spatial_crop_bounds(224, 224, scale=0.75, x_position=2, y_position=0)
