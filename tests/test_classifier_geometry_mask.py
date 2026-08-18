from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.classifier_geometry_mask import (
    apply_background_mask,
    neighbor_ownership_mask,
)


def test_neighbor_mask_removes_only_neighbor_owned_side() -> None:
    mask = neighbor_ownership_mask(
        image_width=100,
        image_height=100,
        boxes=((20.0, 20.0, 80.0, 80.0), (0.0, 20.0, 50.0, 80.0)),
        target_index=0,
        output_size=20,
        margin_ratio=0.0,
        distance_bias=0.0,
        shared_scale=True,
    )

    assert mask[:, :5].any()
    assert not mask[:, -5:].any()


def test_background_mask_uses_each_roi_border_median() -> None:
    tensors = np.ones((2, 3, 4, 4), dtype=np.float32)
    tensors[1] *= 2.0
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[:, 1:3, 1:3] = True

    output = apply_background_mask(tensors, masks)

    assert np.all(output[0, :, 1:3, 1:3] == 1.0)
    assert np.all(output[1, :, 1:3, 1:3] == 2.0)
