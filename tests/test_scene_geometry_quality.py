from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.training.scene_geometry_quality import (
    extract_scene_geometry_features,
)


def test_scene_geometry_features_capture_overlap_and_are_resolution_invariant() -> None:
    record = {
        "width": 100,
        "height": 200,
        "annotations": [
            {"bbox_xywh": [10, 20, 40, 80]},
            {"bbox_xywh": [30, 60, 40, 80]},
        ],
    }
    scaled = {
        "width": 200,
        "height": 400,
        "annotations": [
            {"bbox_xywh": [20, 40, 80, 160]},
            {"bbox_xywh": [60, 120, 80, 160]},
        ],
    }

    features = extract_scene_geometry_features(record, 0)
    scaled_features = extract_scene_geometry_features(scaled, 0)

    assert features.shape == (14,)
    np.testing.assert_allclose(features, scaled_features, atol=1e-6)
    assert features[5] == pytest.approx(0.25)


def test_scene_geometry_features_reject_invalid_index() -> None:
    with pytest.raises(IndexError):
        extract_scene_geometry_features({"width": 1, "height": 1, "annotations": []}, 0)
