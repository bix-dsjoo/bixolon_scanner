import numpy as np

from bixolon_scanner.experiments.bread.dinov3_nested_scene_gate import (
    scene_features,
    zero_error_threshold,
)


def test_zero_error_threshold_is_strictly_above_unsafe_scores() -> None:
    threshold, count = zero_error_threshold(
        np.asarray([0.1, 0.4, 0.8, 0.9]), np.asarray([False, True, False, True])
    )

    assert threshold > 0.8
    assert count == 1


def test_scene_features_do_not_require_ground_truth_diagnostics() -> None:
    features = scene_features(
        {
            "selectable": True,
            "minimum_ranker_safety": 0.8,
            "minimum_identity_margin": 0.2,
            "minimum_identity_stability": 0.7,
            "maximum_residual_ratio": 0.5,
            "all_heads_agree": True,
            "ranker_safety": [0.8, 0.9],
            "identity_margin": [0.2, 0.3],
            "identity_stability": [0.7, 0.8],
            "head_agreement": [True, True],
            "selected_boxes_xyxy": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        },
        {"width": 40, "height": 40},
        {"predicted_count": 2, "confidence": 0.9},
        {"predicted_count": 2, "confidence": 0.8},
    )

    assert features.shape == (81,)
    assert np.isfinite(features).all()
