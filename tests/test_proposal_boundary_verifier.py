import numpy as np

from bixolon_scanner.experiments.bread.proposal_boundary_verifier import (
    boundary_context_features,
    find_candidate_index,
)


def test_find_candidate_index_uses_box_and_score_alignment():
    prediction = {
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8],
    }

    assert find_candidate_index(prediction, [20, 0, 30, 10], 0.8) == 1


def test_boundary_context_features_are_fixed_width_and_finite():
    features = boundary_context_features([0.9, 0.7, 0.4], 1, length=4)

    assert features.shape == (10,)
    assert np.isfinite(features).all()
