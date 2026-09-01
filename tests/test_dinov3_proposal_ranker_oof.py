import numpy as np

from bixolon_scanner.experiments.bread.dinov3_proposal_ranker_oof import (
    classifier_score_features,
    proposal_context_features,
)


def test_classifier_score_features_capture_head_agreement() -> None:
    features = classifier_score_features(
        np.asarray([[3.0, 1.0, 2.0]], dtype=np.float32),
        np.asarray([[0.1, 0.8, 0.2]], dtype=np.float32),
        np.asarray([0.7]),
        np.asarray([False]),
        np.asarray([False]),
    )

    assert features.shape == (1, 14)
    assert features[0, -4] == 0.0
    assert features[0, -3] == 0.7


def test_proposal_context_features_count_overlapping_same_class() -> None:
    features = proposal_context_features(
        np.asarray(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]],
            dtype=np.float32,
        ),
        np.asarray([0.8, 0.7, 0.9]),
        np.asarray([2, 2, 3]),
    )

    assert features.shape == (3, 10)
    assert features[0, 2] == 1.0
    assert features[0, 6] == 1.0
    assert features[2, 2] == 0.0
