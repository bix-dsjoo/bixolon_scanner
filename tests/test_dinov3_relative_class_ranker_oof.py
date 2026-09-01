import numpy as np

from bixolon_scanner.experiments.bread.dinov3_relative_class_ranker_oof import (
    best_pair_labels,
    relative_pair_features,
)


def test_relative_features_rank_and_measure_same_class_consensus() -> None:
    base = np.asarray([[[1.0]], [[3.0]], [[2.0]]])
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [30.0, 30.0, 40.0, 40.0]])
    candidates = np.asarray([[4], [4], [4]])

    features = relative_pair_features(
        base,
        boxes,
        np.asarray([1, 1, 1]),
        candidates,
        np.asarray([[0.5], [0.9], [0.8]]),
        np.asarray([[0.5], [0.8], [0.7]]),
    )

    assert features.shape == (3, 1, 32)
    assert features[1, 0, 2] == 1.0
    assert features[0, 0, 9] > features[2, 0, 9]


def test_best_pair_labels_marks_one_covered_pair_per_group() -> None:
    labels = best_pair_labels(np.asarray([10, 10, 11, 11]), np.asarray([0.6, 0.8, 0.4, 0.3]))

    assert labels.tolist() == [False, True, False, False]
