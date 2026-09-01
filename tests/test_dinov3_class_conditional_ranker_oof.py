import numpy as np

from bixolon_scanner.experiments.bread.dinov3_class_conditional_ranker_oof import (
    candidate_classes,
    class_conditional_targets,
    pair_features,
)


def test_class_conditional_targets_only_score_the_requested_sku() -> None:
    targets = class_conditional_targets(
        np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        np.asarray([[0, 1], [0, 1]]),
        [
            {"category_id": 1, "bbox_xywh": [0.0, 0.0, 10.0, 10.0]},
            {"category_id": 2, "bbox_xywh": [20.0, 20.0, 10.0, 10.0]},
        ],
    )

    assert targets.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_pair_features_are_class_conditional_without_class_identity_one_hot() -> None:
    logits = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    candidates = candidate_classes(logits, top_k=2)
    features = pair_features(
        np.zeros((1, 8), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        logits,
        np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32),
        candidates,
        np.asarray([0.5]),
        np.asarray([False]),
        np.asarray([False]),
    )

    assert candidates.tolist() == [[0, 1]]
    assert features.shape == (1, 2, 35)
    assert features[0, 0, -2:].tolist() == [1.0, 0.0]
    assert features[0, 1, -2:].tolist() == [0.0, 1.0]
