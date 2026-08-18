import numpy as np

from bixolon_scanner.experiments.bread.classifier_normalized_margin import (
    apply_normalized_margin_thresholds,
    l2_normalized_logit_margin,
    select_normalized_margin_policy,
)


def test_l2_normalized_margin_is_invariant_to_positive_logit_scale() -> None:
    scores = np.asarray([[3.0, 2.0, -1.0], [30.0, 20.0, -10.0]])

    confidence = l2_normalized_logit_margin(scores)

    np.testing.assert_allclose(confidence[0], confidence[1])


def test_normalized_margin_threshold_boundary_is_approved() -> None:
    scores = np.asarray([[3.0, 2.0, -1.0]])
    threshold = float(l2_normalized_logit_margin(scores)[0])

    rejected = apply_normalized_margin_thresholds(scores, [threshold, None, None])

    assert rejected.tolist() == [False]


def test_policy_rejects_error_with_minimum_unknowns() -> None:
    scores = np.asarray(
        [
            [3.0, 2.9, 0.0],
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
        ]
    )
    targets = np.asarray([1, 0, 1])

    policy = select_normalized_margin_policy(
        scores,
        targets,
        maximum_approved_errors=0,
        maximum_unknown_top3_misses=0,
    )

    assert policy["approved_error_count"] == 0
    assert policy["unknown_count"] == 1
