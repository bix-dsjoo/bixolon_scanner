import numpy as np

from bixolon_scanner.training.dinov3_presence_verifier import (
    cross_validated_presence_metrics,
    fit_presence_head,
    presence_probabilities,
)


def test_presence_head_probabilities_separate_training_rows() -> None:
    features = np.asarray([[-3.0, -2.0], [-2.0, -3.0], [2.0, 3.0], [3.0, 2.0]])
    labels = np.asarray([0, 0, 1, 1])

    mean, scale, coefficient, intercept = fit_presence_head(
        features, labels, regularization_c=1.0, seed=7
    )
    probabilities = presence_probabilities(features, mean, scale, coefficient, intercept)

    assert np.all(probabilities[:2] < 0.5)
    assert np.all(probabilities[2:] > 0.5)


def test_presence_cross_validation_reports_safe_separation() -> None:
    features = np.asarray(
        [
            [-4.0, -3.0],
            [2.0, 2.5],
            [3.0, 2.0],
            [-3.5, -4.0],
            [2.5, 3.0],
            [3.5, 2.5],
            [-4.5, -3.5],
            [2.2, 3.2],
            [3.2, 2.2],
        ]
    )
    labels = np.asarray([0, 1, 1, 0, 1, 1, 0, 1, 1])
    folds = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])

    metrics, probabilities = cross_validated_presence_metrics(
        features,
        labels,
        folds,
        regularization_c=1.0,
        seed=7,
    )

    assert metrics["empty_correct_count"] == 3
    assert metrics["nonempty_correct_count"] == 6
    assert metrics["probability_separation_margin"] > 0.0
    assert probabilities.shape == (9,)
