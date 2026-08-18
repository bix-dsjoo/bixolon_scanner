import numpy as np

from bixolon_scanner.experiments.bread.detector_acceptance_selector import (
    recapture_at_threshold,
)


def test_acceptance_selector_preserves_baseline_and_adds_probability_gate() -> None:
    probabilities = np.asarray([0.1, 0.7, 0.9])
    baseline = np.asarray([True, False, False])

    assert recapture_at_threshold(probabilities, baseline, threshold=0.8).tolist() == [
        True,
        False,
        True,
    ]
