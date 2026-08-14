from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.archive.bread_1_0_0.pretrained_quality_probe import (
    lower_tail_threshold,
)


def test_lower_tail_threshold_respects_zero_false_budget() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4])

    threshold = lower_tail_threshold(values, 0.01)

    assert np.count_nonzero(values <= threshold) == 0


def test_lower_tail_threshold_uses_available_false_budget() -> None:
    values = np.arange(100, dtype=np.float64)

    threshold = lower_tail_threshold(values, 0.01)

    assert threshold == 0.0
    assert np.count_nonzero(values <= threshold) == 1


def test_lower_tail_threshold_does_not_exceed_budget_on_ties() -> None:
    values = np.asarray([0.0] * 10 + [1.0] * 90)

    threshold = lower_tail_threshold(values, 0.01)

    assert np.count_nonzero(values <= threshold) == 0
