from __future__ import annotations

import numpy as np

from bixolon_scanner.training.calibration import (
    binomial_rate_upper_bound,
    select_approval_threshold,
    softmax,
)
from bixolon_scanner.training.evaluate import evaluate_predictions


def test_threshold_maximizes_coverage_under_false_approval_limit():
    logits = np.asarray([[8, 0], [7, 0], [0.1, 0.2], [0.3, 0.2]], dtype=float)
    targets = np.asarray([0, 0, 1, 1])
    probabilities = softmax(logits)
    result = select_approval_threshold(
        probabilities, targets, max_false_approval_rate=0.0, confidence_level=None
    )
    assert result.approved_count == 3
    assert result.approved_precision == 1.0


def test_risk_control_uses_one_sided_binomial_upper_bound():
    assert binomial_rate_upper_bound(0, 1000) < 0.005
    assert binomial_rate_upper_bound(0, 100) > 0.005


def test_evaluation_combines_oof_archives(tmp_path):
    first = tmp_path / "fold0.npz"
    second = tmp_path / "fold1.npz"
    np.savez(first, logits=np.asarray([[8.0, 0.0], [0.0, 8.0]]), targets=np.asarray([0, 1]))
    np.savez(second, logits=np.asarray([[7.0, 0.0], [0.0, 7.0]]), targets=np.asarray([0, 1]))
    report = evaluate_predictions([first, second])
    assert report["sample_count"] == 4
    assert report["approved_precision"] == 1.0
