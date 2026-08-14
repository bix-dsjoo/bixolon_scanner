from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.archive.bread_1_0_0.classifier_selective_risk_probe import (
    _metrics,
    _softmax,
    _top3_candidates,
)


def test_top3_candidates_return_stable_class_scores() -> None:
    logits = {
        "left": np.asarray([[5.0, 4.0, 3.0, 0.0]], dtype=np.float32),
        "right": np.asarray([[5.0, 0.0, 4.0, 3.0]], dtype=np.float32),
    }
    probabilities = {name: _softmax(values) for name, values in logits.items()}

    candidates = _top3_candidates(logits, probabilities, ("left", "right"))

    assert set(candidates) == {
        "mean_logits",
        "mean_probability",
        "maximum_probability",
        "reciprocal_rank",
        "top3_vote",
    }
    assert np.argsort(-candidates["top3_vote"], axis=1)[0, 0] == 0


def test_metrics_can_use_separate_unknown_top3_scores() -> None:
    logits = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    score = np.asarray([0.1], dtype=np.float32)
    top3_values = np.asarray([[4.0, 3.0, 1.0, 2.0]], dtype=np.float32)

    metrics = _metrics(
        logits,
        score,
        top3_values,
        np.asarray([3], dtype=np.int64),
        np.asarray([True]),
        0.5,
    )

    assert metrics["unknown_count"] == 1
    assert metrics["unknown_top3_accuracy"] == 1.0
    assert metrics["recognition_rate"] == 1.0
