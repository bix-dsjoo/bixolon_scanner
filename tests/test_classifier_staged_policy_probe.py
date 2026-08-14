from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.archive.bread_1_0_0.classifier_staged_policy_probe import (
    select_final_policy,
    select_staged_policy,
)


def _perfect_logits(targets: np.ndarray, offset: float) -> np.ndarray:
    values = np.full((len(targets), 4), -2.0, dtype=np.float32)
    values[np.arange(len(targets)), targets] = 5.0 + offset
    return values


def test_staged_policy_uses_absolute_confidence_and_preserves_gates() -> None:
    targets = np.arange(100, dtype=np.int64) % 4
    logits = {
        "base": _perfect_logits(targets, 0.0),
        "hflip": _perfect_logits(targets, 0.1),
        "rot180": _perfect_logits(targets, 0.2),
    }
    final = select_final_policy(
        logits,
        targets,
        ground_truth_count=100,
        coverage=0.85,
        maximum_views=2,
    )
    staged = select_staged_policy(
        logits,
        targets,
        np.repeat(np.arange(20), 5),
        final_policy=final,
        ground_truth_count=100,
        minimum_coverage=0.85,
    )

    assert final["passes"] is True
    assert staged["metrics"]["overall_recognition_rate"] == 1.0
    assert staged["metrics"]["approved_misrecognition_rate"] == 0.0
    assert staged["mean_view_inferences_per_roi"] == 1.0
