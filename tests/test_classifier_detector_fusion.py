from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.evaluation.classifier_detector_fusion import (
    cross_validated_selection,
    fusion_candidates,
)


def test_fusion_candidates_validate_shapes() -> None:
    with pytest.raises(ValueError, match="matching"):
        fusion_candidates(np.zeros((2, 3)), np.zeros((3, 3)))


def test_cross_validated_selection_uses_other_folds() -> None:
    targets = np.asarray([0, 1, 0, 1, 0, 1])
    folds = np.asarray([0, 0, 1, 1, 2, 2])
    candidates = {
        "always_wrong": 1 - targets,
        "always_right": targets.copy(),
    }

    result = cross_validated_selection(candidates, targets, folds)

    assert result["top1_accuracy"] == 1.0
    assert {row["selected"] for row in result["folds"]} == {"always_right"}


def test_margin_gate_can_use_detector_for_uncertain_classifier() -> None:
    dino = np.asarray([[0.1, 0.11, -1.0], [2.0, 0.0, -1.0]], dtype=np.float32)
    detector = np.asarray([[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=np.float32)

    candidates = fusion_candidates(dino, detector)

    assert any(np.array_equal(values, np.asarray([0, 0])) for values in candidates.values())
