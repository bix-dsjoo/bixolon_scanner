from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_detector_count_recovery.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_detector_count_recovery", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_append_recovery = _MODULE._append_recovery
_containment_fraction = _MODULE._containment_fraction


def _prediction(boxes: list[list[float]], scores: list[float]) -> dict:
    return {
        "boxes_xyxy": boxes,
        "scores": scores,
        "class_ids": [0] * len(boxes),
        "top3_class_ids": [[0]] * len(boxes),
    }


def test_containment_fraction_uses_the_smaller_box() -> None:
    outer = np.asarray([0.0, 0.0, 10.0, 10.0])
    inner = np.asarray([2.0, 2.0, 8.0, 8.0])

    assert _containment_fraction(outer, inner) == 1.0
    assert _containment_fraction(inner, outer) == 1.0


def test_append_recovery_rejects_near_contained_duplicate() -> None:
    primary = _prediction([[0.0, 0.0, 10.0, 10.0]], [0.99])
    backup = _prediction([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 7.1, 7.1]], [0.99, 0.8])

    recovered = _append_recovery(
        primary,
        backup,
        minimum_support_iou=0.48,
        maximum_support_iou=0.52,
        maximum_containment_fraction=0.9,
    )

    assert recovered is None


def test_append_recovery_keeps_partially_overlapping_distinct_hypothesis() -> None:
    primary = _prediction([[0.0, 3.0, 10.0, 10.0]], [0.99])
    backup = _prediction([[0.0, 3.0, 10.0, 10.0], [0.0, 0.0, 10.0, 7.0]], [0.99, 0.8])

    recovered = _append_recovery(
        primary,
        backup,
        minimum_support_iou=0.39,
        maximum_support_iou=0.41,
        maximum_containment_fraction=0.9,
    )

    assert recovered is not None
    assert recovered["boxes_xyxy"][-1] == [0.0, 0.0, 10.0, 7.0]
