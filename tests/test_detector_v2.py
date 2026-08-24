from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts.runtime_package_v2 import DetectorAmbiguityPolicyMetadata
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.detector_v2 import (
    CrossScaleOnnxDetector,
    FixedEnsembleOnnxDetector,
)


def test_cross_scale_agreement_uses_complete_bipartite_matching() -> None:
    primary = [
        Detection(0, 0, 10, 10, 1.0),
        Detection(8, 0, 18, 10, 1.0),
    ]
    recovery = [
        Detection(0, 0, 12, 10, 1.0),
        Detection(8, 0, 18, 10, 1.0),
    ]

    assert CrossScaleOnnxDetector._fully_agree(primary, recovery, 0.5)
    assert not CrossScaleOnnxDetector._fully_agree(primary, recovery[:1], 0.5)


def _selected(boxes: list[list[float]]) -> dict:
    return {
        "boxes_xyxy": boxes,
        "scores": [0.9] * len(boxes),
        "class_ids": [0] * len(boxes),
    }


def test_selective_ambiguity_policy_uses_geometry_and_dense_consensus() -> None:
    policy = DetectorAmbiguityPolicyMetadata(
        mode="selective",
        high_aspect_ratio_minimum=1.9,
        dense_selected_count_minimum=6,
        dense_selected_count_maximum=6,
        dense_agreement_count_minimum=4,
        dense_aspect_ratio_minimum=1.5,
    )
    ordinary = _selected([[index * 20, 0, index * 20 + 10, 10] for index in range(5)])
    elongated = _selected([[0, 0, 20, 10]])
    dense = _selected([[index * 20, 0, index * 20 + 16, 10] for index in range(6)])

    assert not FixedEnsembleOnnxDetector._selective_uncertainty(ordinary, 4, True, policy)
    assert FixedEnsembleOnnxDetector._selective_uncertainty(elongated, 4, True, policy)
    assert FixedEnsembleOnnxDetector._selective_uncertainty(dense, 4, True, policy)
    assert not FixedEnsembleOnnxDetector._selective_uncertainty(dense, 3, True, policy)
    assert not FixedEnsembleOnnxDetector._selective_uncertainty(elongated, 4, False, policy)


def test_ambiguity_policy_rejects_inverted_dense_count_range() -> None:
    with pytest.raises(ValidationError):
        DetectorAmbiguityPolicyMetadata(
            dense_selected_count_minimum=7,
            dense_selected_count_maximum=6,
        )


def test_selective_ambiguity_can_escalate_low_agreement_elongated_scene() -> None:
    policy = DetectorAmbiguityPolicyMetadata(
        mode="selective",
        low_agreement_count_maximum=1,
        low_agreement_aspect_ratio_minimum=2.1,
    )
    elongated = _selected([[0, 0, 21, 10]])

    assert FixedEnsembleOnnxDetector._selective_uncertainty(elongated, 1, False, policy)
    assert not FixedEnsembleOnnxDetector._selective_uncertainty(elongated, 2, False, policy)


@pytest.mark.parametrize(
    "payload",
    (
        {"low_agreement_count_maximum": 1},
        {"low_agreement_aspect_ratio_minimum": 2.1},
    ),
)
def test_ambiguity_policy_requires_complete_low_agreement_pair(payload: dict) -> None:
    with pytest.raises(ValidationError):
        DetectorAmbiguityPolicyMetadata(**payload)


@pytest.mark.parametrize(
    ("scores", "expected"),
    (
        ([0.95] * 5, False),
        ([0.49, 0.9, 0.9, 0.9, 0.9, 0.9], True),
        ([0.88] * 6, True),
        ([0.7] * 6, False),
    ),
)
def test_cascade_score_bands_limit_secondary_execution(scores, expected: bool) -> None:
    cascade = SimpleNamespace(
        secondary_trigger_selected_counts=[6],
        secondary_trigger_minimum_score_maximum=0.5,
        secondary_trigger_minimum_score_minimum=0.87,
        secondary_trigger_on_uncertain=True,
    )

    assert (
        FixedEnsembleOnnxDetector._cascade_requires_secondary({"scores": scores}, cascade)
        is expected
    )


def test_cascade_runs_secondary_for_uncertain_trigger_count() -> None:
    cascade = SimpleNamespace(
        secondary_trigger_selected_counts=[6],
        secondary_trigger_minimum_score_maximum=0.5,
        secondary_trigger_minimum_score_minimum=0.87,
        secondary_trigger_on_uncertain=True,
    )

    assert FixedEnsembleOnnxDetector._cascade_requires_secondary(
        {"scores": [0.7] * 6}, cascade, uncertain=True
    )
