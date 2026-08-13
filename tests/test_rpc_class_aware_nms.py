from __future__ import annotations

from bixolon_scanner.training.rpc_class_aware_nms import (
    _duplicate_ambiguity_features,
    _duplicate_ambiguity_mask,
    _keep_indices,
)


def _detection(box: list[float], score: float) -> dict[str, object]:
    return {"bbox_xyxy": box, "score": score}


def test_class_aware_nms_suppresses_only_same_class_overlap() -> None:
    detections = [
        _detection([0, 0, 100, 100], 0.95),
        _detection([10, 10, 90, 90], 0.80),
        _detection([10, 10, 90, 90], 0.70),
        _detection([200, 200, 300, 300], 0.60),
    ]

    kept = _keep_indices(detections, [4, 4, 8, 4], 0.55)

    assert kept == [0, 2, 3]


def test_class_aware_nms_threshold_is_strict() -> None:
    detections = [
        _detection([0, 0, 100, 100], 0.95),
        _detection([0, 0, 50, 100], 0.80),
    ]

    assert _keep_indices(detections, [1, 1], 0.5) == [0, 1]


def test_duplicate_ambiguity_uses_only_higher_score_same_class() -> None:
    detections = [
        _detection([0, 0, 100, 100], 0.95),
        _detection([25, 25, 75, 75], 0.80),
        _detection([25, 25, 75, 75], 0.70),
    ]

    containment, repeated = _duplicate_ambiguity_features(
        detections, [4, 4, 8]
    )

    assert containment == [0.0, 1.0, 0.0]
    assert repeated == [False, True, False]


def test_duplicate_ambiguity_mask_applies_narrow_overlap_and_quality_gates() -> None:
    combined, overlap, low_quality = _duplicate_ambiguity_mask(
        containment=[0.50, 0.50, 0.0, 0.0],
        repeated=[True, True, True, True],
        ranks=[0.90, 0.90, 0.90, 0.50],
        scores=[0.86, 0.88, 0.90, 0.80],
        quality=[1.0, 1.0, 0.005, 0.005],
        context_threshold=0.005,
        overlap_threshold=0.45,
        overlap_max_score=0.87,
        low_quality_multiplier=1.118,
        minimum_rank=0.85,
    )

    assert overlap.tolist() == [True, False, False, False]
    assert low_quality.tolist() == [False, False, True, False]
    assert combined.tolist() == [True, False, True, False]
