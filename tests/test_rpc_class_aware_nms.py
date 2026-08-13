from __future__ import annotations

from bixolon_scanner.training.rpc_class_aware_nms import (
    _assignment_conflict_mask,
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
        overlap_max_quality=None,
        low_quality_multiplier=1.118,
        low_quality_max_quality=None,
        low_quality_min_score=None,
        minimum_rank=0.85,
    )

    assert overlap.tolist() == [True, False, False, False]
    assert low_quality.tolist() == [False, False, True, False]
    assert combined.tolist() == [True, False, True, False]


def test_duplicate_overlap_can_be_limited_to_low_context_quality() -> None:
    combined, overlap, _low_quality = _duplicate_ambiguity_mask(
        containment=[0.5, 0.5],
        repeated=[True, True],
        ranks=[0.9, 0.9],
        scores=[0.8, 0.8],
        quality=[0.19, 0.20],
        context_threshold=0.005,
        overlap_threshold=0.45,
        overlap_max_score=0.87,
        overlap_max_quality=0.194,
        low_quality_multiplier=None,
        low_quality_max_quality=None,
        low_quality_min_score=None,
        minimum_rank=0.85,
    )

    assert overlap.tolist() == [True, False]
    assert combined.tolist() == [True, False]


def test_assignment_conflict_detects_mutual_swap_and_ranked_duplicate() -> None:
    combined, mutual, duplicate = _assignment_conflict_mask(
        image_ids=[1, 1, 2, 2, 2, 3, 3],
        top_classes=[
            [10, 20],
            [20, 10],
            [30, 99],
            [30, 40],
            [40, 50],
            [60, 70],
            [60, 80],
        ],
        detector_scores=[0.9, 0.8, 0.95, 0.7, 0.9, 0.9, 0.8],
        detector_ranks=[0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5],
        confidence=[1.0] * 7,
        quality=[1.0] * 7,
        minimum_duplicate_rank=0.85,
    )

    assert mutual.tolist() == [True, True, False, False, False, False, False]
    assert duplicate.tolist() == [False, False, False, True, False, False, False]
    assert combined.tolist() == [True, True, False, True, False, False, False]


def test_assignment_conflict_requires_top2_and_equal_lengths() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least Top-2"):
        _assignment_conflict_mask(
            [1],
            [[3]],
            [0.9],
            [0.0],
            [1.0],
            [1.0],
            minimum_duplicate_rank=0.0,
        )
    with pytest.raises(ValueError, match="equal lengths"):
        _assignment_conflict_mask(
            [1, 2],
            [[3, 4]],
            [0.9],
            [0.0],
            [1.0],
            [1.0],
            minimum_duplicate_rank=0.0,
        )


def test_assignment_mutual_thresholds_can_keep_low_quality_conflicts() -> None:
    combined, mutual, duplicate = _assignment_conflict_mask(
        [1, 1, 2, 2],
        [[10, 20], [20, 10], [30, 40], [40, 30]],
        [0.9] * 4,
        [0.0] * 4,
        [0.99995, 0.99996, 0.99995, 0.99996],
        [0.95, 0.96, 0.1, 0.2],
        minimum_duplicate_rank=0.0,
        minimum_mutual_confidence=0.9999,
        minimum_mutual_quality=0.9,
        enable_duplicate_alternative=False,
    )

    assert mutual.tolist() == [True, True, False, False]
    assert duplicate.tolist() == [False, False, False, False]
    assert combined.tolist() == [True, True, False, False]


def test_assignment_mutual_can_be_limited_to_calibrated_class_pairs() -> None:
    combined, mutual, _duplicate = _assignment_conflict_mask(
        [1, 1, 2, 2],
        [[10, 20], [20, 10], [30, 40], [40, 30]],
        [0.9] * 4,
        [0.0] * 4,
        [1.0] * 4,
        [1.0] * 4,
        minimum_duplicate_rank=0.0,
        enable_duplicate_alternative=False,
        mutual_class_pairs={(10, 20)},
    )

    assert mutual.tolist() == [True, True, False, False]
    assert combined.tolist() == [True, True, False, False]
