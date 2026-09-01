from bixolon_scanner.runtime.proposal_selection import (
    ClassAssistedSelectionPolicy,
    count_and_class_assisted_select,
    select_single_object_recovery,
)


def _entry(index, box, score, class_id, approval=0.9, support=4):
    return {
        "proposal_index": index,
        "box": box,
        "detector_score": score,
        "support_count": support,
        "predicted_class": class_id,
        "approval_score": approval,
    }


def test_count_assisted_selection_adds_independent_supported_proposal() -> None:
    base = {"boxes_xyxy": [[0, 0, 10, 10]], "scores": [0.9], "class_ids": [0]}
    raw = {
        "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
        "scores": [0.9, 0.4],
        "support_counts": [4, 4],
    }
    selected, diagnostics = count_and_class_assisted_select(
        base,
        raw,
        [_entry(0, [0, 0, 10, 10], 0.9, 1), _entry(1, [20, 20, 30, 30], 0.4, 2)],
        2,
    )

    assert len(selected["boxes_xyxy"]) == 2
    assert selected["proposal_indices"] == [0, 1]
    assert diagnostics["selected_count"] == 2


def test_single_object_recovery_prefers_largest_compact_proposal() -> None:
    selected = select_single_object_recovery(
        [
            _entry(0, [0, 0, 30, 10], 0.05, 1, approval=0.9),
            _entry(1, [0, 0, 18, 18], 0.04, 2, approval=0.4),
            _entry(2, [0, 0, 12, 12], 0.08, 3, approval=0.8),
        ],
        minimum_score=0.03,
        minimum_support=3,
        maximum_aspect_ratio=2.0,
        minimum_approval_score=0.1,
    )

    assert selected is not None
    assert selected["proposal_indices"] == [1]


def test_localization_refinement_replaces_containing_merge() -> None:
    base = {
        "boxes_xyxy": [[0, 0, 20, 10], [0, 0, 9, 10]],
        "scores": [0.9, 0.85],
        "class_ids": [0, 0],
    }
    raw = {
        "boxes_xyxy": [[0, 0, 20, 10], [0, 0, 9, 10], [11, 0, 20, 10]],
        "scores": [0.9, 0.85, 0.4],
        "support_counts": [4, 4, 4],
    }
    entries = [
        _entry(0, [0, 0, 20, 10], 0.9, 1),
        _entry(1, [0, 0, 9, 10], 0.85, 2),
        _entry(2, [11, 0, 20, 10], 0.4, 1, support=3),
    ]
    policy = ClassAssistedSelectionPolicy(refinement_minimum_separation_gain=0.2)

    selected, diagnostics = count_and_class_assisted_select(base, raw, entries, 2, policy)

    assert selected["proposal_indices"] == [1, 2]
    assert diagnostics["localization_refined"] is True


def test_single_object_expansion_replaces_an_internal_fragment() -> None:
    base = {"boxes_xyxy": [[4, 0, 10, 5]], "scores": [0.9], "class_ids": [0]}
    raw = {
        "boxes_xyxy": [[4, 0, 10, 5], [0, 0.2, 12, 10]],
        "scores": [0.9, 0.04],
        "support_counts": [4, 3],
    }
    entries = [
        _entry(0, raw["boxes_xyxy"][0], 0.9, 2, approval=0.4),
        _entry(1, raw["boxes_xyxy"][1], 0.04, 2, approval=0.8, support=3),
    ]
    policy = ClassAssistedSelectionPolicy(
        single_object_expansion_minimum_containment=0.9,
        single_object_expansion_minimum_area_ratio=2.0,
        single_object_expansion_minimum_approval_score=0.7,
        single_object_expansion_minimum_approval_margin=0.2,
    )

    selected, diagnostics = count_and_class_assisted_select(base, raw, entries, 1, policy)

    assert selected["proposal_indices"] == [1]
    assert diagnostics["localization_refined"] is True


def test_single_object_expansion_preserves_base_without_approval_margin() -> None:
    base = {"boxes_xyxy": [[4, 0, 10, 5]], "scores": [0.9], "class_ids": [0]}
    raw = {
        "boxes_xyxy": [[4, 0, 10, 5], [0, 0.2, 12, 10]],
        "scores": [0.9, 0.04],
        "support_counts": [4, 3],
    }
    entries = [
        _entry(0, raw["boxes_xyxy"][0], 0.9, 2, approval=0.7),
        _entry(1, raw["boxes_xyxy"][1], 0.04, 2, approval=0.8, support=3),
    ]
    policy = ClassAssistedSelectionPolicy(
        single_object_expansion_minimum_containment=0.9,
        single_object_expansion_minimum_area_ratio=2.0,
        single_object_expansion_minimum_approval_score=0.7,
        single_object_expansion_minimum_approval_margin=0.2,
    )

    selected, diagnostics = count_and_class_assisted_select(base, raw, entries, 1, policy)

    assert selected["proposal_indices"] == [0]
    assert diagnostics["localization_refined"] is False


def test_count_reduction_does_not_readd_an_independent_proposal() -> None:
    base = {
        "boxes_xyxy": [
            [0, 0, 10, 10],
            [20, 0, 30, 10],
            [40, 0, 50, 10],
            [41, 0, 51, 10],
        ],
        "scores": [0.9, 0.85, 0.8, 0.7],
        "class_ids": [0, 0, 0, 0],
    }
    raw = {
        "boxes_xyxy": [*base["boxes_xyxy"], [60, 0, 70, 10]],
        "scores": [*base["scores"], 0.6],
        "support_counts": [4, 4, 4, 4, 4],
    }
    entries = [
        _entry(0, raw["boxes_xyxy"][0], 0.9, 1),
        _entry(1, raw["boxes_xyxy"][1], 0.85, 2),
        _entry(2, raw["boxes_xyxy"][2], 0.8, 3),
        _entry(3, raw["boxes_xyxy"][3], 0.7, 3),
        _entry(4, raw["boxes_xyxy"][4], 0.6, 4),
    ]

    selected, diagnostics = count_and_class_assisted_select(base, raw, entries, 3)

    assert selected["proposal_indices"] == [0, 1, 2]
    assert diagnostics["selected_count"] == 3
