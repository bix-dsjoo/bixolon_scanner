import numpy as np

from bixolon_scanner.command_registry import DIAGNOSTIC_COMMANDS
from bixolon_scanner.experiments.bread.dinov3_selective_e2e import (
    aggregate_catalog_heads,
    class_consensus_top_k,
    greedy_top_k_nms,
)


def test_greedy_top_k_nms_keeps_spatially_distinct_scores() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
    selected = greedy_top_k_nms(
        boxes,
        np.asarray([0.9, 0.8, 0.7]),
        count=2,
        iou_threshold=0.5,
    )

    assert selected.tolist() == [0, 2]


def test_greedy_top_k_nms_can_require_unique_catalog_classes() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]])
    selected = greedy_top_k_nms(
        boxes,
        np.asarray([0.9, 0.8, 0.7]),
        count=2,
        iou_threshold=0.5,
        classes=np.asarray([1, 1, 2]),
        suppression_mode="class_unique",
    )

    assert selected.tolist() == [0, 2]


def test_aggregate_catalog_heads_uses_overlap_vote() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
    logits, retrieval, stability = aggregate_catalog_heads(
        boxes,
        np.asarray([0, 2]),
        np.asarray([[2.0, 0.0], [3.0, 0.0], [0.0, 4.0]]),
        np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]]),
        np.asarray([1.0, 1.0, 1.0]),
        minimum_iou=0.5,
    )

    assert np.argmax(logits, axis=1).tolist() == [0, 1]
    assert np.argmax(retrieval, axis=1).tolist() == [0, 1]
    assert stability.tolist() == [1.0, 1.0]


def test_class_consensus_selects_repeated_distinct_sku_evidence() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    selected = class_consensus_top_k(
        boxes,
        np.asarray([0.9, 0.8, 0.7, 0.6]),
        np.asarray([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]),
        count=2,
        iou_threshold=0.5,
        temperature=0.1,
        candidate_limit=4,
    )

    assert selected.tolist() == [0, 2]


def test_selective_e2e_commands_are_diagnostic_only() -> None:
    assert DIAGNOSTIC_COMMANDS[("experiment", "bread-dinov3-proposal-ranker-oof")] == (
        "bixolon_scanner.experiments.bread.dinov3_proposal_ranker_oof",
        "main",
    )
    assert DIAGNOSTIC_COMMANDS[("experiment", "bread-dinov3-selective-e2e")] == (
        "bixolon_scanner.experiments.bread.dinov3_selective_e2e",
        "main",
    )
