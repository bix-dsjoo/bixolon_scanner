from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import bixolon_scanner.runtime.detector_v2 as detector_v2_runtime
from bixolon_scanner.contracts.runtime_package_v2 import (
    DetectorAmbiguityPolicyMetadata,
    DetectorCrowdingPolicyMetadata,
)
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.detector_v2 import (
    CrossScaleOnnxDetector,
    FixedEnsembleOnnxDetector,
)
from bixolon_scanner.runtime.onnx import detector_crowding_requires_recapture


def _crowding_policy() -> DetectorCrowdingPolicyMetadata:
    return DetectorCrowdingPolicyMetadata(
        minimum_image_aspect_ratio=1.0,
        candidate_score_threshold=0.05,
        large_proposal_score_threshold=0.145,
        large_proposal_minimum_area_ratio=0.21,
        large_proposal_corroboration={
            "query_containment_surplus_minimum": 1,
            "selected_center_minimum": 2,
            "selected_count_maximum": 5,
        },
        proximity_maximum_normalized_center_distance=0.48,
        query_cluster_iou_threshold=0.7,
        query_duplicate_minimum_fraction=0.93,
        rotation_recovery_degrees=[90, 180],
        rotation_recovery_minimum_selected_count=5,
        rotation_recovery_maximum_selected_count=5,
        rotation_recovery_minimum_selected_area_fraction=0.4,
        rotation_recovery_minimum_normalized_center_distance=0.63,
        rotation_recovery_minimum_count_gain=1,
        rotation_recovery_agreement_iou_threshold=0.5,
    )


def _duplicates(box: tuple[float, float, float, float], count: int) -> list[Detection]:
    return [Detection(*box, 0.2 - index * 0.001) for index in range(count)]


def test_crowding_policy_allows_large_single_object_without_corroboration() -> None:
    candidates = [Detection(0, 0, 46, 46, 0.2)]

    assert not detector_crowding_requires_recapture(
        candidates,
        selected_detections=candidates,
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )


def test_legacy_crowding_policy_keeps_unconditional_large_proposal_gate() -> None:
    payload = _crowding_policy().model_dump(mode="json")
    payload["large_proposal_corroboration"] = None

    assert detector_crowding_requires_recapture(
        [Detection(0, 0, 46, 46, 0.2)],
        selected_detections=[Detection(0, 0, 46, 46, 0.2)],
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=DetectorCrowdingPolicyMetadata.model_validate(payload),
    )


def test_crowding_policy_recaptures_large_proposal_with_contained_query_surplus() -> None:
    large = Detection(0, 0, 60, 60, 0.2)
    contained = Detection(5, 5, 20, 20, 0.19)

    assert detector_crowding_requires_recapture(
        [large, contained],
        selected_detections=[large],
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )


def test_crowding_policy_recaptures_large_proposal_covering_sparse_selected_centers() -> None:
    large = Detection(0, 0, 60, 60, 0.2)
    selected = [
        Detection(0, 0, 20, 20, 0.9),
        Detection(40, 40, 70, 70, 0.8),
    ]

    assert detector_crowding_requires_recapture(
        [large],
        selected_detections=selected,
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )


def test_crowding_policy_allows_large_proposal_in_dense_selected_layout() -> None:
    large = Detection(0, 0, 60, 60, 0.2)
    selected = [
        Detection(index * 10, index * 10, index * 10 + 20, index * 10 + 20, 0.9)
        for index in range(6)
    ]

    assert not detector_crowding_requires_recapture(
        [large],
        selected_detections=selected,
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )


def test_crowding_policy_requires_proximity_and_query_duplication_together() -> None:
    close_boxes = _duplicates((10, 10, 30, 30), 15) + _duplicates((19, 10, 39, 30), 15)
    low_duplicate_boxes = [
        Detection(10, 10, 30, 30, 0.2),
        Detection(19, 10, 39, 30, 0.19),
    ]
    separated_boxes = _duplicates((10, 10, 30, 30), 15) + _duplicates((70, 70, 90, 90), 15)

    assert detector_crowding_requires_recapture(
        close_boxes,
        selected_detections=[],
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )
    assert not detector_crowding_requires_recapture(
        low_duplicate_boxes,
        selected_detections=[],
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )
    assert not detector_crowding_requires_recapture(
        separated_boxes,
        selected_detections=[],
        image_width=100,
        image_height=100,
        nms_iou_threshold=0.4,
        policy=_crowding_policy(),
    )


def test_crowding_policy_rejects_inverted_score_thresholds() -> None:
    with pytest.raises(ValidationError):
        DetectorCrowdingPolicyMetadata(
            candidate_score_threshold=0.2,
            large_proposal_score_threshold=0.1,
            large_proposal_minimum_area_ratio=0.21,
            proximity_maximum_normalized_center_distance=0.48,
            query_cluster_iou_threshold=0.7,
            query_duplicate_minimum_fraction=0.93,
        )


def test_crowding_policy_rejects_duplicate_recovery_rotations() -> None:
    with pytest.raises(ValidationError):
        DetectorCrowdingPolicyMetadata(
            candidate_score_threshold=0.05,
            large_proposal_score_threshold=0.145,
            large_proposal_minimum_area_ratio=0.21,
            proximity_maximum_normalized_center_distance=0.48,
            query_cluster_iou_threshold=0.7,
            query_duplicate_minimum_fraction=0.93,
            rotation_recovery_degrees=[90, 90],
        )


def test_crowding_policy_rejects_inverted_recovery_count_range() -> None:
    with pytest.raises(ValidationError):
        DetectorCrowdingPolicyMetadata(
            candidate_score_threshold=0.05,
            large_proposal_score_threshold=0.145,
            large_proposal_minimum_area_ratio=0.21,
            proximity_maximum_normalized_center_distance=0.48,
            query_cluster_iou_threshold=0.7,
            query_duplicate_minimum_fraction=0.93,
            rotation_recovery_minimum_selected_count=6,
            rotation_recovery_maximum_selected_count=5,
        )


@pytest.mark.parametrize(("provider", "expected_cuda_graph"), (("cpu", False), ("cuda", True)))
def test_single_detector_builder_passes_runtime_policy(
    monkeypatch, tmp_path, provider: str, expected_cuda_graph: bool
) -> None:
    policy = _crowding_policy()
    captured = {}

    class FakeOnnxDetector:
        def __init__(self, *args, crowding_policy=None, **kwargs) -> None:
            captured["policy"] = crowding_policy
            captured["enable_cuda_graph"] = kwargs["enable_cuda_graph"]
            captured["detector_class_count"] = kwargs["detector_class_count"]

    monkeypatch.setattr(detector_v2_runtime, "OnnxDetector", FakeOnnxDetector)
    package = SimpleNamespace(
        detector_path=tmp_path / "detector.onnx",
        metadata=SimpleNamespace(
            detector=SimpleNamespace(ensemble=None),
            detector_class_count=1,
            detector_refinement=None,
            detector_crowding=policy,
            count_verifier=None,
        ),
    )

    detector_v2_runtime.build_detector_v2(package, provider)

    assert captured["policy"] is policy
    assert captured["enable_cuda_graph"] is expected_cuda_graph
    assert captured["detector_class_count"] == 1


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
