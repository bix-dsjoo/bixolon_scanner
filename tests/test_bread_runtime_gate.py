from bixolon_scanner.contracts import (
    BoundingBox,
    ItemStatus,
    Prediction,
    ScanItem,
    ScanResponse,
    Status,
)
from bixolon_scanner.evaluation.bread_runtime_gate import (
    RuntimeGateCounts,
    _decision_trace,
    build_runtime_gate_metrics,
)


def test_runtime_gate_keeps_recaptured_and_missed_gt_in_approved_denominator() -> None:
    metrics = build_runtime_gate_metrics(
        RuntimeGateCounts(
            image_count=10,
            segmentation_image_count=9,
            image_recapture_count=1,
            judgeable_ground_truth_object_count=100,
            matched_segmentation_count=98,
            false_negative_count=1,
            false_negative_image_count=1,
            approved_count=97,
            approved_misrecognition_count=1,
            unknown_count=1,
            unknown_top3_candidate_out_count=1,
        )
    )

    assert metrics["rates"]["end_to_end_approved_object_rate"] == 0.97
    assert metrics["rates"]["approved_object_misrecognition_rate"] == 0.01
    assert metrics["rates"]["unknown_top3_candidate_out_rate"] == 0.01
    assert metrics["rates"]["segmentation_image_false_negative_rate"] == 1 / 9
    assert not metrics["final_end_to_end_approved_goal_met"]


def test_runtime_gate_boundaries_are_inclusive() -> None:
    metrics = build_runtime_gate_metrics(
        RuntimeGateCounts(
            image_count=1000,
            segmentation_image_count=900,
            judgeable_ground_truth_object_count=1000,
            approved_count=990,
            approved_misrecognition_count=1,
            unknown_top3_candidate_out_count=1,
        )
    )

    assert metrics["operational_gates"]["segmentation_image_rate"]
    assert metrics["operational_gates"]["approved_object_misrecognition_rate"]
    assert metrics["operational_gates"]["unknown_top3_candidate_out_rate"]
    assert metrics["final_end_to_end_approved_goal_met"]


def test_decision_trace_excludes_request_specific_values() -> None:
    response = ScanResponse(
        request_id="request-123",
        status=Status.SEGMENTATION,
        segmentations=[
            ScanItem(
                segmentation_id="segmentation_001",
                bbox=BoundingBox(x=1, y=2, width=3, height=4),
                status=ItemStatus.APPROVED,
                prediction=Prediction(class_id="bread_01", class_name="bread_01"),
                confidence=0.99,
            )
        ],
        processing_time_ms=12.3,
        worker_version="1.1.0",
        detector_version="1.1.0",
        classifier_version="1.1.0",
    )

    trace = _decision_trace(7, response)

    assert trace["image_id"] == 7
    assert "request_id" not in trace
    assert "processing_time_ms" not in trace
    assert trace["segmentations"][0]["prediction"]["class_id"] == "bread_01"
