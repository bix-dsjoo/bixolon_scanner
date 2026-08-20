from __future__ import annotations

import pytest

from bixolon_scanner.contracts import BoundingBox, ItemStatus, Prediction, ScanItem, Status
from bixolon_scanner.evaluation.scanner_v2_private import (
    AnnotationOutcome,
    ImageOutcome,
    aggregate_metrics,
    claim_private_run,
    error_gate,
    one_sided_error_upper_95,
    one_sided_success_lower_95,
    success_gate,
)
from bixolon_scanner.evaluation.scanner_v2_private_preflight import PrivateImageRecord


def test_zero_error_certification_requires_2995_independent_trials() -> None:
    assert one_sided_error_upper_95(0, 2_994) > 0.001
    assert one_sided_error_upper_95(0, 2_995) <= 0.001
    assert error_gate(0, 2_994, 0.001)["passes"] is False
    assert error_gate(0, 2_995, 0.001)["passes"] is True


def test_99_percent_recall_requires_299_zero_failure_trials() -> None:
    assert one_sided_success_lower_95(298, 298) < 0.99
    assert one_sided_success_lower_95(299, 299) >= 0.99
    assert success_gate(298, 298, 0.99)["passes"] is False
    assert success_gate(299, 299, 0.99)["passes"] is True


def test_confidence_bounds_reject_invalid_counts_and_empty_evidence() -> None:
    with pytest.raises(ValueError):
        one_sided_error_upper_95(2, 1)
    with pytest.raises(ValueError):
        one_sided_success_lower_95(-1, 1)
    assert error_gate(0, 0, 0.001)["passes"] is False
    assert success_gate(0, 0, 0.99)["passes"] is False


def test_private_run_claim_is_atomic_and_cannot_be_reused(tmp_path) -> None:
    claim = tmp_path / "private-run-state.json"
    values = {
        "release_lock_sha256": "a" * 64,
        "preflight_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    claim_private_run(claim, **values)

    with pytest.raises(RuntimeError, match="already has"):
        claim_private_run(claim, **values)


def _item(ordinal: int, status: ItemStatus, class_id: str | None = None) -> ScanItem:
    return ScanItem(
        segmentation_id=f"segmentation_{ordinal:03d}",
        bbox=BoundingBox(x=ordinal, y=ordinal, width=10, height=10),
        status=status,
        reason_codes=([] if status is ItemStatus.APPROVED else ["CLASSIFIER_OUT_OF_CATALOG"]),
        prediction=(
            Prediction(class_id=class_id, class_name=class_id)
            if status is ItemStatus.APPROVED and class_id is not None
            else None
        ),
        top3=[],
        confidence=0.9,
    )


def test_private_aggregate_counts_unmatched_and_recapture_scene_approvals_as_wrong() -> None:
    eligible = PrivateImageRecord.model_validate(
        {
            "image_id": "eligible",
            "image_path": "eligible.jpg",
            "image_sha256": "a" * 64,
            "perceptual_hash": "0123456789abcdef",
            "store_id": "store",
            "capture_session_id": "session-a",
            "expected_image_status": "SEGMENTATION",
            "annotations": [
                {
                    "annotation_id": "valid",
                    "bbox_xywh": [0, 0, 10, 10],
                    "target_class_id": "bread_01",
                    "physical_object_id": "physical-a",
                    "catalog_membership": "in_catalog",
                    "expected_item_status": "APPROVED",
                },
                {
                    "annotation_id": "ood",
                    "bbox_xywh": [20, 0, 10, 10],
                    "target_class_id": "ood_01",
                    "physical_object_id": "physical-b",
                    "catalog_membership": "ood",
                    "expected_item_status": "SEGMENT_RECAPTURE",
                },
                {
                    "annotation_id": "invalid",
                    "bbox_xywh": [40, 0, 10, 10],
                    "target_class_id": "bread_02",
                    "physical_object_id": "physical-c",
                    "catalog_membership": "in_catalog",
                    "expected_item_status": "SEGMENT_RECAPTURE",
                },
            ],
        }
    )
    recapture = PrivateImageRecord.model_validate(
        {
            "image_id": "recapture",
            "image_path": "recapture.jpg",
            "image_sha256": "b" * 64,
            "perceptual_hash": "fedcba9876543210",
            "store_id": "store",
            "capture_session_id": "session-b",
            "expected_image_status": "IMAGE_RECAPTURE",
            "annotations": [],
        }
    )
    recapture_false_approval = _item(5, ItemStatus.APPROVED, "bread_01")
    outcomes = {
        "eligible": ImageOutcome(
            status=Status.SEGMENTATION,
            false_negative_count=0,
            false_positive_count=1,
            annotations={
                "valid": AnnotationOutcome(
                    item=_item(1, ItemStatus.APPROVED, "bread_01"),
                    forced_top3={"bread_01", "bread_02", "bread_03"},
                ),
                "ood": AnnotationOutcome(item=_item(2, ItemStatus.SEGMENT_RECAPTURE)),
                "invalid": AnnotationOutcome(item=_item(3, ItemStatus.SEGMENT_RECAPTURE)),
            },
            unmatched_items=[_item(4, ItemStatus.APPROVED, "bread_03")],
            latency_ms=10.0,
        ),
        "recapture": ImageOutcome(
            status=Status.SEGMENTATION,
            false_negative_count=0,
            false_positive_count=1,
            annotations={},
            unmatched_items=[recapture_false_approval],
            latency_ms=10.0,
        ),
    }

    counts, metrics = aggregate_metrics([eligible, recapture], outcomes)

    assert counts["approved_output_count"] == 3
    assert counts["wrong_approved_count"] == 2
    assert counts["unsafe_approved_on_recapture_gt_count"] == 1
    assert metrics["approved_output_misrecognition_rate"] == pytest.approx(2 / 3)
    assert metrics["approved_object_misrecognition_rate_judgeable_gt"] == pytest.approx(2 / 3)
    assert metrics["invalid_roi_correct_segment_recapture_recall"] == 1.0
