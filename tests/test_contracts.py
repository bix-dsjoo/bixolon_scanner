from __future__ import annotations

import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts import (
    BoundingBox,
    Candidate,
    ItemStatus,
    ScanItem,
    ScanResponse,
    Status,
)


def test_unknown_top3_must_be_descending():
    with pytest.raises(ValidationError):
        ScanItem(
            item_id="item_001",
            bbox=BoundingBox(x=1, y=2, width=3, height=4),
            status=ItemStatus.UNKNOWN,
            reason_codes=["BELOW_APPROVAL_THRESHOLD"],
            prediction=None,
            top3=[
                Candidate(class_id="a", class_name="A", confidence=0.2),
                Candidate(class_id="b", class_name="B", confidence=0.4),
            ],
            confidence=0.2,
        )


def test_image_recapture_requires_empty_segmentations_and_direct_versions():
    response = ScanResponse(
        request_id="12345678",
        status=Status.IMAGE_RECAPTURE,
        reason_codes=["DETECTOR_NO_OBJECT"],
        segmentations=[],
        processing_time_ms=1.0,
        worker_version="1.0.0",
        detector_version="1.0.0",
        classifier_version=None,
    )
    assert response.status is Status.IMAGE_RECAPTURE


def test_contained_duplicate_is_a_valid_unknown_reason():
    item = ScanItem(
        segmentation_id="segmentation_001",
        bbox=BoundingBox(x=1, y=2, width=30, height=40),
        status=ItemStatus.UNKNOWN,
        reason_codes=["DETECTOR_CONTAINED_DUPLICATE"],
        prediction=None,
        top3=[Candidate(class_id="a", class_name="A", confidence=0.9)],
        confidence=0.9,
    )
    response = ScanResponse(
        request_id="duplicate-review",
        status=Status.SEGMENTATION,
        reason_codes=["SEGMENT_DUPLICATE_REVIEW_REQUIRED"],
        segmentations=[item],
        processing_time_ms=1.0,
        worker_version="1.0.0",
        detector_version="1.0.0",
        classifier_version="1.0.0",
    )

    assert response.segmentations[0].reason_codes == ["DETECTOR_CONTAINED_DUPLICATE"]


def test_unknown_rejects_unsupported_reason():
    with pytest.raises(ValidationError):
        ScanItem(
            segmentation_id="segmentation_001",
            bbox=BoundingBox(x=1, y=2, width=30, height=40),
            status=ItemStatus.UNKNOWN,
            reason_codes=["UNSUPPORTED_UNKNOWN_REASON"],
            prediction=None,
            top3=[Candidate(class_id="a", class_name="A", confidence=0.9)],
            confidence=0.9,
        )


def test_2_0_unknown_separates_approval_and_ranking_confidence():
    item = ScanItem(
        segmentation_id="segmentation_001",
        bbox=BoundingBox(x=1, y=2, width=30, height=40),
        status=ItemStatus.UNKNOWN,
        reason_codes=["CLASSIFIER_AMBIGUOUS_TOP2"],
        prediction=None,
        top3=[
            Candidate(class_id="a", class_name="A", confidence=0.91),
            Candidate(class_id="b", class_name="B", confidence=0.88),
        ],
        confidence=0.42,
    )

    assert item.confidence == 0.42
    assert item.top3[0].confidence == 0.91
