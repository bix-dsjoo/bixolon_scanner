from __future__ import annotations

import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts import (
    BoundingBox,
    Candidate,
    ItemStatus,
    ModelVersions,
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


def test_recapture_requires_empty_items():
    response = ScanResponse(
        request_id="12345678",
        status=Status.RECAPTURE,
        reason_codes=["DETECTOR_NO_OBJECT"],
        items=[],
        processing_time_ms=1.0,
        model_versions=ModelVersions(detector="1.0.0", classifier=None),
    )
    assert response.status is Status.RECAPTURE
