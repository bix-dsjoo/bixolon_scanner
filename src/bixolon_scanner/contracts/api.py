from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(StrEnum):
    """Image-level Worker outcome.

    APPROVED/UNKNOWN/RECAPTURE remain aliases for Python 0.x consumers.  The
    serialized 1.0 contract only emits SEGMENTATION, IMAGE_RECAPTURE, or ERROR.
    """

    SEGMENTATION = "SEGMENTATION"
    IMAGE_RECAPTURE = "IMAGE_RECAPTURE"
    ERROR = "ERROR"

    APPROVED = "SEGMENTATION"
    UNKNOWN = "SEGMENTATION"
    RECAPTURE = "IMAGE_RECAPTURE"


class ItemStatus(StrEnum):
    APPROVED = "APPROVED"
    UNKNOWN = "UNKNOWN"
    SEGMENT_RECAPTURE = "SEGMENT_RECAPTURE"


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)


class Candidate(Prediction):
    confidence: float = Field(ge=0.0, le=1.0)


class ScanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmentation_id: str = Field(pattern=r"^segmentation_\d{3,}$")
    bbox: BoundingBox
    status: ItemStatus
    reason_codes: list[str] = Field(default_factory=list)
    prediction: Prediction | None = None
    top3: list[Candidate] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def item_id(self) -> str:
        """Compatibility accessor for Python 0.x evaluation code."""
        return self.segmentation_id.replace("segmentation_", "item_", 1)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ScanItem":
        if self.status is ItemStatus.APPROVED:
            if self.prediction is None or self.top3 or self.reason_codes:
                raise ValueError("APPROVED segmentation requires prediction and empty reasons")
        elif self.status is ItemStatus.UNKNOWN:
            if self.prediction is not None or not self.top3:
                raise ValueError("UNKNOWN segmentation requires null prediction and non-empty top3")
            allowed_reasons = {
                "BELOW_APPROVAL_THRESHOLD",
                "DETECTOR_CONTAINED_DUPLICATE",
            }
            if len(self.reason_codes) != 1 or self.reason_codes[0] not in allowed_reasons:
                raise ValueError("UNKNOWN segmentation requires exactly one supported reason")
            scores = [candidate.confidence for candidate in self.top3]
            if scores != sorted(scores, reverse=True):
                raise ValueError("top3 must be sorted by descending confidence")
            if abs(self.confidence - scores[0]) > 1e-6:
                raise ValueError("UNKNOWN confidence must equal Top-1 confidence")
        elif self.prediction is not None or self.top3 or not self.reason_codes:
            raise ValueError("SEGMENT_RECAPTURE requires reasons and no prediction or candidates")
        return self


class ModelVersions(BaseModel):
    """Compatibility value object; 1.0 serializes versions as top-level fields."""

    model_config = ConfigDict(extra="forbid")

    detector: str | None
    classifier: str | None


class ScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=8)
    status: Status
    reason_codes: list[str] = Field(default_factory=list)
    segmentations: list[ScanItem] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)
    worker_version: str
    detector_version: str | None
    classifier_version: str | None

    @property
    def items(self) -> list[ScanItem]:
        """Compatibility accessor for Python 0.x internal consumers."""
        return self.segmentations

    @property
    def model_versions(self) -> ModelVersions:
        """Compatibility accessor for Python 0.x internal consumers."""
        return ModelVersions(detector=self.detector_version, classifier=self.classifier_version)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ScanResponse":
        if self.status in (Status.IMAGE_RECAPTURE, Status.ERROR):
            if self.segmentations:
                raise ValueError(f"{self.status} response must have empty segmentations")
            if not self.reason_codes:
                raise ValueError(f"{self.status} response requires reason_codes")
            if self.status is Status.IMAGE_RECAPTURE and self.detector_version is None:
                raise ValueError("IMAGE_RECAPTURE requires detector_version")
            return self
        if not self.segmentations:
            raise ValueError("SEGMENTATION response requires segmentations")
        if self.detector_version is None or self.classifier_version is None:
            raise ValueError("SEGMENTATION requires executed detector and classifier versions")
        has_below_threshold = any(
            "BELOW_APPROVAL_THRESHOLD" in item.reason_codes for item in self.segmentations
        )
        has_duplicate_review = any(
            "DETECTOR_CONTAINED_DUPLICATE" in item.reason_codes for item in self.segmentations
        )
        has_segment_recapture = any(
            item.status is ItemStatus.SEGMENT_RECAPTURE for item in self.segmentations
        )
        expected_reasons = []
        if has_below_threshold:
            expected_reasons.append("SEGMENT_BELOW_APPROVAL_THRESHOLD")
        if has_duplicate_review:
            expected_reasons.append("SEGMENT_DUPLICATE_REVIEW_REQUIRED")
        if has_segment_recapture:
            expected_reasons.append("SEGMENT_RECAPTURE_REQUIRED")
        if self.reason_codes != expected_reasons:
            raise ValueError("SEGMENTATION aggregate reasons do not match segment outcomes")
        return self
