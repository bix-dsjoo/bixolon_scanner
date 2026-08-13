from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(StrEnum):
    APPROVED = "APPROVED"
    UNKNOWN = "UNKNOWN"
    RECAPTURE = "RECAPTURE"
    ERROR = "ERROR"


class ItemStatus(StrEnum):
    APPROVED = "APPROVED"
    UNKNOWN = "UNKNOWN"


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

    item_id: str = Field(pattern=r"^item_\d{3,}$")
    bbox: BoundingBox
    status: ItemStatus
    reason_codes: list[str] = Field(default_factory=list)
    prediction: Prediction | None = None
    top3: list[Candidate] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ScanItem":
        if self.status is ItemStatus.APPROVED:
            if self.prediction is None or self.top3 or self.reason_codes:
                raise ValueError("APPROVED item requires prediction and empty top3/reason_codes")
        else:
            if self.prediction is not None or not self.top3:
                raise ValueError("UNKNOWN item requires null prediction and non-empty top3")
            if "BELOW_APPROVAL_THRESHOLD" not in self.reason_codes:
                raise ValueError("UNKNOWN item requires BELOW_APPROVAL_THRESHOLD")
            scores = [candidate.confidence for candidate in self.top3]
            if scores != sorted(scores, reverse=True):
                raise ValueError("top3 must be sorted by descending confidence")
            if abs(self.confidence - scores[0]) > 1e-6:
                raise ValueError("UNKNOWN confidence must equal Top-1 confidence")
        return self


class ModelVersions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: str | None
    classifier: str | None


class ScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=8)
    status: Status
    reason_codes: list[str] = Field(default_factory=list)
    items: list[ScanItem] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)
    model_versions: ModelVersions

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ScanResponse":
        if self.status in (Status.RECAPTURE, Status.ERROR):
            if self.items:
                raise ValueError(f"{self.status} response must have empty items")
            if not self.reason_codes:
                raise ValueError(f"{self.status} response requires reason_codes")
            return self
        if not self.items:
            raise ValueError(f"{self.status} response requires items")
        has_unknown = any(item.status is ItemStatus.UNKNOWN for item in self.items)
        if self.status is Status.APPROVED and (has_unknown or self.reason_codes):
            raise ValueError("APPROVED response requires all items approved and no reasons")
        if self.status is Status.UNKNOWN:
            if not has_unknown or "ITEM_BELOW_APPROVAL_THRESHOLD" not in self.reason_codes:
                raise ValueError("UNKNOWN response requires an UNKNOWN item and aggregate reason")
        return self
