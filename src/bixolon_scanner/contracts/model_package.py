from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import PackageValidationError

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")


class DetectorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    version: str
    input_name: str = "pixel_values"
    logits_output: str = "logits"
    boxes_output: str = "pred_boxes"
    input_size: tuple[int, int] = (640, 640)
    color_order: Literal["RGB"] = "RGB"
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    score_threshold: float = Field(ge=0.0, le=1.0)
    uncertainty_score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty_min_area_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_match_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    max_queries: int = Field(gt=0)
    box_format: Literal["normalized_cxcywh"] = "normalized_cxcywh"
    resize_reducing_gap: float | None = Field(default=None, ge=1.0)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_uncertainty_threshold(self) -> "DetectorMetadata":
        if (
            self.uncertainty_score_threshold is not None
            and self.uncertainty_score_threshold >= self.score_threshold
        ):
            raise ValueError("uncertainty threshold must be below the detection threshold")
        return self


class ClassLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str
    class_name: str
    recapture: bool = False


class ClassifierMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    version: str
    input_name: str = "pixel_values"
    logits_output: str = "logits"
    input_size: tuple[int, int] = (224, 224)
    color_order: Literal["RGB"] = "RGB"
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_margin_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    approval_threshold: float = Field(ge=0.0, le=1.0)
    temperature: float = Field(gt=0.0)
    labels: list[ClassLabel] = Field(min_length=1)
    resize_reducing_gap: float | None = Field(default=None, ge=1.0)
    warmup_batch_sizes: list[int] = Field(default_factory=lambda: [1], min_length=1)

    @field_validator("warmup_batch_sizes")
    @classmethod
    def validate_warmup_batch_sizes(cls, value: list[int]) -> list[int]:
        if any(size < 1 for size in value) or len(value) != len(set(value)):
            raise ValueError("warmup_batch_sizes must contain unique positive values")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_unique_labels(self) -> "ClassifierMetadata":
        ids = [label.class_id for label in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError("classifier label IDs must be unique")
        return self


class CountVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    version: str
    input_name: str = "pixel_values"
    logits_output: str = "logits"
    input_size: tuple[int, int] = (320, 320)
    color_order: Literal["RGB"] = "RGB"
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    count_labels: list[int] = Field(min_length=2)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    resize_reducing_gap: float | None = Field(default=None, ge=1.0)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("version must use semantic versioning")
        return value

    @field_validator("count_labels")
    @classmethod
    def validate_count_labels(cls, value: list[int]) -> list[int]:
        if any(count < 0 for count in value) or value != sorted(set(value)):
            raise ValueError("count_labels must contain sorted unique non-negative counts")
        return value


class QualityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_object_area_ratio: float = Field(default=0.005, ge=0.0, le=1.0)
    border_margin_ratio: float = Field(default=0.002, ge=0.0, le=0.25)
    border_policy: Literal["always_recapture", "classifier_confidence"] = "always_recapture"
    min_sharpness: float | None = Field(default=None, ge=0.0)
    min_mean_luminance: float | None = Field(default=None, ge=0.0, le=255.0)
    max_mean_luminance: float | None = Field(default=None, ge=0.0, le=255.0)

    @model_validator(mode="after")
    def validate_luminance(self) -> "QualityMetadata":
        if (
            self.min_mean_luminance is not None
            and self.max_mean_luminance is not None
            and self.min_mean_luminance >= self.max_mean_luminance
        ):
            raise ValueError("minimum luminance must be below maximum luminance")
        return self


class InputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jpeg_draft_size: int | None = Field(default=1500, gt=0)


class ModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str | None = None
    revision: str | None = None
    weight_filename: str | None = None
    weight_sha256: str | None = None

    @field_validator("weight_sha256")
    @classmethod
    def validate_weight_checksum(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("weight_sha256 must be a lowercase SHA-256 digest")
        return value


class CalibrationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(ge=1)
    approved_precision: float = Field(ge=0.0, le=1.0)
    approval_coverage: float = Field(ge=0.0, le=1.0)
    false_approval_rate_upper_95: float = Field(ge=0.0, le=1.0)
    risk_control_satisfied: bool


class DetectorEvaluationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    count_accuracy: float = Field(ge=0.0, le=1.0)
    target_recall_satisfied: bool


class BundleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_mode: Literal["detector_safety_first_0.2.5"]
    model_version: str
    classifier_source_version: str
    classifier_source_sha256: str
    detector_selection_sha256: str
    evaluation_dataset_versions: dict[str, str]

    @field_validator("model_version", "classifier_source_version")
    @classmethod
    def validate_versions(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("bundle provenance versions must use semantic versioning")
        return value

    @field_validator("classifier_source_sha256", "detector_selection_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("bundle provenance requires lowercase SHA-256 digests")
        return value

    @field_validator("evaluation_dataset_versions")
    @classmethod
    def validate_evaluation_dataset_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != {"natural", "hard", "shift"}:
            raise ValueError("bundle provenance requires natural/hard/shift dataset versions")
        if any(not version.strip() for version in value.values()):
            raise ValueError("evaluation dataset versions cannot be empty")
        return value


class PromotionWaiver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: Literal["unknown_top3_accuracy", "evaluation_set_independence"]
    observed: float = Field(ge=0.0, le=1.0)
    target: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=1)
    correct_count: int = Field(ge=0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "PromotionWaiver":
        if self.correct_count > self.sample_count:
            raise ValueError("correct_count cannot exceed sample_count")
        return self


class PromotionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved"]
    method: Literal["all_gates", "manual_waiver"]
    decided_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    waivers: list[PromotionWaiver] = Field(default_factory=list)
    remaining_limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_method(self) -> "PromotionMetadata":
        if self.method == "manual_waiver" and not self.waivers:
            raise ValueError("manual_waiver requires at least one waiver")
        if self.method == "all_gates" and self.waivers:
            raise ValueError("all_gates cannot include waivers")
        return self


class ModelPackageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"]
    package_version: str
    promotion_status: Literal["development", "production"]
    dataset_version: str
    detector: DetectorMetadata
    classifier: ClassifierMetadata
    count_verifier: CountVerifierMetadata | None = None
    input: InputMetadata = Field(default_factory=InputMetadata)
    quality: QualityMetadata
    checksums: dict[str, str]
    licenses: dict[str, str]
    sources: dict[str, ModelSource] = Field(default_factory=dict)
    calibration: CalibrationMetadata | None = None
    detector_evaluation: DetectorEvaluationMetadata | None = None
    bundle_provenance: BundleProvenance | None = None
    promotion: PromotionMetadata | None = None

    @field_validator("package_version")
    @classmethod
    def validate_package_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("package_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_production_decision(self) -> "ModelPackageMetadata":
        if self.promotion_status == "production" and self.promotion is None:
            raise ValueError("production packages require a promotion decision record")
        if self.schema_version == "1.0" and (
            self.count_verifier is not None
            or self.detector.uncertainty_score_threshold is not None
            or self.detector.uncertainty_min_area_ratio != 0.0
            or self.quality.border_policy != "always_recapture"
        ):
            raise ValueError("schema 1.1 is required for confidence quality policies")
        if self.bundle_provenance is not None:
            if self.schema_version != "1.1":
                raise ValueError("bundle provenance requires schema 1.1")
            if not (
                self.package_version
                == self.detector.version
                == self.classifier.version
                == self.bundle_provenance.model_version
            ):
                raise ValueError(
                    "detector target packages require one shared inference model version"
                )
        return self


class ModelPackage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    metadata: ModelPackageMetadata

    @property
    def detector_path(self) -> Path:
        return self.root / self.metadata.detector.filename

    @property
    def classifier_path(self) -> Path:
        return self.root / self.metadata.classifier.filename

    @property
    def count_verifier_path(self) -> Path | None:
        if self.metadata.count_verifier is None:
            return None
        return self.root / self.metadata.count_verifier.filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_package_file(root: Path, filename: str) -> Path:
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PackageValidationError from exc
    if not candidate.is_file():
        raise PackageValidationError
    return candidate


def load_model_package(package_dir: Path) -> ModelPackage:
    root = package_dir.resolve()
    metadata_path = root / "metadata.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = ModelPackageMetadata.model_validate(raw)
        required = [metadata.detector.filename, metadata.classifier.filename]
        if metadata.count_verifier is not None:
            required.append(metadata.count_verifier.filename)
        for filename in required:
            path = _resolve_package_file(root, filename)
            expected = metadata.checksums.get(filename)
            if expected is None or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise PackageValidationError
            if sha256_file(path) != expected:
                raise PackageValidationError
        if not metadata.licenses:
            raise PackageValidationError
    except PackageValidationError:
        raise
    except Exception as exc:
        raise PackageValidationError from exc
    return ModelPackage(root=root, metadata=metadata)
