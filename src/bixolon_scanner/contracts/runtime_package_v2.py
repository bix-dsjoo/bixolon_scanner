from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import SEMVER, SHA256, sha256_file
from .errors import PackageValidationError
from .model_package import DetectorMetadata, InputMetadata, ModelSource, QualityMetadata
from .package_files import resolve_package_file, validate_package_filename


class EmbedderMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "embedder.onnx"
    embedder_id: str = Field(min_length=1)
    version: str
    input_name: str = "pixel_values"
    output_name: str = "embeddings"
    input_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_margin_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    crop_mode: Literal["box_resize", "square_context"] = "square_context"
    embedding_dimension: int = Field(gt=0)
    l2_normalized: bool = False
    resize_reducing_gap: float | None = Field(default=3.0, ge=1.0)
    warmup_batch_sizes: list[int] = Field(default_factory=lambda: [1, 3, 5, 8])
    neighbor_mask: bool = True
    neighbor_distance_bias: float = Field(default=0.0, ge=0.0)
    neighbor_shared_scale: bool = False

    _validate_filename = field_validator("filename")(validate_package_filename)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("embedder version must use semantic versioning")
        return value


class MetricProjectionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    input_dimension: int = Field(gt=0)
    output_dimension: int = Field(gt=0)
    residual_weight: float = Field(default=1.0, ge=0.0)
    projection_weight: float = Field(default=0.0, ge=0.0)

    _validate_filename = field_validator("filename")(validate_package_filename)

    @model_validator(mode="after")
    def validate_projection(self) -> "MetricProjectionMetadata":
        if (self.filename is None) != (self.projection_weight == 0.0):
            raise ValueError(
                "metric projection file and non-zero weight must be configured together"
            )
        if self.residual_weight == 0.0 and self.projection_weight == 0.0:
            raise ValueError("metric projection cannot disable every feature branch")
        return self


class CatalogSupportAugmentationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    views_per_source: int = Field(default=0, ge=0, le=16)
    seed: int = Field(default=20260819, ge=0)
    output_size: int = Field(default=224, ge=64)
    canvas_scale_min: float = Field(default=0.72, gt=0.0, le=1.0)
    canvas_scale_max: float = Field(default=0.98, gt=0.0, le=1.0)
    rotation_degrees: float = Field(default=180.0, ge=0.0, le=180.0)
    perspective_fraction: float = Field(default=0.04, ge=0.0, le=0.25)
    brightness_min: float = Field(default=0.8, gt=0.0)
    brightness_max: float = Field(default=1.2, gt=0.0)
    contrast_min: float = Field(default=0.85, gt=0.0)
    contrast_max: float = Field(default=1.15, gt=0.0)
    saturation_min: float = Field(default=0.85, gt=0.0)
    saturation_max: float = Field(default=1.15, gt=0.0)
    blur_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    blur_radius_max: float = Field(default=0.7, ge=0.0)
    jpeg_quality_min: int = Field(default=82, ge=1, le=100)
    jpeg_quality_max: int = Field(default=96, ge=1, le=100)
    crop_mode: Literal[
        "white_alpha_composite", "padded_letterbox", "border_connected_composite"
    ] = "border_connected_composite"
    procedural_gradient: bool = True
    procedural_shadow: bool = True
    compiler_batch_size: int = Field(default=64, ge=1, le=256)

    @model_validator(mode="after")
    def validate_ranges(self) -> "CatalogSupportAugmentationMetadata":
        for lower_name, upper_name in (
            ("canvas_scale_min", "canvas_scale_max"),
            ("brightness_min", "brightness_max"),
            ("contrast_min", "contrast_max"),
            ("saturation_min", "saturation_max"),
            ("jpeg_quality_min", "jpeg_quality_max"),
        ):
            if getattr(self, lower_name) > getattr(self, upper_name):
                raise ValueError(f"{lower_name} cannot exceed {upper_name}")
        return self


class CatalogDecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    prototype_weight: float = Field(ge=0.0, le=1.0)
    support_top_k: int = Field(ge=1, le=10)
    approval_minimum_similarity: float = Field(ge=-1.0, le=1.0)
    approval_minimum_margin: float = Field(ge=0.0, le=2.0)
    ood_maximum_similarity: float = Field(ge=-1.0, le=1.0)
    top3_minimum_similarity: float = Field(ge=-1.0, le=1.0)
    catalog_conflict_similarity: float = Field(ge=-1.0, le=1.0)
    ridge_approval_metric: Literal["l2_normalized_logit_margin", "top2_pair_probability"] = (
        "l2_normalized_logit_margin"
    )
    ridge_approval_minimum_margin: float | None = Field(default=None, ge=0.0)
    ridge_approval_minimum_pair_probability: float | None = Field(default=None, ge=0.5, le=1.0)
    ridge_disagreement_minimum_pair_probability: float | None = Field(default=None, ge=0.5, le=1.0)
    ridge_pair_temperature: float = Field(default=1.0, gt=0.0)
    ridge_top3_minimum_inverse_entropy: float | None = Field(default=None, le=0.0)
    ridge_alpha: float = Field(default=0.01, gt=0.0)
    support_augmentation: CatalogSupportAugmentationMetadata = Field(
        default_factory=CatalogSupportAugmentationMetadata
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("classifier policy version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "CatalogDecisionPolicy":
        if self.ood_maximum_similarity > self.approval_minimum_similarity:
            raise ValueError("OOD similarity threshold cannot exceed approval similarity threshold")
        if self.top3_minimum_similarity > self.approval_minimum_similarity:
            raise ValueError("Top-3 threshold cannot exceed approval similarity threshold")
        if self.ridge_approval_metric == "top2_pair_probability":
            if self.ridge_approval_minimum_pair_probability is None:
                raise ValueError("Top-2 pair approval requires a pair probability threshold")
            if self.ridge_approval_minimum_margin is not None:
                raise ValueError("Top-2 pair approval cannot also configure the legacy margin")
            if (
                self.ridge_disagreement_minimum_pair_probability is not None
                and self.ridge_disagreement_minimum_pair_probability
                < self.ridge_approval_minimum_pair_probability
            ):
                raise ValueError(
                    "Ridge disagreement threshold cannot be lower than the base pair threshold"
                )
        elif (
            self.ridge_approval_minimum_pair_probability is not None
            or self.ridge_disagreement_minimum_pair_probability is not None
        ):
            raise ValueError("Legacy Ridge margin approval cannot configure pair thresholds")
        return self


class DetectorRefinementMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    input_size: tuple[int, int] = (768, 768)
    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    containment_threshold: float = Field(gt=0.0, le=1.0)
    group_minimum: int = Field(ge=2)
    agreement_iou_threshold: float = Field(ge=0.0, le=1.0)

    _validate_filename = field_validator("filename")(validate_package_filename)


class DetectorAmbiguityPolicyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["all", "selective"] = "all"
    high_aspect_ratio_minimum: float = Field(default=2.0, ge=1.0)
    dense_selected_count_minimum: int = Field(default=6, ge=1)
    dense_selected_count_maximum: int = Field(default=6, ge=1)
    dense_agreement_count_minimum: int = Field(default=4, ge=1)
    dense_aspect_ratio_minimum: float = Field(default=1.5, ge=1.0)

    @model_validator(mode="after")
    def validate_count_range(self) -> "DetectorAmbiguityPolicyMetadata":
        if self.dense_selected_count_minimum > self.dense_selected_count_maximum:
            raise ValueError("dense selected count minimum cannot exceed maximum")
        return self


class RuntimePackageV2Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    worker_version: str
    # Legacy packages may contain this field. Version bundles no longer emit it.
    promotion_status: Literal["development", "independent_test_pending", "production"] | None = None
    dataset_version: str
    detector_policy_version: str
    detector_class_count: int = Field(default=20, gt=0)
    detector: DetectorMetadata
    detector_refinement: DetectorRefinementMetadata | None = None
    detector_ambiguity: DetectorAmbiguityPolicyMetadata = Field(
        default_factory=DetectorAmbiguityPolicyMetadata
    )
    embedder: EmbedderMetadata
    metric_projection: MetricProjectionMetadata
    classifier_policy: CatalogDecisionPolicy
    input: InputMetadata = Field(default_factory=InputMetadata)
    quality: QualityMetadata
    checksums: dict[str, str]
    licenses: dict[str, str]
    license_files: list[str] = Field(default_factory=list)
    sources: dict[str, ModelSource] = Field(default_factory=dict)

    @field_validator("worker_version", "detector_policy_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("runtime versions must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> "RuntimePackageV2Metadata":
        if self.metric_projection.input_dimension != self.embedder.embedding_dimension:
            raise ValueError("metric projection input must match embedder output")
        augmentation = self.classifier_policy.support_augmentation
        if (
            augmentation.views_per_source
            and augmentation.output_size != self.embedder.input_size[0]
        ):
            raise ValueError("support augmentation size must match the embedder input")
        if len(self.license_files) != len(set(self.license_files)):
            raise ValueError("runtime license files must be unique")
        if any(
            Path(filename).is_absolute() or ".." in Path(filename).parts
            for filename in self.license_files
        ):
            raise ValueError("runtime license files must be confined relative paths")
        return self


@dataclass(frozen=True)
class RuntimePackageV2:
    root: Path
    metadata: RuntimePackageV2Metadata
    detector_path: Path
    embedder_path: Path
    metric_projection_path: Path | None


def load_runtime_package_v2(root: Path) -> RuntimePackageV2:
    package_root = root.resolve()
    try:
        payload = json.loads((package_root / "metadata.json").read_text(encoding="utf-8"))
        metadata = RuntimePackageV2Metadata.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PackageValidationError from exc
    required = {metadata.detector.filename, metadata.embedder.filename}
    if metadata.detector.ensemble is not None:
        required.update(member.filename for member in metadata.detector.ensemble.members)
    if metadata.detector_refinement is not None:
        required.add(metadata.detector_refinement.filename)
    if metadata.metric_projection.filename is not None:
        required.add(metadata.metric_projection.filename)
    required.update(metadata.license_files)
    if set(metadata.checksums) != required:
        raise PackageValidationError
    resolved_files: dict[str, Path] = {}
    for filename, expected in metadata.checksums.items():
        if not SHA256.fullmatch(expected):
            raise PackageValidationError
        path = resolve_package_file(package_root, filename)
        if sha256_file(path) != expected:
            raise PackageValidationError
        resolved_files[filename] = path
    projection_path = (
        None
        if metadata.metric_projection.filename is None
        else resolved_files[metadata.metric_projection.filename]
    )
    return RuntimePackageV2(
        root=package_root,
        metadata=metadata,
        detector_path=resolved_files[metadata.detector.filename],
        embedder_path=resolved_files[metadata.embedder.filename],
        metric_projection_path=projection_path,
    )
