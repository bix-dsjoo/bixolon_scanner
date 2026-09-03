from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import SEMVER, SHA256, sha256_file
from .errors import PackageValidationError
from .model_package import (
    CountVerifierMetadata,
    DetectorMetadata,
    InputMetadata,
    ModelSource,
    QualityMetadata,
)
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
    fixed_batch_size: int | None = Field(default=None, ge=1)
    horizontal_flip_tta: bool = False
    rotation_180_tta: bool = False
    neighbor_mask: bool = True
    neighbor_distance_bias: float = Field(default=0.0, ge=-1.0)
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
    ridge_approval_thresholds: list[float | None] | None = None
    ridge_disagreement_minimum_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    ridge_disagreement_minimum_pair_probability: float | None = Field(default=None, ge=0.5, le=1.0)
    ridge_pair_temperature: float = Field(default=1.0, gt=0.0)
    ridge_top3_minimum_inverse_entropy: float | None = Field(default=None, le=0.0)
    ridge_require_retrieval_agreement: bool = False
    ridge_retrieval_minimum_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    detector_corroboration_minimum_score: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_corroboration_maximum_approval_score: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    detector_corroboration_low_similarity_minimum_score: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    detector_corroboration_low_similarity_maximum_retrieval: float | None = Field(
        default=None, ge=-1.0, le=1.0
    )
    detector_corroboration_low_similarity_minimum_approval_score: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
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
        if (self.detector_corroboration_minimum_score is None) != (
            self.detector_corroboration_maximum_approval_score is None
        ):
            raise ValueError(
                "detector corroboration score and approval thresholds must be configured together"
            )
        low_similarity_thresholds = (
            self.detector_corroboration_low_similarity_minimum_score,
            self.detector_corroboration_low_similarity_maximum_retrieval,
            self.detector_corroboration_low_similarity_minimum_approval_score,
        )
        if any(value is None for value in low_similarity_thresholds) and any(
            value is not None for value in low_similarity_thresholds
        ):
            raise ValueError(
                "low-similarity detector corroboration thresholds must be configured together"
            )
        if self.ood_maximum_similarity > self.approval_minimum_similarity:
            raise ValueError("OOD similarity threshold cannot exceed approval similarity threshold")
        if self.top3_minimum_similarity > self.approval_minimum_similarity:
            raise ValueError("Top-3 threshold cannot exceed approval similarity threshold")
        if self.ridge_approval_metric == "top2_pair_probability":
            if self.ridge_approval_minimum_pair_probability is None:
                raise ValueError("Top-2 pair approval requires a pair probability threshold")
            if self.ridge_approval_minimum_margin is not None:
                raise ValueError("Top-2 pair approval cannot also configure the legacy margin")
            if self.ridge_disagreement_minimum_margin is not None:
                raise ValueError("Top-2 pair approval cannot configure a disagreement margin")
            if (
                self.ridge_disagreement_minimum_pair_probability is not None
                and self.ridge_disagreement_minimum_pair_probability
                < self.ridge_approval_minimum_pair_probability
            ):
                raise ValueError(
                    "Ridge disagreement threshold cannot be lower than the base pair threshold"
                )
        else:
            if (
                self.ridge_approval_minimum_pair_probability is not None
                or self.ridge_disagreement_minimum_pair_probability is not None
            ):
                raise ValueError("Legacy Ridge margin approval cannot configure pair thresholds")
            if (
                self.ridge_disagreement_minimum_margin is not None
                and self.ridge_approval_minimum_margin is not None
                and self.ridge_disagreement_minimum_margin < self.ridge_approval_minimum_margin
            ):
                raise ValueError("Ridge disagreement margin cannot be lower than the base margin")
        if self.ridge_approval_thresholds is not None and any(
            threshold is not None and not 0.0 <= threshold <= 1.0
            for threshold in self.ridge_approval_thresholds
        ):
            raise ValueError("Ridge per-class approval thresholds must be in [0, 1]")
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
    low_agreement_count_maximum: int | None = Field(default=None, ge=0)
    low_agreement_aspect_ratio_minimum: float | None = Field(default=None, ge=1.0)

    @model_validator(mode="after")
    def validate_count_range(self) -> "DetectorAmbiguityPolicyMetadata":
        if self.dense_selected_count_minimum > self.dense_selected_count_maximum:
            raise ValueError("dense selected count minimum cannot exceed maximum")
        if (self.low_agreement_count_maximum is None) != (
            self.low_agreement_aspect_ratio_minimum is None
        ):
            raise ValueError("low-agreement ambiguity thresholds must be configured together")
        return self


class LargeProposalCorroborationMetadata(BaseModel):
    """Evidence required before a large detector proposal is treated as crowding."""

    model_config = ConfigDict(extra="forbid")

    query_containment_surplus_minimum: int = Field(ge=1)
    selected_center_minimum: int = Field(ge=2)
    selected_count_maximum: int = Field(ge=1)


class DetectorCrowdingPolicyMetadata(BaseModel):
    """Label-free detector policy for merged or missing object recapture."""

    model_config = ConfigDict(extra="forbid")

    minimum_image_aspect_ratio: float = Field(default=0.0, ge=0.0)
    candidate_score_threshold: float = Field(ge=0.0, le=1.0)
    large_proposal_score_threshold: float = Field(ge=0.0, le=1.0)
    large_proposal_minimum_area_ratio: float = Field(gt=0.0, le=1.0)
    large_proposal_corroboration: LargeProposalCorroborationMetadata | None = None
    proximity_maximum_normalized_center_distance: float = Field(gt=0.0)
    query_cluster_iou_threshold: float = Field(gt=0.0, le=1.0)
    query_duplicate_minimum_fraction: float = Field(ge=0.0, le=1.0)
    rotation_recovery_degrees: list[Literal[90, 180, 270]] = Field(default_factory=list)
    rotation_recovery_minimum_selected_count: int = Field(default=1, ge=1)
    rotation_recovery_maximum_selected_count: int | None = Field(default=None, ge=1)
    rotation_recovery_minimum_selected_area_fraction: float = Field(default=0.0, ge=0.0)
    rotation_recovery_minimum_normalized_center_distance: float = Field(default=0.0, ge=0.0)
    rotation_recovery_minimum_count_gain: int = Field(default=1, ge=1)
    rotation_recovery_agreement_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_score_thresholds(self) -> "DetectorCrowdingPolicyMetadata":
        if self.candidate_score_threshold > self.large_proposal_score_threshold:
            raise ValueError("crowding candidate threshold cannot exceed large-proposal threshold")
        if len(self.rotation_recovery_degrees) != len(set(self.rotation_recovery_degrees)):
            raise ValueError("crowding rotation recovery degrees must be unique")
        if (
            self.rotation_recovery_maximum_selected_count is not None
            and self.rotation_recovery_minimum_selected_count
            > self.rotation_recovery_maximum_selected_count
        ):
            raise ValueError("crowding rotation recovery selected-count range is inverted")
        return self


class ClassifierVerificationMetadata(BaseModel):
    """Product-independent selective agreement policy for ambiguous classifications."""

    model_config = ConfigDict(extra="forbid")

    ambiguity_maximum_approval_score: float = Field(ge=0.0, le=1.0)
    verify_all_approved_candidates: bool = False
    unknown_recapture_on_dual_verifier_rejection: bool = False
    unknown_recapture_on_any_verifier_rejection: bool = False
    rotation_degrees: Literal[180] = 180
    independent_embedder: EmbedderMetadata
    independent_metric_projection: MetricProjectionMetadata

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ClassifierVerificationMetadata":
        if (
            self.unknown_recapture_on_dual_verifier_rejection
            and self.unknown_recapture_on_any_verifier_rejection
        ):
            raise ValueError("classifier verifier dual and any rejection policies conflict")
        if (
            self.independent_metric_projection.input_dimension
            != self.independent_embedder.embedding_dimension
        ):
            raise ValueError("verification projection input must match its embedder output")
        return self


class ClassifierFallbackApprovalRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_detection_count: int = Field(default=1, ge=1)
    maximum_detection_count: int | None = Field(default=None, ge=1)
    maximum_approval_score: float = Field(ge=0.0, le=1.0)
    require_detector_disagreement: bool = True
    minimum_box_aspect_ratio: float | None = Field(default=None, ge=1.0)
    maximum_approval_score_decrease: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_detection_count_range(self) -> "ClassifierFallbackApprovalRule":
        if (
            self.maximum_detection_count is not None
            and self.maximum_detection_count < self.minimum_detection_count
        ):
            raise ValueError("classifier fallback detection count range is inverted")
        return self


class ClassifierResolutionFallbackMetadata(BaseModel):
    """Higher-resolution classifier used only for unsafe fast-path decisions."""

    model_config = ConfigDict(extra="forbid")

    embedder: EmbedderMetadata
    fallback_on_unknown: bool = True
    fallback_on_unsafe: bool = False
    selective_roi_only: bool = False
    fuse_unapproved_top3: bool = False
    minimum_fallback_approval_score: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_detector_support: int = Field(default=3, ge=1)
    approval_disagreement_rules: list[ClassifierFallbackApprovalRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trigger(self) -> "ClassifierResolutionFallbackMetadata":
        if (
            not self.fallback_on_unknown
            and not self.fallback_on_unsafe
            and not self.approval_disagreement_rules
        ):
            raise ValueError("classifier resolution fallback requires at least one trigger")
        return self


class DetectorPrimaryClassifierRoutingMetadata(BaseModel):
    """Allow calibrated detector classes to bypass the Catalog embedder."""

    model_config = ConfigDict(extra="forbid")

    direct_approval_class_indices: list[int] = Field(min_length=1)
    minimum_detector_score: float = Field(default=0.98, ge=0.0, le=1.0)
    require_unique_class_per_image: bool = True

    @field_validator("direct_approval_class_indices")
    @classmethod
    def validate_direct_approval_classes(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("detector-primary class indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("detector-primary class indices must be unique")
        return value


class RuntimePackageV2Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    worker_version: str
    # Legacy packages may contain this field. Version bundles no longer emit it.
    promotion_status: Literal["development", "independent_test_pending", "production"] | None = None
    dataset_version: str
    detector_policy_version: str
    detector_class_mode: Literal["class_agnostic", "class_aware"] = "class_aware"
    detector_class_count: int = Field(default=20, gt=0)
    detector: DetectorMetadata
    detector_refinement: DetectorRefinementMetadata | None = None
    detector_ambiguity: DetectorAmbiguityPolicyMetadata = Field(
        default_factory=DetectorAmbiguityPolicyMetadata
    )
    detector_crowding: DetectorCrowdingPolicyMetadata | None = None
    count_verifier: CountVerifierMetadata | None = None
    embedder: EmbedderMetadata
    metric_projection: MetricProjectionMetadata
    classifier_policy: CatalogDecisionPolicy
    classifier_verification: ClassifierVerificationMetadata | None = None
    classifier_resolution_fallback: ClassifierResolutionFallbackMetadata | None = None
    detector_primary_classifier_routing: DetectorPrimaryClassifierRoutingMetadata | None = None
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
    def validate_detector_class_contract(self) -> "RuntimePackageV2Metadata":
        routing = self.detector_primary_classifier_routing
        if self.detector_class_mode != "class_agnostic":
            if routing is not None and any(
                index >= self.detector_class_count
                for index in routing.direct_approval_class_indices
            ):
                raise ValueError("detector-primary class index exceeds detector class count")
            return self
        if self.detector_class_count != 1:
            raise ValueError("class-agnostic detector must expose one objectness class")
        if routing is not None:
            raise ValueError("class-agnostic detector cannot approve Catalog classes directly")
        if self.quality.detector_classifier_consensus is not None:
            raise ValueError("class-agnostic detector cannot use detector-class consensus")
        detector_corroboration = (
            self.classifier_policy.detector_corroboration_minimum_score,
            self.classifier_policy.detector_corroboration_maximum_approval_score,
            self.classifier_policy.detector_corroboration_low_similarity_minimum_score,
            self.classifier_policy.detector_corroboration_low_similarity_maximum_retrieval,
            self.classifier_policy.detector_corroboration_low_similarity_minimum_approval_score,
        )
        if any(value is not None for value in detector_corroboration):
            raise ValueError("class-agnostic detector cannot corroborate Catalog classes")
        fallback = self.classifier_resolution_fallback
        if fallback is not None and any(
            rule.require_detector_disagreement for rule in fallback.approval_disagreement_rules
        ):
            raise ValueError(
                "class-agnostic detector fallback rules cannot require class disagreement"
            )
        return self

    @model_validator(mode="after")
    def validate_detector_crowding_thresholds(self) -> "RuntimePackageV2Metadata":
        crowding = self.detector_crowding
        if crowding is not None and (
            self.detector.ensemble is not None or self.detector_refinement is not None
        ):
            raise ValueError("detector crowding currently requires a single-scale detector")
        if (
            crowding is not None
            and crowding.large_proposal_score_threshold > self.detector.score_threshold
        ):
            raise ValueError(
                "crowding large-proposal threshold cannot exceed detector score threshold"
            )
        return self

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
        verification = self.classifier_verification
        if verification is not None:
            if verification.independent_embedder.filename == self.embedder.filename:
                raise ValueError("verification embedder must use a distinct package filename")
            if verification.independent_embedder.version != self.classifier_policy.version:
                raise ValueError("verification embedder must use the product policy version")
        fallback = self.classifier_resolution_fallback
        if fallback is not None:
            fallback_embedder = fallback.embedder
            if fallback_embedder.filename == self.embedder.filename:
                raise ValueError(
                    "classifier fallback embedder must use a distinct package filename"
                )
            if fallback_embedder.version != self.embedder.version:
                raise ValueError("classifier fallback embedder version must match the primary")
            if fallback_embedder.embedder_id != self.embedder.embedder_id:
                raise ValueError("classifier fallback must use the primary embedder architecture")
            if fallback_embedder.embedding_dimension != self.embedder.embedding_dimension:
                raise ValueError("classifier fallback embedding dimension must match the primary")
            if (
                any(
                    fallback_size < primary_size
                    for fallback_size, primary_size in zip(
                        fallback_embedder.input_size,
                        self.embedder.input_size,
                        strict=True,
                    )
                )
                or fallback_embedder.input_size == self.embedder.input_size
            ):
                raise ValueError("classifier fallback input must be strictly higher resolution")
            comparable_fields = (
                "input_name",
                "output_name",
                "mean",
                "std",
                "crop_margin_ratio",
                "crop_mode",
                "l2_normalized",
                "resize_reducing_gap",
                "horizontal_flip_tta",
                "rotation_180_tta",
                "neighbor_mask",
                "neighbor_distance_bias",
                "neighbor_shared_scale",
            )
            if any(
                getattr(fallback_embedder, field) != getattr(self.embedder, field)
                for field in comparable_fields
            ):
                raise ValueError("classifier fallback preprocessing must match the primary")
        return self


@dataclass(frozen=True)
class RuntimePackageV2:
    root: Path
    metadata: RuntimePackageV2Metadata
    detector_path: Path
    count_verifier_path: Path | None
    embedder_path: Path
    classifier_fallback_embedder_path: Path | None
    metric_projection_path: Path | None
    verification_embedder_path: Path | None
    verification_metric_projection_path: Path | None


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
    if metadata.count_verifier is not None:
        required.add(metadata.count_verifier.filename)
    if metadata.metric_projection.filename is not None:
        required.add(metadata.metric_projection.filename)
    if metadata.classifier_verification is not None:
        required.add(metadata.classifier_verification.independent_embedder.filename)
        verification_projection = metadata.classifier_verification.independent_metric_projection
        if verification_projection.filename is not None:
            required.add(verification_projection.filename)
    if metadata.classifier_resolution_fallback is not None:
        required.add(metadata.classifier_resolution_fallback.embedder.filename)
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
    verification = metadata.classifier_verification
    verification_projection_path = (
        None
        if verification is None or verification.independent_metric_projection.filename is None
        else resolved_files[verification.independent_metric_projection.filename]
    )
    return RuntimePackageV2(
        root=package_root,
        metadata=metadata,
        detector_path=resolved_files[metadata.detector.filename],
        count_verifier_path=(
            None
            if metadata.count_verifier is None
            else resolved_files[metadata.count_verifier.filename]
        ),
        embedder_path=resolved_files[metadata.embedder.filename],
        classifier_fallback_embedder_path=(
            None
            if metadata.classifier_resolution_fallback is None
            else resolved_files[metadata.classifier_resolution_fallback.embedder.filename]
        ),
        metric_projection_path=projection_path,
        verification_embedder_path=(
            None
            if verification is None
            else resolved_files[verification.independent_embedder.filename]
        ),
        verification_metric_projection_path=verification_projection_path,
    )
