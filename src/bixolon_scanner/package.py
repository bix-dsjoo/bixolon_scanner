"""Compatibility imports for model package contracts and loading."""

from .contracts.model_package import (
    BundleProvenance,
    CalibrationMetadata,
    ClassifierMetadata,
    ClassLabel,
    CountVerifierMetadata,
    DetectorEvaluationMetadata,
    DetectorMetadata,
    InputMetadata,
    ModelPackage,
    ModelPackageMetadata,
    ModelSource,
    PromotionMetadata,
    PromotionWaiver,
    QualityMetadata,
    load_model_package,
    sha256_file,
)

__all__ = [
    "BundleProvenance",
    "CalibrationMetadata",
    "ClassifierMetadata",
    "ClassLabel",
    "CountVerifierMetadata",
    "DetectorEvaluationMetadata",
    "DetectorMetadata",
    "InputMetadata",
    "ModelPackage",
    "ModelPackageMetadata",
    "ModelSource",
    "PromotionMetadata",
    "PromotionWaiver",
    "QualityMetadata",
    "load_model_package",
    "sha256_file",
]
