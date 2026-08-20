"""Stable API response contracts."""

from .api import (
    BoundingBox,
    Candidate,
    ItemStatus,
    ModelVersions,
    Prediction,
    ScanItem,
    ScanResponse,
    Status,
)
from .catalog import (
    CatalogActivation,
    CatalogLabel,
    CatalogMetadata,
    CatalogRestrictedPair,
    CatalogState,
    StoreCatalogPackage,
    load_store_catalog_package,
)
from .runtime_package_v2 import (
    CatalogDecisionPolicy,
    DetectorAmbiguityPolicyMetadata,
    DetectorRefinementMetadata,
    EmbedderMetadata,
    MetricProjectionMetadata,
    RuntimePackageV2,
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)

__all__ = [
    "BoundingBox",
    "Candidate",
    "ItemStatus",
    "ModelVersions",
    "Prediction",
    "ScanItem",
    "ScanResponse",
    "Status",
    "CatalogActivation",
    "CatalogLabel",
    "CatalogMetadata",
    "CatalogRestrictedPair",
    "CatalogState",
    "StoreCatalogPackage",
    "load_store_catalog_package",
    "CatalogDecisionPolicy",
    "DetectorAmbiguityPolicyMetadata",
    "DetectorRefinementMetadata",
    "EmbedderMetadata",
    "MetricProjectionMetadata",
    "RuntimePackageV2",
    "RuntimePackageV2Metadata",
    "load_runtime_package_v2",
]
