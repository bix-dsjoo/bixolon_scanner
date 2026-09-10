from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.errors import PackageValidationError
from bixolon_scanner.contracts.runtime_package_v2 import (
    CatalogDecisionPolicy,
    ClassifierVerificationMetadata,
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)


def test_classifier_verification_unknown_recapture_defaults_off_and_can_be_enabled() -> None:
    payload = {
        "ambiguity_maximum_approval_score": 0.5,
        "independent_embedder": {
            "filename": "verifier.onnx",
            "embedder_id": "test-verifier",
            "version": "0.1.7",
            "embedding_dimension": 8,
            "fixed_batch_size": 1,
        },
        "independent_metric_projection": {
            "input_dimension": 8,
            "output_dimension": 8,
        },
    }

    disabled = ClassifierVerificationMetadata.model_validate(payload)
    enabled = ClassifierVerificationMetadata.model_validate(
        {**payload, "unknown_recapture_on_dual_verifier_rejection": True}
    )
    any_enabled = ClassifierVerificationMetadata.model_validate(
        {**payload, "unknown_recapture_on_any_verifier_rejection": True}
    )

    assert disabled.unknown_recapture_on_dual_verifier_rejection is False
    assert disabled.unknown_recapture_on_any_verifier_rejection is False
    assert disabled.verify_all_approved_candidates is False
    assert enabled.unknown_recapture_on_dual_verifier_rejection is True
    assert any_enabled.unknown_recapture_on_any_verifier_rejection is True

    all_approved = ClassifierVerificationMetadata.model_validate(
        {**payload, "verify_all_approved_candidates": True}
    )
    assert all_approved.verify_all_approved_candidates is True

    with pytest.raises(ValidationError, match="dual and any rejection policies conflict"):
        ClassifierVerificationMetadata.model_validate(
            {
                **payload,
                "unknown_recapture_on_dual_verifier_rejection": True,
                "unknown_recapture_on_any_verifier_rejection": True,
            }
        )


def test_batch_variants_require_checksum_and_reject_corruption(tmp_path):
    payload = _metadata(tmp_path)
    payload["embedder"].update(
        fixed_batch_size=2,
        batch_variants=[{"filename": "batch1.onnx", "batch_size": 1}],
    )
    variant = tmp_path / "batch1.onnx"
    variant.write_bytes(b"variant")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(tmp_path)
    payload["checksums"]["batch1.onnx"] = sha256_file(variant)
    metadata_path.write_text(json.dumps(payload))
    assert load_runtime_package_v2(tmp_path).metadata.embedder.batch_variants[0].batch_size == 1
    variant.write_bytes(b"corrupt")
    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(tmp_path)


def _metadata(root: Path) -> dict:
    (root / "detector.onnx").write_bytes(b"detector")
    (root / "embedder.onnx").write_bytes(b"embedder")
    license_path = root / "licenses" / "APACHE-2.0.txt"
    license_path.parent.mkdir()
    license_path.write_text("Apache License 2.0\n", encoding="utf-8")
    files = ["detector.onnx", "embedder.onnx", "licenses/APACHE-2.0.txt"]
    return {
        "schema_version": "2.0",
        "worker_version": "2.0.0-rc.7",
        "promotion_status": "independent_test_pending",
        "dataset_version": "test-dataset",
        "detector_policy_version": "2.0.0-rc.7",
        "detector_class_count": 2,
        "detector": {
            "filename": "detector.onnx",
            "version": "2.0.0-rc.7",
            "score_threshold": 0.5,
            "nms_iou_threshold": 0.5,
            "max_queries": 100,
        },
        "embedder": {
            "filename": "embedder.onnx",
            "embedder_id": "test-embedder",
            "version": "2.0.0-rc.7",
            "embedding_dimension": 8,
        },
        "metric_projection": {
            "input_dimension": 8,
            "output_dimension": 8,
        },
        "classifier_policy": {
            "version": "2.0.0-rc.7",
            "prototype_weight": 0.5,
            "support_top_k": 3,
            "approval_minimum_similarity": 0.5,
            "approval_minimum_margin": 0.1,
            "ood_maximum_similarity": 0.1,
            "top3_minimum_similarity": 0.2,
            "catalog_conflict_similarity": 0.9,
        },
        "quality": {},
        "checksums": {filename: sha256_file(root / filename) for filename in files},
        "licenses": {"detector": "Apache-2.0", "classifier": "Apache-2.0"},
        "license_files": ["licenses/APACHE-2.0.txt"],
    }


def _write_package(root: Path, payload: dict) -> None:
    (root / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_runtime_package_v2_loads_checked_license_files(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    _write_package(tmp_path, payload)

    package = load_runtime_package_v2(tmp_path)

    assert package.metadata.license_files == ["licenses/APACHE-2.0.txt"]
    assert package.detector_path == tmp_path.resolve() / "detector.onnx"


def test_runtime_package_v2_loads_optional_detector_classifier_consensus(
    tmp_path: Path,
) -> None:
    payload = _metadata(tmp_path)
    payload["quality"]["detector_classifier_consensus"] = {
        "minimum_detector_score": 0.735,
        "approved_disagreement_maximum_classifier_score": 0.8,
        "promote_safe_unknown_on_top3_consensus": True,
        "promote_recapture_on_top1_consensus": True,
        "inject_detector_class_into_unknown_top3": True,
    }
    _write_package(tmp_path, payload)

    package = load_runtime_package_v2(tmp_path)

    consensus = package.metadata.quality.detector_classifier_consensus
    assert consensus is not None
    assert consensus.minimum_detector_score == 0.735
    assert consensus.approved_disagreement_maximum_classifier_score == 0.8
    assert consensus.promote_safe_unknown_on_top3_consensus is True


def test_class_agnostic_runtime_decouples_detector_and_classifier_class_counts(
    tmp_path: Path,
) -> None:
    payload = _metadata(tmp_path)
    payload["detector_class_mode"] = "class_agnostic"
    payload["detector_class_count"] = 1
    payload["classifier_policy"]["ridge_approval_thresholds"] = [0.7, None]

    metadata = RuntimePackageV2Metadata.model_validate(payload)

    assert metadata.detector_class_count == 1
    assert metadata.classifier_policy.ridge_approval_thresholds == [0.7, None]


def test_class_agnostic_runtime_rejects_class_coupled_policies(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["detector_class_mode"] = "class_agnostic"
    payload["detector_class_count"] = 1
    payload["quality"]["detector_classifier_consensus"] = {
        "minimum_detector_score": 0.7,
        "approved_disagreement_maximum_classifier_score": 0.8,
    }

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


def test_runtime_accepts_bounded_detector_primary_classifier_routing(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["detector_primary_classifier_routing"] = {
        "direct_approval_class_indices": [0, 1],
        "minimum_detector_score": 0.98,
        "require_unique_class_per_image": True,
    }

    metadata = RuntimePackageV2Metadata.model_validate(payload)

    routing = metadata.detector_primary_classifier_routing
    assert routing is not None
    assert routing.direct_approval_class_indices == [0, 1]
    assert routing.minimum_detector_score == 0.98


@pytest.mark.parametrize(
    "routing",
    [
        {"direct_approval_class_indices": [0, 0]},
        {"direct_approval_class_indices": [2]},
    ],
)
def test_runtime_rejects_invalid_detector_primary_classes(tmp_path: Path, routing: dict) -> None:
    payload = _metadata(tmp_path)
    payload["detector_primary_classifier_routing"] = routing

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


def test_class_agnostic_runtime_rejects_detector_primary_routing(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["detector_class_mode"] = "class_agnostic"
    payload["detector_class_count"] = 1
    payload["detector_primary_classifier_routing"] = {"direct_approval_class_indices": [0]}

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


def test_runtime_package_v2_loads_checked_classifier_resolution_fallback(
    tmp_path: Path,
) -> None:
    payload = _metadata(tmp_path)
    payload["embedder"]["input_size"] = [192, 192]
    fallback_path = tmp_path / "embedder-fallback.onnx"
    fallback_path.write_bytes(b"fallback embedder")
    fallback_embedder = dict(payload["embedder"])
    fallback_embedder.update(
        filename="embedder-fallback.onnx",
        input_size=[224, 224],
    )
    payload["classifier_resolution_fallback"] = {
        "embedder": fallback_embedder,
        "fallback_on_unknown": True,
        "fallback_on_unsafe": True,
        "minimum_detector_support": 3,
        "approval_disagreement_rules": [
            {
                "minimum_detection_count": 5,
                "maximum_detection_count": 5,
                "maximum_approval_score": 0.6,
                "require_detector_disagreement": False,
                "minimum_box_aspect_ratio": 3.0,
                "maximum_approval_score_decrease": 0.2,
            }
        ],
        "fuse_unapproved_top3": True,
        "minimum_fallback_approval_score": 0.08,
    }
    payload["checksums"]["embedder-fallback.onnx"] = sha256_file(fallback_path)
    _write_package(tmp_path, payload)

    package = load_runtime_package_v2(tmp_path)

    assert package.classifier_fallback_embedder_path == fallback_path.resolve()
    assert package.metadata.classifier_resolution_fallback is not None
    assert package.metadata.classifier_resolution_fallback.fallback_on_unsafe is True
    assert package.metadata.classifier_resolution_fallback.selective_roi_only is False
    assert package.metadata.classifier_resolution_fallback.fuse_unapproved_top3 is True
    assert package.metadata.classifier_resolution_fallback.minimum_fallback_approval_score == 0.08
    assert (
        package.metadata.classifier_resolution_fallback.approval_disagreement_rules[
            0
        ].minimum_detection_count
        == 5
    )
    assert (
        package.metadata.classifier_resolution_fallback.approval_disagreement_rules[
            0
        ].maximum_detection_count
        == 5
    )
    assert (
        package.metadata.classifier_resolution_fallback.approval_disagreement_rules[
            0
        ].require_detector_disagreement
        is False
    )
    assert (
        package.metadata.classifier_resolution_fallback.approval_disagreement_rules[
            0
        ].minimum_box_aspect_ratio
        == 3.0
    )
    assert (
        package.metadata.classifier_resolution_fallback.approval_disagreement_rules[
            0
        ].maximum_approval_score_decrease
        == 0.2
    )


def test_classifier_resolution_fallback_accepts_selective_roi_execution(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["embedder"]["input_size"] = [192, 192]
    fallback_path = tmp_path / "embedder-fallback.onnx"
    fallback_path.write_bytes(b"fallback embedder")
    fallback_embedder = dict(payload["embedder"])
    fallback_embedder.update(filename="embedder-fallback.onnx", input_size=[224, 224])
    payload["classifier_resolution_fallback"] = {
        "embedder": fallback_embedder,
        "fallback_on_unknown": True,
        "selective_roi_only": True,
    }
    payload["checksums"]["embedder-fallback.onnx"] = sha256_file(fallback_path)
    _write_package(tmp_path, payload)

    package = load_runtime_package_v2(tmp_path)

    assert package.metadata.classifier_resolution_fallback is not None
    assert package.metadata.classifier_resolution_fallback.selective_roi_only is True


def test_classifier_resolution_fallback_requires_higher_resolution(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    fallback_embedder = dict(payload["embedder"])
    fallback_embedder["filename"] = "embedder-fallback.onnx"
    payload["classifier_resolution_fallback"] = {
        "embedder": fallback_embedder,
        "fallback_on_unknown": True,
    }

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


def test_cross_architecture_detail_requires_explicit_checksummed_catalog(tmp_path):
    payload = _metadata(tmp_path)
    payload["embedder"]["input_size"] = [192, 192]
    fallback = {
        **payload["embedder"],
        "filename": "detail.onnx",
        "input_size": [224, 224],
        "embedder_id": "different-detail-space",
    }
    payload["classifier_resolution_fallback"] = {"embedder": fallback}
    with pytest.raises(ValidationError, match="architecture"):
        RuntimePackageV2Metadata.model_validate(payload)
    payload["classifier_resolution_fallback"].update(
        catalog_directory="detail-catalog", catalog_checksums_sha256="a" * 64
    )
    accepted = RuntimePackageV2Metadata.model_validate(payload)
    assert accepted.classifier_resolution_fallback.embedder.embedder_id == "different-detail-space"


@pytest.mark.parametrize(
    "directory,digest",
    [("../outside", "a" * 64), ("detail", None), (None, "a" * 64), ("detail", "bad")],
)
def test_separate_detail_catalog_rejects_incomplete_or_unsafe_reference(
    tmp_path, directory, digest
):
    payload = _metadata(tmp_path)
    payload["embedder"]["input_size"] = [192, 192]
    payload["classifier_resolution_fallback"] = {
        "embedder": {**payload["embedder"], "filename": "detail.onnx", "input_size": [224, 224]},
        "catalog_directory": directory,
        "catalog_checksums_sha256": digest,
    }
    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


def test_runtime_package_v2_accepts_attested_production_status(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["promotion_status"] = "production"
    _write_package(tmp_path, payload)

    assert load_runtime_package_v2(tmp_path).metadata.promotion_status == "production"


def test_runtime_package_v2_rejects_unchecked_license_file(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["checksums"].pop("licenses/APACHE-2.0.txt")
    _write_package(tmp_path, payload)

    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(tmp_path)


def test_runtime_package_v2_rejects_tampered_license_file(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    _write_package(tmp_path, payload)
    (tmp_path / "licenses" / "APACHE-2.0.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(tmp_path)


def test_runtime_package_v2_rejects_license_path_escape(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["license_files"] = ["../outside.txt"]

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


def test_runtime_package_v2_rejects_model_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    payload = _metadata(root)
    payload["detector"]["filename"] = "../outside.onnx"
    payload["checksums"].pop("detector.onnx")
    payload["checksums"]["../outside.onnx"] = sha256_file(outside)
    _write_package(root, payload)

    with pytest.raises(PackageValidationError):
        load_runtime_package_v2(root)


def test_pair_probability_policy_requires_stricter_disagreement_threshold() -> None:
    payload = {
        "version": "2.0.0-rc.8",
        "prototype_weight": 0.5,
        "support_top_k": 3,
        "approval_minimum_similarity": 1.0,
        "approval_minimum_margin": 0.1,
        "ood_maximum_similarity": 0.4,
        "top3_minimum_similarity": -1.0,
        "catalog_conflict_similarity": 0.95,
        "ridge_approval_metric": "top2_pair_probability",
        "ridge_approval_minimum_pair_probability": 0.55,
        "ridge_disagreement_minimum_pair_probability": 0.54,
    }

    with pytest.raises(ValidationError):
        CatalogDecisionPolicy.model_validate(payload)


def test_margin_policy_requires_stricter_disagreement_threshold() -> None:
    payload = {
        "version": "0.1.2",
        "prototype_weight": 0.5,
        "support_top_k": 3,
        "approval_minimum_similarity": 1.0,
        "approval_minimum_margin": 0.1,
        "ood_maximum_similarity": -1.0,
        "top3_minimum_similarity": -1.0,
        "catalog_conflict_similarity": 0.95,
        "ridge_approval_minimum_margin": 0.2,
        "ridge_disagreement_minimum_margin": 0.19,
    }

    with pytest.raises(ValidationError):
        CatalogDecisionPolicy.model_validate(payload)


def test_detector_corroboration_requires_both_global_thresholds() -> None:
    payload = {
        "version": "0.1.2",
        "prototype_weight": 0.5,
        "support_top_k": 3,
        "approval_minimum_similarity": 1.0,
        "approval_minimum_margin": 0.1,
        "ood_maximum_similarity": -1.0,
        "top3_minimum_similarity": -1.0,
        "catalog_conflict_similarity": 0.95,
        "ridge_approval_minimum_margin": 0.0,
        "detector_corroboration_minimum_score": 0.93,
    }

    with pytest.raises(ValidationError):
        CatalogDecisionPolicy.model_validate(payload)


def test_low_similarity_detector_corroboration_requires_all_thresholds() -> None:
    payload = {
        "version": "0.1.2",
        "prototype_weight": 0.5,
        "support_top_k": 3,
        "approval_minimum_similarity": 1.0,
        "approval_minimum_margin": 0.1,
        "ood_maximum_similarity": -1.0,
        "top3_minimum_similarity": -1.0,
        "catalog_conflict_similarity": 0.95,
        "ridge_approval_minimum_margin": 0.0,
        "detector_corroboration_low_similarity_minimum_score": 0.7,
        "detector_corroboration_low_similarity_maximum_retrieval": 0.8,
    }

    with pytest.raises(ValidationError):
        CatalogDecisionPolicy.model_validate(payload)


def test_runtime_package_v2_accepts_per_class_ridge_thresholds(tmp_path: Path) -> None:
    payload = _metadata(tmp_path)
    payload["classifier_policy"]["ridge_approval_thresholds"] = [0.715, None]

    metadata = RuntimePackageV2Metadata.model_validate(payload)

    assert metadata.classifier_policy.ridge_approval_thresholds == [0.715, None]


@pytest.mark.parametrize(
    "thresholds",
    (
        [-0.1, None],
        [1.1, None],
    ),
)
def test_runtime_package_v2_rejects_invalid_per_class_ridge_thresholds(
    tmp_path: Path,
    thresholds: list[float | None],
) -> None:
    payload = _metadata(tmp_path)
    payload["classifier_policy"]["ridge_approval_thresholds"] = thresholds

    with pytest.raises(ValidationError):
        RuntimePackageV2Metadata.model_validate(payload)


@pytest.mark.parametrize(
    "output,threshold,valid",
    [
        (None, None, True),
        ("multi_object_probabilities", 0.8, True),
        (None, 0.8, False),
        ("multi_object_probabilities", None, False),
        ("embeddings", 0.8, False),
    ],
)
def test_roi_integrity_metadata_requires_matching_output_and_policy(
    tmp_path, output, threshold, valid
):
    payload = _metadata(tmp_path)
    payload["embedder"]["multi_object_output_name"] = output
    payload["quality"]["multi_object_recapture_threshold"] = threshold
    if valid:
        RuntimePackageV2Metadata.model_validate(payload)
    else:
        with pytest.raises(ValidationError):
            RuntimePackageV2Metadata.model_validate(payload)
