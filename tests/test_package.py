from __future__ import annotations

import json

import pytest

from bixolon_scanner.errors import PackageValidationError
from bixolon_scanner.package import load_model_package, sha256_file


def _metadata(detector_sha: str, classifier_sha: str):
    return {
        "schema_version": "1.0",
        "package_version": "0.1.0",
        "promotion_status": "development",
        "dataset_version": "bread-test",
        "detector": {
            "filename": "detector.onnx",
            "version": "0.1.0",
            "score_threshold": 0.3,
            "nms_iou_threshold": 0.7,
            "max_object_aspect_ratio": 5.0,
            "max_queries": 300,
        },
        "classifier": {
            "filename": "classifier.onnx",
            "version": "0.1.0",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "approval_threshold": 0.95,
            "temperature": 1.0,
            "labels": [{"class_id": "bread_01", "class_name": "Walnut Donut"}],
        },
        "quality": {},
        "checksums": {"detector.onnx": detector_sha, "classifier.onnx": classifier_sha},
        "licenses": {"detector": "Apache-2.0", "classifier": "DINOv3 License"},
        "sources": {
            "classifier": {
                "architecture": "dinov3_convnext_tiny",
                "revision": "6876159a11b4df116f30f667f8c9888617df0751",
                "weight_filename": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
                "weight_sha256": "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54",
            }
        },
    }


def test_package_checksum_validation(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    package = load_model_package(tmp_path)
    assert package.metadata.promotion_status == "development"
    assert package.metadata.sources["classifier"].architecture == "dinov3_convnext_tiny"
    assert package.metadata.detector.max_object_aspect_ratio == 5.0

    detector.write_bytes(b"tampered")
    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_detector_aspect_ratio_policy_must_exceed_one(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["detector"]["max_object_aspect_ratio"] = 1.0
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_detector_ensemble_requires_every_member_checksum(tmp_path):
    detector = tmp_path / "detector.onnx"
    second = tmp_path / "detector-2.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    second.write_bytes(b"detector-2")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["detector"]["ensemble"] = {
        "members": [
            {"filename": "detector.onnx", "score_threshold": 0.02},
            {"filename": "detector-2.onnx", "score_threshold": 0.02},
        ],
        "fusion": {"cluster_iou_threshold": 0.5},
        "base_selection": {
            "score_threshold": 0.37,
            "nms_iou_threshold": 0.5,
            "containment_threshold": 0.85,
            "group_minimum": 2,
        },
        "ambiguity_union": [
            {
                "availability_score_threshold": 0.002,
                "availability_nms_iou_threshold": 0.5,
                "availability_containment_threshold": 0.9,
                "availability_group_minimum": 2,
                "minimum_selected_count": 5,
                "extra_candidate_count": 1,
                "extra_count_mode": "at_least",
                "next_score_threshold_inclusive": 0.05,
            }
        ],
        "class_verified_selector": {
            "candidate_minimum_score": 0.02,
            "candidate_minimum_support": 2,
            "candidate_duplicate_iou": 0.9,
            "base_match_iou": 0.9,
            "group_relation_iou": 0.3,
            "group_area_ratio": 0.8,
            "group_margin_ratio": 1.5,
            "group_novel_margin": 500.0,
            "group_minimum_score": 0.04,
            "independent_maximum_iou": 0.5,
            "independent_margin": 2000.0,
            "independent_minimum_score": 0.05,
            "unique_class_per_image_contract": True,
        },
        "maximum_box_area_ratio": 0.3,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)

    metadata["checksums"]["detector-2.onnx"] = sha256_file(second)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    package = load_model_package(tmp_path)
    assert package.detector_paths == [detector, second]


def test_detector_draft_refinement_requires_consensus_policy(tmp_path):
    detector = tmp_path / "detector.onnx"
    second = tmp_path / "detector-2.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    second.write_bytes(b"detector-2")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["detector"]["ensemble"] = {
        "members": [
            {"filename": "detector.onnx", "score_threshold": 0.02},
            {"filename": "detector-2.onnx", "score_threshold": 0.02},
        ],
        "fusion": {"cluster_iou_threshold": 0.5},
        "base_selection": {
            "score_threshold": 0.37,
            "nms_iou_threshold": 0.5,
            "containment_threshold": 0.85,
            "group_minimum": 2,
        },
        "draft_refinement": {
            "draft_size": 1500,
            "ambiguity_refinement_maximum_selected_count": 6,
            "maximum_agreeing_policy_count": 2,
            "minimum_selected_count": 7,
            "minimum_selected_box_aspect_ratio_extremity": 2.0,
            "consensus_ambiguity_bypass_maximum_selected_count": 4,
            "consensus_ambiguity_bypass_minimum_selected_score": 0.75,
            "unanimous_ambiguity_bypass_maximum_selected_count": 5,
            "unanimous_ambiguity_bypass_minimum_selected_score": 0.87,
            "full_resolution_unresolved_ambiguity_maximum_selected_count": 5,
            "full_resolution_on_selected_count_change": True,
        },
        "ambiguity_union": [
            {
                "availability_score_threshold": 0.002,
                "availability_nms_iou_threshold": 0.5,
                "availability_containment_threshold": 0.9,
                "availability_group_minimum": 2,
                "minimum_selected_count": 5,
                "extra_candidate_count": 1,
                "extra_count_mode": "at_least",
                "next_score_threshold_inclusive": 0.05,
            }
        ],
        "class_verified_selector": {
            "candidate_minimum_score": 0.02,
            "candidate_minimum_support": 2,
            "candidate_duplicate_iou": 0.9,
            "base_match_iou": 0.9,
            "group_relation_iou": 0.3,
            "group_area_ratio": 0.8,
            "group_margin_ratio": 1.5,
            "group_novel_margin": 500.0,
            "group_minimum_score": 0.04,
            "independent_maximum_iou": 0.5,
            "independent_margin": 2000.0,
            "independent_minimum_score": 0.05,
            "unique_class_per_image_contract": True,
        },
        "maximum_box_area_ratio": 0.3,
    }
    metadata["checksums"]["detector-2.onnx"] = sha256_file(second)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_production_package_requires_auditable_promotion_record(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["promotion_status"] = "production"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)

    metadata["promotion"] = {
        "decision": "approved",
        "method": "manual_waiver",
        "decided_on": "2026-08-10",
        "waivers": [
            {
                "gate": "unknown_top3_accuracy",
                "observed": 16 / 19,
                "target": 0.95,
                "sample_count": 19,
                "correct_count": 16,
                "reason": "Project owner accepted the residual risk from the small sample.",
            },
            {
                "gate": "evaluation_set_independence",
                "observed": 1 / 300,
                "target": 1.0,
                "sample_count": 300,
                "correct_count": 1,
                "reason": "Only one evaluation image is independent.",
            },
        ],
        "remaining_limitations": ["RECAPTURE recall is not certified."],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    package = load_model_package(tmp_path)
    assert package.metadata.promotion_status == "production"
    assert package.metadata.promotion is not None
    assert package.metadata.promotion.waivers[0].sample_count == 19
    assert package.metadata.promotion.waivers[1].gate == "evaluation_set_independence"


def test_optional_count_verifier_requires_valid_checksum(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    count_verifier = tmp_path / "count_verifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    count_verifier.write_bytes(b"count")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["count_verifier"] = {
        "filename": "count_verifier.onnx",
        "version": "0.1.0",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "count_labels": [3, 4, 5, 6, 7],
        "confidence_threshold": 0.95,
    }
    metadata["schema_version"] = "1.1"
    metadata["checksums"]["count_verifier.onnx"] = sha256_file(count_verifier)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    package = load_model_package(tmp_path)

    assert package.count_verifier_path == count_verifier
    count_verifier.write_bytes(b"tampered")
    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_schema_10_rejects_new_quality_policy(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["quality"]["border_policy"] = "classifier_confidence"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_contained_duplicate_policy_requires_schema_11_and_unit_interval(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["quality"]["duplicate_review_containment_threshold"] = 0.999
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)

    metadata["schema_version"] = "1.1"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    package = load_model_package(tmp_path)
    assert package.metadata.quality.duplicate_review_containment_threshold == 0.999

    metadata["quality"]["duplicate_review_containment_threshold"] = 1.01
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_schema_10_rejects_uncertainty_area_policy(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["detector"]["uncertainty_min_area_ratio"] = 0.039
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_legacy_detector_target_bundle_accepts_independent_model_versions(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata.update(
        {
            "schema_version": "1.1",
            "package_version": "0.2.5",
            "bundle_provenance": {
                "target_mode": "detector_safety_first_0.2.5",
                "model_version": "0.2.5",
                "classifier_source_version": "0.2.4",
                "classifier_source_sha256": sha256_file(classifier),
                "detector_selection_sha256": "a" * 64,
                "evaluation_dataset_versions": {
                    "natural": "natural-v1",
                    "hard": "hard-v1",
                    "shift": "shift-v1",
                },
            },
        }
    )
    metadata["detector"]["version"] = "0.2.5"
    metadata["classifier"]["version"] = "0.2.5"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    package = load_model_package(tmp_path)
    assert package.metadata.package_version == "0.2.5"
    assert package.metadata.detector.version == "0.2.5"
    assert package.metadata.classifier.version == "0.2.5"
    assert package.metadata.bundle_provenance.classifier_source_version == "0.2.4"
    assert package.metadata.bundle_provenance.evaluation_dataset_versions == {
        "natural": "natural-v1",
        "hard": "hard-v1",
        "shift": "shift-v1",
    }

    metadata["classifier"]["version"] = "0.2.4"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    package = load_model_package(tmp_path)
    assert package.metadata.detector.version == "0.2.5"
    assert package.metadata.classifier.version == "0.2.4"


def test_official_schema_requires_all_versions_to_start_at_one(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata["schema_version"] = "2.0"
    metadata["worker_version"] = "1.0.0"
    metadata.pop("package_version", None)
    metadata["detector"]["version"] = "1.0.0"
    metadata["classifier"]["version"] = "1.0.0"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert load_model_package(tmp_path).metadata.worker_version == "1.0.0"

    metadata["classifier"]["version"] = "0.9.0"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_model_source_training_pipeline_provenance_is_optional_but_atomic(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    source = metadata["sources"]["classifier"]
    source["training_pipeline_version"] = "1.0.0"
    source["training_contract_sha256"] = "a" * 64
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_model_package(tmp_path)
    assert loaded.metadata.sources["classifier"].training_pipeline_version == "1.0.0"

    source.pop("training_contract_sha256")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)


def test_schema_21_requires_both_component_training_contracts(tmp_path):
    detector = tmp_path / "detector.onnx"
    classifier = tmp_path / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier")
    metadata = _metadata(sha256_file(detector), sha256_file(classifier))
    metadata.update({"schema_version": "2.1", "worker_version": "1.0.0"})
    metadata.pop("package_version", None)
    metadata["detector"]["version"] = "1.0.0"
    metadata["classifier"]["version"] = "1.0.0"
    metadata["sources"]["detector"] = {
        "architecture": "D-FINE-N",
        "revision": "a" * 40,
        "weight_sha256": "b" * 64,
        "training_pipeline_version": "1.0.0",
        "training_contract_sha256": "c" * 64,
        "training_dataset_version": "bread-test-1",
        "training_manifest_sha256": "e" * 64,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PackageValidationError):
        load_model_package(tmp_path)

    metadata["sources"]["classifier"].update(
        {
            "training_pipeline_version": "1.0.0",
            "training_contract_sha256": "d" * 64,
            "training_dataset_version": "bread-test-1",
            "training_manifest_sha256": "f" * 64,
        }
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert load_model_package(tmp_path).metadata.schema_version == "2.1"
