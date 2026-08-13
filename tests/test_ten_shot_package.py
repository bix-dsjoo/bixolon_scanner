from __future__ import annotations

import copy

import pytest

from bixolon_scanner.training.ten_shot_package import build_ten_shot_metadata


def _base():
    return {
        "schema_version": "1.1",
        "package_version": "0.1.1",
        "promotion_status": "production",
        "dataset_version": "old",
        "input": {"jpeg_draft_size": 1500},
        "detector": {"filename": "detector.onnx", "version": "0.1.0", "score_threshold": 0.56},
        "classifier": {
            "filename": "classifier.onnx",
            "version": "0.1.0",
            "crop_margin_ratio": 0.05,
            "resize_reducing_gap": 1.0,
            "warmup_batch_sizes": [1, 2],
        },
        "quality": {"border_policy": "classifier_confidence"},
        "checksums": {"detector.onnx": "d", "classifier.onnx": "old-c"},
        "sources": {"classifier": {}},
        "calibration": {},
        "promotion": {"decision": "approved"},
    }


def _manifest():
    return {
        "dataset_version": "bread-10shot-abc",
        "manifest_sha256": "m" * 64,
        "labels": [
            {"category_id": 1, "class_id": "bread_01", "class_name": "One"},
            {"category_id": 2, "class_id": "bread_02", "class_name": "Two"},
        ],
    }


def _checkpoint():
    return {
        "architecture": "ten_shot_residual_cosine",
        "dataset_version": "bread-10shot-abc",
        "manifest_sha256": "m" * 64,
        "feature_cache_sha256": "f" * 64,
        "num_classes": 2,
        "image_size": 224,
        "backbone_kind": "dinov3_convnext_tiny",
        "source_revision": "revision",
        "source_weight_filename": "weights.pth",
        "source_weight_sha256": "w" * 64,
    }


def _calibration():
    return {
        "approval_threshold": 0.9,
        "temperature": 0.5,
        "sample_count": 100,
        "approved_precision": 1.0,
        "approval_coverage": 0.8,
        "approved_false_rate_upper_95": 0.004,
        "risk_control_satisfied": True,
    }


def test_ten_shot_metadata_changes_only_classifier_side_of_base_contract():
    base = _base()
    metadata = build_ten_shot_metadata(
        base_metadata=base,
        manifest_metadata=_manifest(),
        calibration=_calibration(),
        classifier_sha256="new-c",
        classifier_version="0.2.0",
        package_version="0.2.0",
        checkpoint=_checkpoint(),
    )
    assert metadata["detector"] == base["detector"]
    assert metadata["quality"] == base["quality"]
    assert metadata["input"] == base["input"]
    assert metadata["checksums"]["detector.onnx"] == "d"
    assert metadata["classifier"]["version"] == "0.2.0"
    assert metadata["sources"]["classifier"]["architecture"] == ("ten_shot_residual_cosine")
    assert metadata["promotion_status"] == "development"
    assert "promotion" not in metadata


def test_ten_shot_metadata_records_embedded_crop_without_changing_public_input():
    base = _base()
    metadata = build_ten_shot_metadata(
        base_metadata=base,
        manifest_metadata=_manifest(),
        calibration=_calibration(),
        classifier_sha256="new-c",
        classifier_version="0.2.3",
        package_version="0.2.3",
        checkpoint=_checkpoint(),
        inference_center_crop_scale=0.88,
    )
    assert metadata["classifier"]["input_size"] == [224, 224]
    assert metadata["sources"]["classifier"]["architecture"] == (
        "ten_shot_residual_cosine+center_crop_resize_0.88"
    )


def test_ten_shot_metadata_rejects_checkpoint_manifest_mismatch():
    checkpoint = copy.deepcopy(_checkpoint())
    checkpoint["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest checksum"):
        build_ten_shot_metadata(
            base_metadata=_base(),
            manifest_metadata=_manifest(),
            calibration=_calibration(),
            classifier_sha256="new-c",
            classifier_version="0.2.0",
            package_version="0.2.0",
            checkpoint=checkpoint,
        )


def test_manual_waiver_requires_explicit_human_approval_for_production_package():
    pending = build_ten_shot_metadata(
        base_metadata=_base(),
        manifest_metadata=_manifest(),
        calibration=_calibration(),
        classifier_sha256="new-c",
        classifier_version="0.2.0",
        package_version="0.2.0",
        checkpoint=_checkpoint(),
        promotion_status="manual_waiver",
    )
    assert pending["promotion_status"] == "development"
    approved = build_ten_shot_metadata(
        base_metadata=_base(),
        manifest_metadata=_manifest(),
        calibration=_calibration(),
        classifier_sha256="new-c",
        classifier_version="0.2.0",
        package_version="0.2.0",
        checkpoint=_checkpoint(),
        promotion_status="manual_waiver",
        manual_waiver_approved=True,
    )
    assert approved["promotion_status"] == "production"
    assert approved["promotion"]["method"] == "manual_waiver"
