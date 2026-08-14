from __future__ import annotations

from bixolon_scanner.training.dfine_package import (
    apply_staged_classifier_metadata,
    dfine_package_metadata,
)


def test_dfine_package_metadata_keeps_classifier_lineage_and_marks_development():
    base = {
        "worker_version": "1.0.0-rc.4",
        "promotion_status": "production",
        "promotion": {"decision": "old"},
        "detector": {"filename": "detector.onnx", "version": "1.0.0-rc.4"},
        "classifier": {"filename": "classifier.onnx", "version": "1.0.0-rc.1"},
        "checksums": {"detector.onnx": "old", "classifier.onnx": "classifier-hash"},
    }

    result = dfine_package_metadata(
        base,
        worker_version="1.0.0-rc.9",
        detector_version="1.0.0-rc.9",
        detector_sha256="detector-hash",
        score_threshold=0.42,
        input_size=(768, 768),
        detector_evaluation={
            "metrics": {"recall": 0.99, "precision": 0.98, "count_accuracy": 0.9},
            "target_recall_satisfied": True,
        },
        source_revision="revision",
        checkpoint_filename="best_stg1.pth",
        checkpoint_sha256="checkpoint-hash",
    )

    assert result["worker_version"] == "1.0.0-rc.9"
    assert result["promotion_status"] == "development"
    assert "promotion" not in result
    assert result["detector"]["version"] == "1.0.0-rc.9"
    assert result["detector"]["score_threshold"] == 0.42
    assert result["detector"]["input_size"] == [768, 768]
    assert result["classifier"]["version"] == "1.0.0-rc.1"
    assert result["checksums"] == {
        "detector.onnx": "detector-hash",
        "classifier.onnx": "classifier-hash",
    }
    assert result["sources"]["detector"]["weight_sha256"] == "checkpoint-hash"
    assert result["sources"]["detector"]["weight_filename"] == "best_stg1.pth"
    assert result["detector_evaluation"] == {
        "recall": 0.99,
        "precision": 0.98,
        "count_accuracy": 0.9,
        "target_recall_satisfied": True,
    }


def test_apply_staged_classifier_metadata_records_runtime_policy():
    metadata = {
        "classifier": {"filename": "classifier.onnx", "version": "old"},
        "checksums": {"classifier.onnx": "old"},
        "calibration": {"sample_count": 1},
    }
    policy = {
        "final_policy": {
            "views": ["vflip", "rot15"],
            "top3_views": ["vflip", "rot15", "rot30"],
            "threshold": 0.99,
        },
        "staged_policy": {
            "first_view": "vflip",
            "early_approval_threshold": 0.999,
        },
    }

    apply_staged_classifier_metadata(
        metadata,
        classifier_version="1.0.0",
        classifier_sha256="classifier-hash",
        policy=policy,
        checkpoint_filename="best.pt",
        checkpoint_sha256="checkpoint-hash",
    )

    classifier = metadata["classifier"]
    assert classifier["version"] == "1.0.0"
    assert classifier["approval_threshold"] == 0.99
    assert classifier["staged_inference"]["final_views"] == ["vflip", "rot15"]
    assert classifier["staged_inference"]["top3_views"] == ["vflip", "rot15", "rot30"]
    assert metadata["checksums"]["classifier.onnx"] == "classifier-hash"
    assert "calibration" not in metadata


def test_package_metadata_records_independent_training_pipeline_provenance():
    base = {
        "worker_version": "1.0.0",
        "promotion_status": "development",
        "detector": {"filename": "detector.onnx", "version": "1.0.0"},
        "classifier": {"filename": "classifier.onnx", "version": "1.0.0"},
        "checksums": {"detector.onnx": "old", "classifier.onnx": "classifier-hash"},
    }
    result = dfine_package_metadata(
        base,
        worker_version="1.0.1",
        detector_version="1.1.0",
        detector_sha256="detector-hash",
        score_threshold=0.5,
        input_size=(640, 640),
        detector_evaluation={
            "metrics": {"recall": 1.0, "precision": 1.0, "count_accuracy": 1.0},
            "target_recall_satisfied": True,
        },
        source_revision="a" * 40,
        checkpoint_filename="best.pth",
        checkpoint_sha256="b" * 64,
        training_pipeline_version="1.0.0",
        training_contract_sha256="c" * 64,
        training_dataset_version="bread-test-1",
        training_manifest_sha256="d" * 64,
    )

    assert result["sources"]["detector"]["training_pipeline_version"] == "1.0.0"
    assert result["sources"]["detector"]["training_contract_sha256"] == "c" * 64
    assert result["sources"]["detector"]["training_dataset_version"] == "bread-test-1"


def test_package_metadata_requires_complete_training_pipeline_provenance():
    import pytest

    base = {
        "detector": {"filename": "detector.onnx"},
        "classifier": {"filename": "classifier.onnx"},
        "checksums": {"classifier.onnx": "classifier-hash"},
    }
    with pytest.raises(ValueError, match="required together"):
        dfine_package_metadata(
            base,
            worker_version="1.0.0",
            detector_version="1.0.0",
            detector_sha256="detector-hash",
            score_threshold=0.5,
            input_size=(640, 640),
            detector_evaluation={
                "metrics": {"recall": 1.0, "precision": 1.0, "count_accuracy": 1.0},
                "target_recall_satisfied": True,
            },
            source_revision="a" * 40,
            checkpoint_filename="best.pth",
            checkpoint_sha256="b" * 64,
            training_pipeline_version="1.0.0",
        )
