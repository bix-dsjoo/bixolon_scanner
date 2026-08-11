from __future__ import annotations

import json

from bixolon_scanner.package import sha256_file
from bixolon_scanner.training.export import _copy_reused_classifier


def test_reused_classifier_is_copied_byte_for_byte(tmp_path):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    detector = package_dir / "detector.onnx"
    classifier = package_dir / "classifier.onnx"
    detector.write_bytes(b"detector")
    classifier.write_bytes(b"classifier-exact-bytes")
    metadata = {
        "schema_version": "1.1",
        "package_version": "0.1.1",
        "promotion_status": "development",
        "dataset_version": "bread-test",
        "detector": {
            "filename": detector.name,
            "version": "0.1.0",
            "score_threshold": 0.56,
            "uncertainty_score_threshold": 0.2,
            "uncertainty_min_area_ratio": 0.039,
            "uncertainty_match_iou_threshold": 0.5,
            "nms_iou_threshold": 0.7,
            "max_queries": 300,
        },
        "classifier": {
            "filename": classifier.name,
            "version": "0.1.0",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "approval_threshold": 0.95,
            "temperature": 1.0,
            "labels": [{"class_id": "bread_01", "class_name": "Walnut Donut"}],
        },
        "quality": {"border_policy": "classifier_confidence"},
        "checksums": {
            detector.name: sha256_file(detector),
            classifier.name: sha256_file(classifier),
        },
        "licenses": {"detector": "Apache-2.0", "classifier": "DINOv3 License"},
    }
    (package_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / "output" / "classifier.onnx"
    destination.parent.mkdir()

    copied_metadata = _copy_reused_classifier(package_dir, destination)

    assert destination.read_bytes() == classifier.read_bytes()
    assert copied_metadata["classifier"]["version"] == "0.1.0"
