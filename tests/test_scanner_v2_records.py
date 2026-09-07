from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from bixolon_scanner.contracts import ItemStatus
from bixolon_scanner.evaluation.scanner_v2 import (
    RecordingClassifier,
    _approved_false_positive_count,
    _records,
)
from bixolon_scanner.pipeline.ports import ClassificationResult


def test_recording_classifier_preserves_runtime_fallback_capabilities() -> None:
    primary_result = ClassificationResult(
        logits=np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        ranking_logits=np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        approval_scores=np.asarray([0.8, 0.4], dtype=np.float32),
    )
    fallback_result = ClassificationResult(
        logits=np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        ranking_logits=np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=np.float32),
        approval_scores=np.asarray([0.9, 0.9], dtype=np.float32),
    )
    selected_result = ClassificationResult(
        logits=np.asarray([[4.0, 1.0]], dtype=np.float32),
        ranking_logits=np.asarray([[4.0, 1.0]], dtype=np.float32),
        approval_scores=np.asarray([0.95], dtype=np.float32),
    )

    class Classifier:
        version = "0.1.7"
        metadata = object()
        resolution_fallback_metadata = object()
        assisted_policy = "policy"

        def classify(self, image, detections):
            return primary_result

        def classify_fallback(self, image, detections):
            return fallback_result

        def classify_fallback_selected(self, image, detections, detection_indices):
            return selected_result

    wrapped = RecordingClassifier(Classifier())

    assert wrapped.resolution_fallback_metadata is not None
    assert wrapped.classify_fallback(None, []) is fallback_result
    assert wrapped.last_result is fallback_result
    wrapped.classify(None, [object(), object()])
    returned = wrapped.classify_fallback_selected(None, [object(), object()], [1])
    assert returned is selected_result
    assert wrapped.last_result.logits.tolist() == [[2.0, 1.0], [4.0, 1.0]]
    assert wrapped.last_result.approval_scores.tolist() == pytest.approx([0.8, 0.95])
    assert wrapped.assisted_policy == "policy"


def test_records_loads_coco_json(tmp_path) -> None:
    image = tmp_path / "images" / "one.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "images/one.jpg"}],
                "annotations": [{"image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )

    rows = _records(annotation, tmp_path)

    assert rows == [
        {
            "id": 1,
            "file_name": "images/one.jpg",
            "image_id": 1,
            "image_path": "images/one.jpg",
            "annotations": [{"bbox_xywh": [1.0, 2.0, 3.0, 4.0], "category_id": 7}],
            "resolved_path": image,
        }
    ]


def test_records_rejects_coco_path_escape(tmp_path) -> None:
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "../outside.jpg"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escaped"):
        _records(annotation, tmp_path)


def test_records_resolves_coco_path_relative_to_annotation(tmp_path) -> None:
    image = tmp_path / "images" / "one.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    annotation = tmp_path / "annotations" / "instances.json"
    annotation.parent.mkdir()
    annotation.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "../images/one.jpg"}]}),
        encoding="utf-8",
    )

    rows = _records(annotation, tmp_path)

    assert rows[0]["image_path"] == "images/one.jpg"
    assert rows[0]["resolved_path"] == image


def test_approved_false_positive_count_only_counts_unmatched_approvals() -> None:
    segmentations = [
        SimpleNamespace(status=ItemStatus.APPROVED),
        SimpleNamespace(status=ItemStatus.UNKNOWN),
        SimpleNamespace(status=ItemStatus.APPROVED),
    ]

    assert _approved_false_positive_count(segmentations, {0: 2}) == 1
