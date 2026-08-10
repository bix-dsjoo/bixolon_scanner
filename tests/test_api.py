from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from bixolon_scanner.api import create_app
from bixolon_scanner.config import WorkerSettings
from bixolon_scanner.inference import Detection, DetectionResult
from bixolon_scanner.pipeline import DecisionPipeline


class Detector:
    version = "1.0.0"

    def detect(self, image):
        return DetectionResult([Detection(10, 10, 50, 50, 0.99)])


class Classifier:
    version = "1.0.0"

    def classify(self, image, detections):
        return np.asarray([[8.0, 0.0, -1.0]], dtype=np.float32)


def _jpeg() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (100, 100), (128, 128, 128)).save(stream, format="JPEG")
    return stream.getvalue()


def test_scan_contract(classifier_metadata, quality_metadata):
    pipeline = DecisionPipeline(Detector(), Classifier(), classifier_metadata, quality_metadata)
    app = create_app(settings=WorkerSettings(), pipeline=pipeline)
    with TestClient(app) as client:
        response = client.post("/v1/scan", files={"image": ("scan.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["items"][0]["prediction"]["class_id"] == "bread_01"
    assert "prediction" not in body


def test_missing_image_uses_common_error_response(classifier_metadata, quality_metadata):
    pipeline = DecisionPipeline(Detector(), Classifier(), classifier_metadata, quality_metadata)
    app = create_app(settings=WorkerSettings(), pipeline=pipeline)
    with TestClient(app) as client:
        response = client.post("/v1/scan")
    assert response.status_code == 422
    assert response.json()["status"] == "ERROR"
    assert response.json()["reason_codes"] == ["MISSING_IMAGE_FIELD"]


def test_unsupported_format_is_415(classifier_metadata, quality_metadata):
    pipeline = DecisionPipeline(Detector(), Classifier(), classifier_metadata, quality_metadata)
    app = create_app(settings=WorkerSettings(), pipeline=pipeline)
    with TestClient(app) as client:
        response = client.post("/v1/scan", files={"image": ("file.gif", b"GIF89a", "image/gif")})
    assert response.status_code in {415, 422}
    assert response.json()["status"] == "ERROR"

