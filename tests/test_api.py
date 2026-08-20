from __future__ import annotations

import json
import threading
import time
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from bixolon_scanner.api import create_app
from bixolon_scanner.config import WorkerSettings
from bixolon_scanner.inference import Detection, DetectionResult
from bixolon_scanner.pipeline import DecisionPipeline
from bixolon_scanner.worker import api as worker_api


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
    assert body["status"] == "SEGMENTATION"
    assert body["segmentations"][0]["prediction"]["class_id"] == "bread_01"
    assert body["worker_version"] == "1.0.0"
    assert body["detector_version"] == "1.0.0"
    assert body["classifier_version"] == "1.0.0"
    assert "items" not in body
    assert "model_versions" not in body
    assert "prediction" not in body


def test_ready_contract_includes_independent_versions(classifier_metadata, quality_metadata):
    pipeline = DecisionPipeline(Detector(), Classifier(), classifier_metadata, quality_metadata)
    app = create_app(settings=WorkerSettings(), pipeline=pipeline)
    with TestClient(app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "provider": "injected",
        "worker_version": "1.0.0",
        "detector_version": "1.0.0",
        "classifier_version": "1.0.0",
    }


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


def test_timeout_keeps_inference_slot_until_background_scan_finishes(
    classifier_metadata,
    quality_metadata,
):
    started = threading.Event()
    release = threading.Event()

    class BlockingDetector(Detector):
        def detect(self, image):
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test detector was not released")
            return super().detect(image)

    pipeline = DecisionPipeline(
        BlockingDetector(),
        Classifier(),
        classifier_metadata,
        quality_metadata,
    )
    app = create_app(
        settings=WorkerSettings(request_timeout_seconds=0.02),
        pipeline=pipeline,
    )
    with TestClient(app) as client:
        response = client.post("/v1/scan", files={"image": ("scan.jpg", _jpeg(), "image/jpeg")})
        assert started.is_set()
        assert response.status_code == 500
        assert response.json()["reason_codes"] == ["MODEL_EXECUTION_FAILED"]
        assert app.state.semaphore.locked()

        release.set()
        deadline = time.monotonic() + 1
        while app.state.semaphore.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not app.state.semaphore.locked()


def test_v2_runtime_warms_models_before_readiness(
    tmp_path,
    monkeypatch,
    classifier_metadata,
    quality_metadata,
):
    events: list[str] = []
    package_dir = tmp_path / "runtime"
    package_dir.mkdir()
    (package_dir / "metadata.json").write_text(
        json.dumps({"schema_version": "2.0"}),
        encoding="utf-8",
    )
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    runtime = SimpleNamespace(
        metadata=SimpleNamespace(
            quality=quality_metadata,
            worker_version="2.0.0",
            embedder=SimpleNamespace(version="2.0.0"),
            detector_policy_version="2.0.0",
            classifier_policy=SimpleNamespace(version="2.0.0"),
            input=SimpleNamespace(jpeg_draft_size=1200),
        )
    )
    catalog = SimpleNamespace(metadata=SimpleNamespace(catalog_version="2.0.0"))

    class WarmDetector(Detector):
        version = "2.0.0"

        def warmup(self):
            events.append("detector")

    class WarmEmbedder:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def warmup(self):
            events.append("embedder")

    class CatalogClassifier(Classifier):
        version = "2.0.0"

        def __init__(self, *args):
            del args
            self.metadata = classifier_metadata

    monkeypatch.setattr(worker_api, "load_runtime_package_v2", lambda _: runtime)
    monkeypatch.setattr(worker_api, "load_store_catalog_package", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(worker_api, "select_provider", lambda _: "cpu")
    monkeypatch.setattr(
        worker_api,
        "build_detector_v2",
        lambda *args, **kwargs: WarmDetector(),
    )
    monkeypatch.setattr(worker_api, "OnnxEmbedder", WarmEmbedder)
    monkeypatch.setattr(worker_api, "OnnxCatalogClassifier", CatalogClassifier)

    app = create_app(
        settings=WorkerSettings(package_dir=package_dir, catalog_dir=catalog_dir),
    )
    with TestClient(app) as client:
        assert events == ["detector", "embedder"]
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["worker_version"] == "2.0.0"
