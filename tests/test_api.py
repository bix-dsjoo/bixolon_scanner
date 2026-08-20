from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

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


def test_v2_worker_warms_models_before_readiness(tmp_path, monkeypatch):
    from bixolon_scanner.worker import api as worker_api

    package_dir = tmp_path / "runtime"
    package_dir.mkdir()
    (package_dir / "metadata.json").write_text('{"schema_version":"2.0"}', encoding="utf-8")
    events: list[str] = []

    class Warmable:
        def __init__(self, name: str) -> None:
            self.name = name

        def warmup(self) -> None:
            events.append(self.name)

    detector = Warmable("detector")
    embedder = Warmable("embedder")
    runtime = SimpleNamespace(
        metadata=SimpleNamespace(
            quality=object(),
            worker_version="2.0.0-rc.test",
            embedder=SimpleNamespace(version="2.0.0-rc.test"),
            detector_policy_version="2.0.0-rc.test",
            classifier_policy=SimpleNamespace(version="2.0.0-rc.test"),
            input=SimpleNamespace(jpeg_draft_size=1500),
        )
    )
    catalog = SimpleNamespace(metadata=SimpleNamespace(catalog_version="catalog.test"))
    classifier = SimpleNamespace(metadata=object())
    pipeline = SimpleNamespace(worker_version="2.0.0-rc.test")

    monkeypatch.setattr(worker_api, "load_runtime_package_v2", lambda _path: runtime)
    monkeypatch.setattr(worker_api, "load_store_catalog_package", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(worker_api, "select_provider", lambda _provider: "CUDAExecutionProvider")
    monkeypatch.setattr(worker_api, "build_detector_v2", lambda *args, **kwargs: detector)
    monkeypatch.setattr(worker_api, "OnnxEmbedder", lambda *args, **kwargs: embedder)

    def build_classifier(*args, **kwargs):
        assert events == ["detector", "embedder"]
        return classifier

    monkeypatch.setattr(worker_api, "OnnxCatalogClassifier", build_classifier)
    monkeypatch.setattr(worker_api, "DecisionPipeline", lambda *args, **kwargs: pipeline)

    app = create_app(
        settings=WorkerSettings(
            package_dir=package_dir,
            catalog_dir=tmp_path / "catalog",
            catalog_store_id="store",
            catalog_key_id="key",
            catalog_signing_key=SecretStr("secret"),
            provider="cuda",
        )
    )
    with TestClient(app):
        assert app.state.ready is True
        assert events == ["detector", "embedder"]
