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
from bixolon_scanner.contracts.errors import ProviderInitializationError
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
            count_verifier=None,
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
    monkeypatch.setattr(
        worker_api,
        "build_catalog_classifier",
        lambda *args, **kwargs: (CatalogClassifier(), WarmEmbedder()),
    )

    app = create_app(
        settings=WorkerSettings(package_dir=package_dir, catalog_dir=catalog_dir),
    )
    with TestClient(app) as client:
        assert events == ["detector", "embedder"]
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["worker_version"] == "2.0.0"


def test_v2_runtime_falls_back_to_detector_provider_when_gpu_embedder_fails(
    tmp_path,
    monkeypatch,
    classifier_metadata,
    quality_metadata,
):
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
            count_verifier=None,
            input=SimpleNamespace(jpeg_draft_size=1200),
        )
    )
    catalog = SimpleNamespace(metadata=SimpleNamespace(catalog_version="2.0.0"))
    attempted_providers: list[str] = []
    closed_providers: list[str] = []
    detector_count_providers: list[tuple[str, bool]] = []

    class WarmDetector(Detector):
        version = "2.0.0"

        def warmup(self):
            return None

    class WarmEmbedder:
        def __init__(self, provider):
            self.provider = provider

        def warmup(self):
            if self.provider == "openvino_gpu":
                raise ProviderInitializationError
            return None

    class CatalogClassifier(Classifier):
        version = "2.0.0"

        def __init__(self, provider):
            self.provider = provider
            self.metadata = classifier_metadata

        def close(self):
            closed_providers.append(self.provider)

    def build_classifier(*args, **kwargs):
        del kwargs
        selected_provider = args[2]
        attempted_providers.append(selected_provider)
        return CatalogClassifier(selected_provider), WarmEmbedder(selected_provider)

    def build_detector(*args, **kwargs):
        del args
        detector_count_providers.append(
            (
                kwargs.get("count_verifier_provider"),
                kwargs.get("parallel_count_verifier", False),
            )
        )
        return WarmDetector()

    monkeypatch.setattr(worker_api, "load_runtime_package_v2", lambda _: runtime)
    monkeypatch.setattr(worker_api, "load_store_catalog_package", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(worker_api, "select_provider", lambda value: value)
    monkeypatch.setattr(worker_api, "build_detector_v2", build_detector)
    monkeypatch.setattr(worker_api, "build_catalog_classifier", build_classifier)

    app = create_app(
        settings=WorkerSettings(
            package_dir=package_dir,
            catalog_dir=catalog_dir,
            provider="openvino",
            embedder_provider="openvino_gpu",
            embedder_fallback_provider="same",
        ),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["provider"] == "openvino"
    assert attempted_providers == ["openvino_gpu", "openvino"]
    assert closed_providers == ["openvino_gpu", "openvino"]
    assert detector_count_providers == [(None, False)]


def test_v2_runtime_runs_gpu_presence_in_parallel_after_gpu_embedder_warmup(
    tmp_path,
    monkeypatch,
    classifier_metadata,
    quality_metadata,
):
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
            count_verifier=SimpleNamespace(),
            input=SimpleNamespace(jpeg_draft_size=1200),
        )
    )
    catalog = SimpleNamespace(metadata=SimpleNamespace(catalog_version="2.0.0"))
    events: list[str] = []

    class WarmDetector(Detector):
        version = "2.0.0"

        def warmup(self):
            events.append("detector")

    class WarmEmbedder:
        def warmup(self):
            events.append("embedder")

    class CatalogClassifier(Classifier):
        version = "2.0.0"

        def __init__(self):
            self.metadata = classifier_metadata

    def build_detector(*args, **kwargs):
        del args
        assert kwargs.get("count_verifier_provider") is None
        assert kwargs.get("parallel_count_verifier", False) is False
        return WarmDetector()

    def replace_count_verifier(detector, package, selected_provider, *args, **kwargs):
        del detector, package, args
        assert selected_provider == "openvino_gpu"
        assert kwargs["parallel_verification"] is True
        events.append("presence")

    monkeypatch.setattr(worker_api, "load_runtime_package_v2", lambda _: runtime)
    monkeypatch.setattr(worker_api, "load_store_catalog_package", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(worker_api, "select_provider", lambda value: value)
    monkeypatch.setattr(worker_api, "build_detector_v2", build_detector)
    monkeypatch.setattr(worker_api, "replace_count_verifier_v2", replace_count_verifier)
    monkeypatch.setattr(
        worker_api,
        "build_catalog_classifier",
        lambda *args, **kwargs: (CatalogClassifier(), WarmEmbedder()),
    )

    app = create_app(
        settings=WorkerSettings(
            package_dir=package_dir,
            catalog_dir=catalog_dir,
            provider="openvino",
            embedder_provider="openvino_gpu",
        ),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["provider"] == "openvino+openvino_gpu"
    assert events == ["detector", "embedder", "presence"]
