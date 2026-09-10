from __future__ import annotations

import threading
import time
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bixolon_scanner.contracts.errors import ModelExecutionError
from bixolon_scanner.pipeline import DecisionPipeline
from bixolon_scanner.pipeline.classification import normalize_classification
from bixolon_scanner.pipeline.ports import ClassificationResult
from bixolon_scanner.runtime.catalog import OnnxEmbedder
from bixolon_scanner.runtime.onnx import OnnxDetector
from bixolon_scanner.worker.api import create_app
from bixolon_scanner.worker.settings import WorkerSettings


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("output", ["logits", "boxes"])
def test_detector_rejects_nonfinite_outputs(value, output):
    detector = object.__new__(OnnxDetector)
    logits = np.zeros((2, 1), dtype=np.float32)
    boxes = np.full((2, 4), 0.5, dtype=np.float32)
    (logits if output == "logits" else boxes)[0, 0] = value
    with pytest.raises(ModelExecutionError):
        detector._postprocess_detection_outputs(
            logits, boxes, original_width=100, original_height=100
        )


@pytest.mark.parametrize(
    "field",
    [
        "logits",
        "ranking_logits",
        "approval_scores",
        "ranking_scores",
        "retrieval_logits",
        "top3_safety_scores",
    ],
)
def test_classifier_rejects_nonfinite_evidence(classifier_metadata, field):
    values = {
        "logits": np.zeros((1, 3), dtype=np.float32),
        "ranking_logits": np.zeros((1, 3), dtype=np.float32),
    }
    shape = (1,) if field in {"approval_scores", "top3_safety_scores"} else (1, 3)
    values[field] = np.full(shape, np.nan, dtype=np.float32)
    with pytest.raises(ModelExecutionError):
        normalize_classification(
            ClassificationResult(**values), detection_count=1, metadata=classifier_metadata
        )


def test_explicit_consensus_class_mask_remains_valid(classifier_metadata):
    result = ClassificationResult(
        logits=np.array([[2.0, 1.0, -np.inf]], dtype=np.float32),
        ranking_logits=np.array([[2.0, 1.0, -np.inf]], dtype=np.float32),
        ranking_scores=np.array([[0.8, 0.2, 0.0]], dtype=np.float32),
        approval_scores=np.array([0.9], dtype=np.float32),
    )
    batch = normalize_classification(result, detection_count=1, metadata=classifier_metadata)
    assert batch.probabilities[0, 2] == 0
    assert batch.decision_indices[0, 0] == 0
    result.ranking_scores[0, 2] = 0.1
    with pytest.raises(ModelExecutionError):
        normalize_classification(result, detection_count=1, metadata=classifier_metadata)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_raw_embedder_rejects_nonfinite_model_tensors(value):
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        input_name="pixels", output_name="embeddings", embedding_dimension=3
    )
    embedder.runner = SimpleNamespace(
        run=lambda *args: (np.array([[1.0, value, 0.0]], dtype=np.float32),)
    )
    with pytest.raises(ModelExecutionError):
        embedder._run_raw_tensors(np.zeros((1, 3, 2, 2), dtype=np.float32))


def test_slow_decode_does_not_block_health_or_decode_waiting_requests(
    monkeypatch, classifier_metadata, quality_metadata
):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocked_decode(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(30)
        return Image.new("RGB", (100, 100))

    class Uncalled:
        version = "1.0.0"

        def detect(self, image):
            raise AssertionError("expired decode must not start detector")

    monkeypatch.setattr("bixolon_scanner.worker.api.decode_image", blocked_decode)
    pipeline = DecisionPipeline(Uncalled(), Uncalled(), classifier_metadata, quality_metadata)
    app = create_app(settings=WorkerSettings(request_timeout_seconds=0.1), pipeline=pipeline)
    stream = BytesIO()
    Image.new("RGB", (10, 10)).save(stream, format="PNG")
    with TestClient(app) as client:
        try:
            response = client.post("/v1/scan", files={"image": ("image.png", stream.getvalue())})
            assert entered.is_set()
            assert response.status_code == 500
            assert response.json()["status"] == "ERROR"
            assert client.get("/health/live").status_code == 200
            # Windows Python 3.11's asyncio clock (GetTickCount64) can fire a
            # timeout up to one 15.625ms tick before the perf_counter deadline
            # used by readiness. Assert the overdue state after that deadline,
            # while the deliberately blocked decode still owns the semaphore.
            assert app.state.semaphore.locked()
            assert app.state.inference_deadline is not None
            remaining = app.state.inference_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining + time.get_clock_info("monotonic").resolution)
            assert client.get("/health/ready").status_code == 503
            second = client.post("/v1/scan", files={"image": ("image.png", stream.getvalue())})
            assert second.status_code == 500
            assert calls == [1]
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while app.state.semaphore.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not app.state.semaphore.locked()
        assert client.get("/health/ready").status_code == 200


def test_unexpected_request_error_does_not_log_private_exception_text(
    monkeypatch, caplog, classifier_metadata, quality_metadata
):
    def broken_decode(*args, **kwargs):
        raise RuntimeError("private-customer-image-path")

    monkeypatch.setattr("bixolon_scanner.worker.api.decode_image", broken_decode)
    unused = SimpleNamespace(version="1.0.0")
    pipeline = DecisionPipeline(unused, unused, classifier_metadata, quality_metadata)
    with TestClient(create_app(pipeline=pipeline), raise_server_exceptions=False) as client:
        response = client.post("/v1/scan", files={"image": ("scan.png", b"bytes")})
    assert response.status_code == 500
    assert response.json()["status"] == "ERROR"
    assert "private-customer-image-path" not in response.text + caplog.text
    failures = [record for record in caplog.records if record.message == "request_failed"]
    assert len(failures) == 1
    assert failures[0].exception_type == "RuntimeError"
    assert failures[0].exc_info is None
