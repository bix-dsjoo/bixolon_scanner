from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..contracts import (
    ScanResponse,
    Status,
    load_runtime_package_v2,
    load_store_catalog_package,
)
from ..contracts.errors import MissingImageError, ModelExecutionError, ScannerError
from ..contracts.model_package import load_model_package
from ..pipeline import DecisionPipeline
from ..runtime.catalog import OnnxCatalogClassifier, OnnxEmbedder
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image
from ..runtime.onnx import build_onnx_adapters, select_provider
from .settings import WorkerSettings

LOGGER = logging.getLogger(__name__)


def _request_id(request: Request | None = None) -> str:
    if request is not None and hasattr(request.state, "request_id"):
        return request.state.request_id
    return uuid.uuid4().hex


def _error_response(
    request_id: str,
    reason_code: str,
    elapsed_ms: float,
    status_code: int,
    *,
    worker_version: str = __version__,
) -> JSONResponse:
    body = ScanResponse(
        request_id=request_id,
        status=Status.ERROR,
        reason_codes=[reason_code],
        segmentations=[],
        processing_time_ms=max(0.0, elapsed_ms),
        worker_version=worker_version,
        detector_version=None,
        classifier_version=None,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def create_app(
    *, settings: WorkerSettings | None = None, pipeline: DecisionPipeline | None = None
) -> FastAPI:
    worker_settings = settings or WorkerSettings()
    injected_pipeline = pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bixolon-inference",
        )
        app.state.inference_executor = inference_executor
        managed_pipeline: DecisionPipeline | None = None
        managed_detector = None
        try:
            if injected_pipeline is None:
                metadata_payload = json.loads(
                    (worker_settings.package_dir / "metadata.json").read_text(encoding="utf-8")
                )
                if metadata_payload.get("schema_version") == "2.0":
                    if worker_settings.catalog_dir is None:
                        raise ValueError("2.0 Worker requires a Store Catalog directory")
                    runtime_package = load_runtime_package_v2(worker_settings.package_dir)
                    catalog = load_store_catalog_package(
                        worker_settings.catalog_dir,
                        signing_key=(
                            None
                            if worker_settings.catalog_signing_key is None
                            else worker_settings.catalog_signing_key.get_secret_value().encode()
                        ),
                        expected_store_id=worker_settings.catalog_store_id,
                        expected_key_id=worker_settings.catalog_key_id,
                    )
                    provider = select_provider(worker_settings.provider)
                    detector = build_detector_v2(
                        runtime_package,
                        provider,
                        worker_settings.cuda_dll_dir,
                    )
                    managed_detector = detector
                    embedder = OnnxEmbedder(
                        runtime_package,
                        provider,
                        worker_settings.cuda_dll_dir,
                    )
                    classifier = OnnxCatalogClassifier(runtime_package, catalog, embedder)
                    detector.warmup()
                    embedder.warmup()
                    managed_pipeline = DecisionPipeline(
                        detector,
                        classifier,
                        classifier.metadata,
                        runtime_package.metadata.quality,
                        worker_version=runtime_package.metadata.worker_version,
                        embedder_version=runtime_package.metadata.embedder.version,
                        detector_policy_version=runtime_package.metadata.detector_policy_version,
                        classifier_policy_version=runtime_package.metadata.classifier_policy.version,
                        catalog_version=catalog.metadata.catalog_version,
                    )
                    app.state.pipeline = managed_pipeline
                    app.state.jpeg_draft_size = runtime_package.metadata.input.jpeg_draft_size
                else:
                    model_package = load_model_package(worker_settings.package_dir)
                    detector, classifier, provider = build_onnx_adapters(
                        model_package,
                        worker_settings.provider,
                        cuda_dll_dir=worker_settings.cuda_dll_dir,
                    )
                    managed_detector = detector
                    managed_pipeline = DecisionPipeline(
                        detector,
                        classifier,
                        model_package.metadata.classifier,
                        model_package.metadata.quality,
                        model_package.metadata.count_verifier,
                        worker_version=model_package.metadata.package_version,
                    )
                    app.state.pipeline = managed_pipeline
                    app.state.jpeg_draft_size = model_package.metadata.input.jpeg_draft_size
                app.state.provider = provider
            else:
                app.state.pipeline = injected_pipeline
                app.state.provider = "injected"
                app.state.jpeg_draft_size = worker_settings.jpeg_draft_size
            app.state.worker_version = app.state.pipeline.worker_version
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await asyncio.to_thread(
                inference_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            if managed_pipeline is not None:
                managed_pipeline.close()
            elif managed_detector is not None:
                close = getattr(managed_detector, "close", None)
                if callable(close):
                    close()

    app = FastAPI(title="Bixolon Image Decision Worker", version=__version__, lifespan=lifespan)
    app.state.ready = False
    app.state.semaphore = asyncio.Semaphore(1)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        request.state.started = time.perf_counter()
        return await call_next(request)

    @app.exception_handler(ScannerError)
    async def scanner_error_handler(request: Request, exc: ScannerError):
        elapsed = (time.perf_counter() - request.state.started) * 1000.0
        LOGGER.warning(
            "request_rejected",
            extra={"request_id": _request_id(request), "reason_code": exc.reason_code},
        )
        return _error_response(
            _request_id(request),
            exc.reason_code,
            elapsed,
            exc.http_status,
            worker_version=getattr(app.state, "worker_version", __version__),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        del exc
        elapsed = (time.perf_counter() - request.state.started) * 1000.0
        error = MissingImageError()
        return _error_response(
            _request_id(request),
            error.reason_code,
            elapsed,
            error.http_status,
            worker_version=getattr(app.state, "worker_version", __version__),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        elapsed = (time.perf_counter() - request.state.started) * 1000.0
        LOGGER.exception(
            "request_failed",
            extra={"request_id": _request_id(request), "exception_type": type(exc).__name__},
        )
        return _error_response(
            _request_id(request),
            "WORKER_ERROR",
            elapsed,
            500,
            worker_version=getattr(app.state, "worker_version", __version__),
        )

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        if not app.state.ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        versions = app.state.pipeline.versions
        payload = {
            "status": "ready",
            "provider": app.state.provider,
            "worker_version": app.state.worker_version,
            "detector_version": versions.detector,
            "classifier_version": versions.classifier,
        }
        optional_versions = {
            "embedder_version": app.state.pipeline.embedder_version,
            "detector_policy_version": app.state.pipeline.detector_policy_version,
            "classifier_policy_version": app.state.pipeline.classifier_policy_version,
            "catalog_version": app.state.pipeline.catalog_version,
        }
        payload.update(
            {key: value for key, value in optional_versions.items() if value is not None}
        )
        return payload

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan(request: Request, image: UploadFile = File(...)):
        data = await image.read(worker_settings.max_upload_bytes + 1)
        decode_started = time.perf_counter()
        decoded = decode_image(
            data,
            max_bytes=worker_settings.max_upload_bytes,
            max_pixels=worker_settings.max_image_pixels,
            jpeg_draft_size=app.state.jpeg_draft_size,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        try:
            async with asyncio.timeout(worker_settings.request_timeout_seconds):
                await app.state.semaphore.acquire()
                try:
                    loop = asyncio.get_running_loop()
                    inference = loop.run_in_executor(
                        app.state.inference_executor,
                        app.state.pipeline.scan,
                        decoded,
                        _request_id(request),
                    )
                except BaseException:
                    app.state.semaphore.release()
                    raise
                inference.add_done_callback(lambda _: app.state.semaphore.release())
                response = await asyncio.shield(inference)
                total_ms = (time.perf_counter() - request.state.started) * 1000.0
                completed = response.model_copy(update={"processing_time_ms": total_ms})
                segment_status_counts = {
                    status: sum(item.status.value == status for item in completed.segmentations)
                    for status in ("APPROVED", "UNKNOWN", "SEGMENT_RECAPTURE")
                }
                LOGGER.info(
                    "scan_request_complete",
                    extra={
                        "request_id": completed.request_id,
                        "status": completed.status.value,
                        "reason_codes": completed.reason_codes,
                        "segmentation_count": len(completed.segmentations),
                        "approved_count": segment_status_counts["APPROVED"],
                        "unknown_count": segment_status_counts["UNKNOWN"],
                        "segment_recapture_count": segment_status_counts["SEGMENT_RECAPTURE"],
                        "decode_ms": round(decode_ms, 3),
                        "processing_time_ms": round(total_ms, 3),
                        "worker_version": completed.worker_version,
                        "detector_version": completed.detector_version,
                        "classifier_version": completed.classifier_version,
                        "embedder_version": completed.embedder_version,
                        "detector_policy_version": completed.detector_policy_version,
                        "classifier_policy_version": completed.classifier_policy_version,
                        "catalog_version": completed.catalog_version,
                    },
                )
                return completed
        except TimeoutError as exc:
            raise ModelExecutionError from exc

    return app
