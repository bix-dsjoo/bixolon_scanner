from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..contracts import ScanResponse, Status
from ..contracts.errors import MissingImageError, ModelExecutionError, ScannerError
from ..pipeline import DecisionPipeline
from ..runtime.imaging import decode_image
from .runtime_factory import WorkerRuntime, build_worker_runtime
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
        runtime: WorkerRuntime | None = None
        try:
            runtime = build_worker_runtime(worker_settings, injected_pipeline)
            app.state.pipeline = runtime.pipeline
            app.state.provider = runtime.provider
            app.state.jpeg_draft_size = runtime.jpeg_draft_size
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
            if runtime is not None:
                runtime.close()

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
