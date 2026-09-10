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

    def decode_and_scan(data: bytes, request_id: str, deadline: float):
        decode_started = time.perf_counter()
        decoded = decode_image(
            data,
            max_bytes=worker_settings.max_upload_bytes,
            max_pixels=worker_settings.max_image_pixels,
            jpeg_draft_size=app.state.jpeg_draft_size,
        )
        try:
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if time.perf_counter() >= deadline:
                raise ModelExecutionError
            return app.state.runtime.scan(decoded, request_id), decode_ms
        finally:
            decoded.close()

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
            app.state.runtime = runtime
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
    app.state.inference_deadline = None

    def inference_finished(future):
        app.state.inference_deadline = None
        app.state.semaphore.release()
        # Retrieve failures even when the HTTP request has already timed out.
        if not future.cancelled():
            future.exception()

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
        LOGGER.error(
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
        overdue = (
            app.state.inference_deadline is not None
            and time.perf_counter() >= app.state.inference_deadline
        )
        runtime = getattr(app.state, "runtime", None)
        if (
            not app.state.ready
            or overdue
            or (runtime is not None and (runtime.recovering or runtime.failed))
        ):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        active_pipeline = runtime.pipeline
        versions = active_pipeline.versions
        payload = {
            "status": "ready",
            "provider": runtime.provider,
            "worker_version": app.state.worker_version,
            "detector_version": versions.detector,
            "classifier_version": versions.classifier,
        }
        optional_versions = {
            "embedder_version": active_pipeline.embedder_version,
            "detector_policy_version": active_pipeline.detector_policy_version,
            "classifier_policy_version": active_pipeline.classifier_policy_version,
            "catalog_version": active_pipeline.catalog_version,
        }
        payload.update(
            {key: value for key, value in optional_versions.items() if value is not None}
        )
        return payload

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan(request: Request, image: UploadFile = File(...)):
        deadline = request.state.started + worker_settings.request_timeout_seconds
        queue_started = time.perf_counter()
        try:
            remaining = max(0.0, deadline - time.perf_counter())
            async with asyncio.timeout(remaining):
                await app.state.semaphore.acquire()
                queue_wait_ms = (time.perf_counter() - queue_started) * 1000.0
                try:
                    data = await image.read(worker_settings.max_upload_bytes + 1)
                    loop = asyncio.get_running_loop()
                    app.state.inference_deadline = deadline
                    inference = loop.run_in_executor(
                        app.state.inference_executor,
                        decode_and_scan,
                        data,
                        _request_id(request),
                        deadline,
                    )
                except BaseException:
                    app.state.inference_deadline = None
                    app.state.semaphore.release()
                    raise
                inference.add_done_callback(inference_finished)
                response, decode_ms = await asyncio.shield(inference)
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
                        "queue_wait_ms": round(queue_wait_ms, 3),
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
