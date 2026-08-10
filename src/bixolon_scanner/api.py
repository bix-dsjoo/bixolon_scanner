from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import WorkerSettings
from .contracts import ModelVersions, ScanResponse, Status
from .errors import MissingImageError, ScannerError
from .imaging import decode_image
from .inference import build_onnx_adapters
from .package import load_model_package
from .pipeline import DecisionPipeline


LOGGER = logging.getLogger(__name__)


def _request_id(request: Request | None = None) -> str:
    if request is not None and hasattr(request.state, "request_id"):
        return request.state.request_id
    return uuid.uuid4().hex


def _error_response(request_id: str, reason_code: str, elapsed_ms: float, status_code: int) -> JSONResponse:
    body = ScanResponse(
        request_id=request_id,
        status=Status.ERROR,
        reason_codes=[reason_code],
        items=[],
        processing_time_ms=max(0.0, elapsed_ms),
        model_versions=ModelVersions(detector=None, classifier=None),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def create_app(
    *, settings: WorkerSettings | None = None, pipeline: DecisionPipeline | None = None
) -> FastAPI:
    worker_settings = settings or WorkerSettings()
    injected_pipeline = pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if injected_pipeline is None:
            model_package = load_model_package(worker_settings.package_dir)
            detector, classifier, provider = build_onnx_adapters(
                model_package,
                worker_settings.provider,
                cuda_dll_dir=worker_settings.cuda_dll_dir,
            )
            app.state.pipeline = DecisionPipeline(
                detector,
                classifier,
                model_package.metadata.classifier,
                model_package.metadata.quality,
                model_package.metadata.count_verifier,
            )
            app.state.provider = provider
            app.state.jpeg_draft_size = model_package.metadata.input.jpeg_draft_size
        else:
            app.state.pipeline = injected_pipeline
            app.state.provider = "injected"
            app.state.jpeg_draft_size = worker_settings.jpeg_draft_size
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="Bixolon Image Decision Worker", version="1.0.0", lifespan=lifespan)
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
        return _error_response(_request_id(request), exc.reason_code, elapsed, exc.http_status)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        del exc
        elapsed = (time.perf_counter() - request.state.started) * 1000.0
        error = MissingImageError()
        return _error_response(_request_id(request), error.reason_code, elapsed, error.http_status)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        elapsed = (time.perf_counter() - request.state.started) * 1000.0
        LOGGER.exception(
            "request_failed",
            extra={"request_id": _request_id(request), "exception_type": type(exc).__name__},
        )
        return _error_response(_request_id(request), "WORKER_ERROR", elapsed, 500)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        if not app.state.ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "provider": app.state.provider}

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
                async with app.state.semaphore:
                    response = await asyncio.to_thread(
                        app.state.pipeline.scan, decoded, _request_id(request)
                    )
                    total_ms = (time.perf_counter() - request.state.started) * 1000.0
                    completed = response.model_copy(
                        update={
                            "processing_time_ms": total_ms
                        }
                    )
                    LOGGER.info(
                        "scan_request_complete",
                        extra={
                            "request_id": completed.request_id,
                            "status": completed.status.value,
                            "reason_codes": completed.reason_codes,
                            "item_count": len(completed.items),
                            "decode_ms": round(decode_ms, 3),
                            "processing_time_ms": round(total_ms, 3),
                            "detector_version": completed.model_versions.detector,
                            "classifier_version": completed.model_versions.classifier,
                        },
                    )
                    return completed
        except TimeoutError as exc:
            from .errors import ModelExecutionError

            raise ModelExecutionError from exc

    return app
