from __future__ import annotations

import logging
from dataclasses import dataclass

from ..configuration import load_json_config
from ..contracts import load_runtime_package_v2, load_store_catalog_package
from ..contracts.errors import ModelExecutionError, ProviderInitializationError
from ..contracts.model_package import load_model_package
from ..pipeline import DecisionPipeline
from ..runtime.catalog import build_catalog_classifier
from ..runtime.detector_v2 import build_detector_v2, replace_count_verifier_v2
from ..runtime.onnx import build_onnx_adapters, select_provider
from .settings import WorkerSettings

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkerRuntime:
    pipeline: DecisionPipeline
    provider: str
    jpeg_draft_size: int
    owned: bool = True

    def close(self) -> None:
        if self.owned:
            self.pipeline.close()


def _close_resource(resource: object | None) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _build_and_warm_classifier(runtime_package, catalog, selected_provider, settings):
    classifier, embedder = build_catalog_classifier(
        runtime_package,
        catalog,
        selected_provider,
        settings.cuda_dll_dir,
        cpu_intra_op_threads=settings.cpu_embedder_intra_op_threads,
        openvino_cache_dir=settings.openvino_cache_dir,
    )
    try:
        embedder.warmup()
        classifier_warmup = getattr(classifier, "warmup", None)
        if callable(classifier_warmup):
            classifier_warmup()
    except Exception:
        if callable(getattr(classifier, "close", None)):
            _close_resource(classifier)
        else:
            _close_resource(embedder)
        raise
    return classifier, embedder


def _build_v2_runtime(settings: WorkerSettings) -> WorkerRuntime:
    if settings.catalog_dir is None:
        raise ValueError("2.0 Worker requires a Store Catalog directory")
    runtime_package = load_runtime_package_v2(settings.package_dir)
    catalog = load_store_catalog_package(
        settings.catalog_dir,
        signing_key=(
            None
            if settings.catalog_signing_key is None
            else settings.catalog_signing_key.get_secret_value().encode()
        ),
        expected_store_id=settings.catalog_store_id,
        expected_key_id=settings.catalog_key_id,
    )
    provider = select_provider(settings.provider)
    embedder_provider = (
        provider
        if settings.embedder_provider == "same"
        else select_provider(settings.embedder_provider)
    )
    detector = None
    classifier = None
    try:
        detector = build_detector_v2(
            runtime_package,
            provider,
            settings.cuda_dll_dir,
            cpu_detector_workers=settings.cpu_detector_workers,
            cpu_intra_op_threads=settings.cpu_detector_intra_op_threads,
            openvino_cache_dir=settings.openvino_cache_dir,
        )
        detector.warmup()
        try:
            classifier, _embedder = _build_and_warm_classifier(
                runtime_package,
                catalog,
                embedder_provider,
                settings,
            )
            if (
                embedder_provider == "openvino_gpu"
                and runtime_package.metadata.count_verifier is not None
            ):
                replace_count_verifier_v2(
                    detector,
                    runtime_package,
                    embedder_provider,
                    settings.cuda_dll_dir,
                    cpu_intra_op_threads=settings.cpu_detector_intra_op_threads,
                    openvino_cache_dir=settings.openvino_cache_dir,
                    parallel_verification=True,
                )
        except (ProviderInitializationError, ModelExecutionError) as exc:
            fallback_enabled = (
                settings.embedder_fallback_provider == "same" and embedder_provider != provider
            )
            if not fallback_enabled:
                raise
            LOGGER.warning(
                "embedder_provider_fallback",
                extra={
                    "requested_provider": embedder_provider,
                    "fallback_provider": provider,
                    "exception_type": type(exc).__name__,
                },
            )
            embedder_provider = provider
            classifier, _embedder = _build_and_warm_classifier(
                runtime_package,
                catalog,
                embedder_provider,
                settings,
            )
        pipeline = DecisionPipeline(
            detector,
            classifier,
            classifier.metadata,
            runtime_package.metadata.quality,
            runtime_package.metadata.count_verifier,
            worker_version=runtime_package.metadata.worker_version,
            embedder_version=runtime_package.metadata.embedder.version,
            detector_policy_version=runtime_package.metadata.detector_policy_version,
            classifier_policy_version=runtime_package.metadata.classifier_policy.version,
            catalog_version=catalog.metadata.catalog_version,
        )
    except BaseException:
        _close_resource(classifier)
        _close_resource(detector)
        raise
    provider_label = (
        provider if provider == embedder_provider else f"{provider}+{embedder_provider}"
    )
    return WorkerRuntime(
        pipeline=pipeline,
        provider=provider_label,
        jpeg_draft_size=runtime_package.metadata.input.jpeg_draft_size,
    )


def _build_legacy_runtime(settings: WorkerSettings) -> WorkerRuntime:
    model_package = load_model_package(settings.package_dir)
    detector, classifier, provider = build_onnx_adapters(
        model_package,
        settings.provider,
        cuda_dll_dir=settings.cuda_dll_dir,
    )
    try:
        pipeline = DecisionPipeline(
            detector,
            classifier,
            model_package.metadata.classifier,
            model_package.metadata.quality,
            model_package.metadata.count_verifier,
            worker_version=model_package.metadata.package_version,
        )
    except BaseException:
        _close_resource(classifier)
        _close_resource(detector)
        raise
    return WorkerRuntime(
        pipeline=pipeline,
        provider=provider,
        jpeg_draft_size=model_package.metadata.input.jpeg_draft_size,
    )


def build_worker_runtime(
    settings: WorkerSettings,
    injected_pipeline: DecisionPipeline | None = None,
) -> WorkerRuntime:
    if injected_pipeline is not None:
        return WorkerRuntime(
            pipeline=injected_pipeline,
            provider="injected",
            jpeg_draft_size=settings.jpeg_draft_size,
            owned=False,
        )
    metadata = load_json_config(settings.package_dir / "metadata.json")
    if metadata.get("schema_version") == "2.0":
        return _build_v2_runtime(settings)
    return _build_legacy_runtime(settings)


__all__ = ["WorkerRuntime", "build_worker_runtime"]
