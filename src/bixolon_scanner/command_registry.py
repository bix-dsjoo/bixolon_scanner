"""Command registry split by the support level exposed to users."""

from __future__ import annotations

CommandTarget = tuple[str, str]
CommandPath = tuple[str, ...]

ACTIVE_COMMANDS: dict[CommandPath, CommandTarget] = {
    ("bundle", "verify"): (
        "bixolon_scanner.operations.version_bundle",
        "verify_main",
    ),
}

# Kept callable for existing operator scripts and dedicated console aliases. These
# are not part of the single-version lifecycle CLI advertised by normal help.
COMPATIBILITY_COMMANDS: dict[CommandPath, CommandTarget] = {
    ("worker",): ("bixolon_scanner.worker.cli", "serve"),
    ("operations", "ingest-logs"): (
        "bixolon_scanner.operations.operational_logs",
        "main",
    ),
    ("operations", "export-review"): (
        "bixolon_scanner.operations.scan_log_review_export",
        "main",
    ),
    ("operations", "metrics"): (
        "bixolon_scanner.operations.worker_metrics",
        "main",
    ),
    ("catalog", "activate"): (
        "bixolon_scanner.operations.catalog_activation",
        "main",
    ),
}

DIAGNOSTIC_COMMANDS: dict[CommandPath, CommandTarget] = {
    ("data", "manifest"): ("bixolon_scanner.training.manifest", "main"),
    ("data", "ten-shot-manifest"): (
        "bixolon_scanner.training.ten_shot_manifest",
        "main",
    ),
    ("train", "classifier"): ("bixolon_scanner.training.train_classifier", "main"),
    ("train", "detector"): ("bixolon_scanner.training.train_detector", "main"),
    ("train", "verify-pipeline"): (
        "bixolon_scanner.training.pipeline_contract",
        "main",
    ),
    ("train", "lock-detector-checkpoint"): (
        "bixolon_scanner.training.checkpoint_lock",
        "main",
    ),
    ("train", "write-run-evidence"): (
        "bixolon_scanner.training.run_evidence",
        "main",
    ),
    ("evaluate", "classifier"): ("bixolon_scanner.evaluation.classifier", "main"),
    ("evaluate", "detector"): ("bixolon_scanner.evaluation.detector", "main"),
    ("evaluate", "aggregate-detector"): (
        "bixolon_scanner.evaluation.aggregate_detector",
        "main",
    ),
    ("evaluate", "worker"): ("bixolon_scanner.evaluation.worker", "main"),
    ("evaluate", "difficulty"): ("bixolon_scanner.evaluation.difficulty", "main"),
    ("evaluate", "operational"): (
        "bixolon_scanner.evaluation.operational",
        "main",
    ),
    ("evaluate", "parity"): ("bixolon_scanner.evaluation.parity", "main"),
    ("evaluate", "benchmark"): ("bixolon_scanner.evaluation.benchmark", "main"),
    ("evaluate", "compare-difficulty"): (
        "bixolon_scanner.evaluation.compare_difficulty",
        "main",
    ),
    ("model", "export"): ("bixolon_scanner.training.export", "main"),
    ("model", "export-embedder"): (
        "bixolon_scanner.training.export_embedder",
        "main",
    ),
    ("model", "export-dinov2-embedder"): (
        "bixolon_scanner.training.export_dinov2_embedder",
        "main",
    ),
    ("experiment", "bread-10shot"): (
        "bixolon_scanner.experiments.bread.ten_shot",
        "main",
    ),
    ("experiment", "bread-data-scale"): (
        "bixolon_scanner.experiments.bread.data_scale",
        "main",
    ),
    ("experiment", "detector-target"): (
        "bixolon_scanner.experiments.detector.target",
        "main",
    ),
    ("experiment", "rpc-data-scale"): (
        "bixolon_scanner.experiments.rpc200.data_scale",
        "main",
    ),
    ("experiment", "rpc-operational"): (
        "bixolon_scanner.experiments.rpc200.operational",
        "main",
    ),
    ("tools", "cache-classifier"): (
        "bixolon_scanner.training.cache_classifier",
        "main",
    ),
    ("tools", "cache-detector"): (
        "bixolon_scanner.training.cache_detector",
        "main",
    ),
    ("tools", "predict-classifier"): (
        "bixolon_scanner.training.predict_classifier",
        "main",
    ),
    ("tools", "render-detections"): (
        "bixolon_scanner.training.render_detection_overlays",
        "main",
    ),
}

# These remain callable for reproducibility and old scripts, but are intentionally
# absent from the normal help because they are not current product lifecycle APIs.
LEGACY_COMMANDS: dict[CommandPath, CommandTarget] = {
    ("evaluate", "bread-1.1-runtime"): (
        "bixolon_scanner.evaluation.bread_runtime_gate",
        "main",
    ),
    ("evaluate", "bread-1.1-runtime-parity"): (
        "bixolon_scanner.evaluation.bread_runtime_parity",
        "main",
    ),
    ("evaluate", "scanner-2.0"): (
        "bixolon_scanner.evaluation.scanner_v2",
        "main",
    ),
    ("evaluate", "scanner-2.0-parity"): (
        "bixolon_scanner.evaluation.scanner_v2_parity",
        "main",
    ),
    ("evaluate", "scanner-2.0-embedder-parity"): (
        "bixolon_scanner.evaluation.embedder_v2_parity",
        "main",
    ),
    ("evaluate", "scanner-2.0-packaged-worker-smoke"): (
        "bixolon_scanner.evaluation.scanner_v2_packaged_worker_smoke",
        "main",
    ),
    ("experiment", "bread-1.1-development-identity"): (
        "bixolon_scanner.experiments.bread.development_identity_manifest",
        "main",
    ),
    ("experiment", "scanner-2.0-development-identity"): (
        "bixolon_scanner.evaluation.scanner_v2_development_identity",
        "main",
    ),
    ("experiment", "bread-classifier-200-only"): (
        "bixolon_scanner.experiments.bread.classifier_200_only",
        "main",
    ),
    ("experiment", "bread-catalog-metric"): (
        "bixolon_scanner.experiments.bread.catalog_metric",
        "main",
    ),
    ("model", "bread-2.0-runtime"): (
        "bixolon_scanner.experiments.bread.build_scanner_v2",
        "main",
    ),
    ("model", "bread-1.1-candidate-package"): (
        "bixolon_scanner.experiments.bread.runtime_candidate_package",
        "main",
    ),
}


def _merge_registries() -> dict[CommandPath, CommandTarget]:
    registries = (
        ACTIVE_COMMANDS,
        COMPATIBILITY_COMMANDS,
        DIAGNOSTIC_COMMANDS,
        LEGACY_COMMANDS,
    )
    merged: dict[CommandPath, CommandTarget] = {}
    for registry in registries:
        overlap = merged.keys() & registry.keys()
        if overlap:
            formatted = ", ".join(" ".join(path) for path in sorted(overlap))
            raise RuntimeError(f"command registered more than once: {formatted}")
        merged.update(registry)
    return merged


COMMANDS = _merge_registries()

__all__ = [
    "ACTIVE_COMMANDS",
    "COMMANDS",
    "COMPATIBILITY_COMMANDS",
    "DIAGNOSTIC_COMMANDS",
    "LEGACY_COMMANDS",
    "CommandPath",
    "CommandTarget",
]
