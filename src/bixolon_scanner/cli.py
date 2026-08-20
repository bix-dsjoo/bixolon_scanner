"""Unified CLI with compatibility access to the Worker command."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable

from .worker.cli import serve

CommandTarget = tuple[str, str]

COMMANDS: dict[tuple[str, ...], CommandTarget] = {
    ("worker",): ("bixolon_scanner.worker.cli", "serve"),
    ("data", "manifest"): ("bixolon_scanner.training.manifest", "main"),
    ("data", "ten-shot-manifest"): ("bixolon_scanner.training.ten_shot_manifest", "main"),
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
    ("evaluate", "operational"): ("bixolon_scanner.evaluation.operational", "main"),
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
    ("evaluate", "scanner-2.0-private-preflight"): (
        "bixolon_scanner.evaluation.scanner_v2_private_preflight",
        "main",
    ),
    ("evaluate", "scanner-2.0-private"): (
        "bixolon_scanner.evaluation.scanner_v2_private",
        "main",
    ),
    ("evaluate", "scanner-2.0-packaged-worker-smoke"): (
        "bixolon_scanner.evaluation.scanner_v2_packaged_worker_smoke",
        "main",
    ),
    ("evaluate", "bread-1.1-independent-preflight"): (
        "bixolon_scanner.experiments.bread.independent_preflight",
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
    ("model", "bread-2.0-runtime"): (
        "bixolon_scanner.experiments.bread.build_scanner_v2",
        "main",
    ),
    ("model", "promote"): ("bixolon_scanner.training.promote_package", "main"),
    ("model", "ten-shot-package"): ("bixolon_scanner.training.ten_shot_package", "main"),
    ("model", "ten-shot-finalize"): ("bixolon_scanner.training.ten_shot_finalize", "main"),
    ("model", "bread-1.1-candidate-package"): (
        "bixolon_scanner.experiments.bread.runtime_candidate_package",
        "main",
    ),
    ("experiment", "bread-10shot"): ("bixolon_scanner.experiments.bread.ten_shot", "main"),
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
    ("release", "verify"): (
        "bixolon_scanner.operations.release_composition",
        "main",
    ),
    ("release", "lock-scanner-2.0"): (
        "bixolon_scanner.operations.scanner_v2_release_lock",
        "main",
    ),
    ("release", "promote-scanner-2.0"): (
        "bixolon_scanner.operations.scanner_v2_promote",
        "main",
    ),
    ("release", "promote-scanner-2.0-owner-waiver"): (
        "bixolon_scanner.operations.scanner_v2_owner_waiver_promote",
        "main",
    ),
    ("tools", "cache-classifier"): (
        "bixolon_scanner.training.cache_classifier",
        "main",
    ),
    ("tools", "cache-detector"): ("bixolon_scanner.training.cache_detector", "main"),
    ("tools", "predict-classifier"): (
        "bixolon_scanner.training.predict_classifier",
        "main",
    ),
    ("tools", "render-detections"): (
        "bixolon_scanner.training.render_detection_overlays",
        "main",
    ),
}


def _help() -> str:
    lines = ["usage: bixolon <group> <command> [options]", "", "commands:"]
    lines.extend(f"  {' '.join(path)}" for path in sorted(COMMANDS))
    lines.append("")
    lines.append("Use `bixolon <group> <command> --help` for command-specific options.")
    return "\n".join(lines)


def _resolve(argv: list[str]) -> tuple[tuple[str, ...], list[str], CommandTarget] | None:
    for length in (2, 1):
        path = tuple(argv[:length])
        target = COMMANDS.get(path)
        if target is not None:
            return path, argv[length:], target
    return None


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        print(_help())
        return
    resolved = _resolve(arguments)
    if resolved is None:
        print(_help(), file=sys.stderr)
        raise SystemExit(2)
    path, remaining, (module_name, function_name) = resolved
    module = import_module(module_name)
    command: Callable[[], None] = getattr(module, function_name)
    original = sys.argv
    try:
        sys.argv = [f"bixolon {' '.join(path)}", *remaining]
        command()
    finally:
        sys.argv = original


__all__ = ["COMMANDS", "main", "serve"]
