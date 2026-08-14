from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ....contracts.model_package import load_model_package
from ....runtime.onnx import OrtRunner


def normalized_scores(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-8)


def rank_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    order = np.argsort(-scores, axis=1, kind="stable")
    correct = order[:, 0] == targets
    return {
        "sample_count": len(targets),
        "top1_accuracy": float(correct.mean()),
        "top1_error_count": int((~correct).sum()),
        "top3_accuracy": float(np.any(order[:, :3] == targets[:, None], axis=1).mean()),
    }


def _targets(path: Path) -> np.ndarray:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return np.asarray([int(row["target"]) for row in records], dtype=np.int64)


def _legacy_logits(
    tensors: np.ndarray,
    *,
    package_dir: Path,
    provider: str,
    cuda_dll_dir: Path | None,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    package = load_model_package(package_dir)
    runner = OrtRunner(package.classifier_path, provider, cuda_dll_dir)
    metadata = package.metadata.classifier
    values: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, len(tensors), batch_size):
        (logits,) = runner.run(
            [metadata.logits_output],
            metadata.input_name,
            np.asarray(tensors[offset : offset + batch_size], dtype=np.float32),
        )
        values.append(np.asarray(logits, dtype=np.float32))
    return np.concatenate(values), time.perf_counter() - started


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    tensors = np.load(args.tensors, mmap_mode="r")
    candidate = np.load(args.candidate_logits)
    targets = _targets(args.records)
    if len(tensors) != len(candidate) or len(candidate) != len(targets):
        raise ValueError("ensemble inputs are not aligned")
    legacy, legacy_seconds = _legacy_logits(
        tensors,
        package_dir=args.package_dir,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
        batch_size=args.batch_size,
    )
    candidate_scores = normalized_scores(candidate)
    legacy_scores = normalized_scores(legacy)
    candidate_top1 = np.argmax(candidate_scores, axis=1)
    legacy_top1 = np.argmax(legacy_scores, axis=1)
    weights = np.linspace(0.0, 1.0, args.weight_steps + 1)
    sweep = []
    for candidate_weight in weights:
        scores = candidate_weight * candidate_scores + (1.0 - candidate_weight) * legacy_scores
        sweep.append(
            {
                "candidate_weight": float(candidate_weight),
                **rank_metrics(scores, targets),
            }
        )
    selected = max(
        sweep,
        key=lambda row: (row["top1_accuracy"], row["top3_accuracy"], -row["candidate_weight"]),
    )
    result = {
        "schema_version": "1.0",
        "evaluation": "classifier_ensemble_complementarity_probe",
        "promotion_status": "diagnostic_only",
        "sample_count": len(targets),
        "candidate": rank_metrics(candidate_scores, targets),
        "legacy": rank_metrics(legacy_scores, targets),
        "oracle_either_top1": {
            "accuracy": float(
                np.logical_or(candidate_top1 == targets, legacy_top1 == targets).mean()
            ),
            "error_count": int(
                np.logical_and(candidate_top1 != targets, legacy_top1 != targets).sum()
            ),
        },
        "selected_on_same_development_set": selected,
        "weight_sweep": sweep,
        "legacy_cuda_ms_per_crop": legacy_seconds * 1000.0 / len(targets),
        "limitations": [
            "The blend weight is selected on the same development set and is not promotion evidence.",
            "Candidate runtime, ONNX parity and end-to-end Worker latency are not measured here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe classifier ensemble complementarity")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-steps", type=int, default=20)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
