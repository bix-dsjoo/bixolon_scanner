from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.onnx import OrtRunner
from .classifier_geometry_mask import apply_background_mask, neighbor_ownership_mask
from .classifier_rate_policy import build_rate_policy, unanimous_top1_confirmation


def final_policy_decisions(
    base_logits: tuple[np.ndarray, np.ndarray],
    confirmation_logits: list[np.ndarray],
    *,
    base_view_names: tuple[str, str],
    first_view_weight: float,
    ranking_tie_break_bias_span: float,
    approval_thresholds: np.ndarray,
    top3_safety_threshold: float,
) -> dict[str, np.ndarray]:
    policy = build_rate_policy(
        base_logits[0],
        base_logits[1],
        left_name=base_view_names[0],
        right_name=base_view_names[1],
        first_view_weight=first_view_weight,
        ranking_tie_break_bias_span=ranking_tie_break_bias_span,
    )
    approved = policy.approval_score >= approval_thresholds[policy.predictions]
    unknown = (~approved) & (policy.top3_safety_score >= top3_safety_threshold)
    if confirmation_logits:
        confirmation_predictions = np.stack(
            [np.argmax(values, axis=1) for values in confirmation_logits],
            axis=1,
        )
        confirmed = unanimous_top1_confirmation(
            policy.predictions,
            confirmation_predictions,
        )
        withheld = approved & (~confirmed)
        approved[withheld] = False
        unknown[withheld] = True
    else:
        confirmed = np.ones(len(policy.predictions), dtype=bool)
    return {
        "predictions": policy.predictions,
        "top3": policy.top3,
        "approval_scores": policy.approval_score,
        "top3_safety_scores": policy.top3_safety_score,
        "approved": approved,
        "unknown": unknown,
        "confirmation": confirmed,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _masks(args: argparse.Namespace, rows: list[dict[str, Any]], output_size: int) -> np.ndarray:
    predictions = {int(row["image_id"]): row for row in _load_rows(args.predictions)}
    manifest = {int(row["image_id"]): row for row in _load_rows(args.manifest)}
    return np.stack(
        [
            neighbor_ownership_mask(
                image_width=int(manifest[int(row["image_id"])]["width"]),
                image_height=int(manifest[int(row["image_id"])]["height"]),
                boxes=predictions[int(row["image_id"])]["boxes_xyxy"],
                target_index=int(row["detection_index"]),
                output_size=output_size,
                margin_ratio=args.margin_ratio,
                distance_bias=0.0,
                shared_scale=False,
            )
            for row in rows
        ]
    )


def _provider_logits(
    args: argparse.Namespace,
    *,
    provider: str,
    tensors: np.ndarray,
    masks: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    runners = [
        OrtRunner(
            path,
            provider,
            args.cuda_dll_dir if provider == "cuda" else None,
        )
        for path in args.models
    ]
    outputs: list[list[np.ndarray]] = [[] for _ in runners]
    started = time.perf_counter()
    for start in range(0, len(tensors), args.batch_size):
        batch = np.array(tensors[start : start + args.batch_size], dtype=np.float32, copy=True)
        masked = apply_background_mask(batch, masks[start : start + len(batch)])
        for index, runner in enumerate(runners):
            (logits,) = runner.run(["logits"], "pixel_values", masked)
            outputs[index].append(np.asarray(logits, dtype=np.float32))
    elapsed = time.perf_counter() - started
    return [np.concatenate(parts) for parts in outputs], elapsed


def _model_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    actual_top1 = np.argmax(actual, axis=1)
    expected_top1 = np.argmax(expected, axis=1)
    actual_top3 = np.argsort(-actual, axis=1, kind="stable")[:, :3]
    expected_top3 = np.argsort(-expected, axis=1, kind="stable")[:, :3]
    return {
        "maximum_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "top1": _row_parity(actual_top1, expected_top1),
        "top3": _row_parity(actual_top3, expected_top3),
    }


def _row_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    if actual_values.shape != expected_values.shape:
        raise ValueError(
            f"parity arrays differ in shape: {actual_values.shape} != {expected_values.shape}"
        )
    differing = actual_values != expected_values
    if differing.ndim > 1:
        differing = np.any(differing, axis=tuple(range(1, differing.ndim)))
    indices = np.flatnonzero(differing)
    return {
        "equal": len(indices) == 0,
        "mismatch_count": int(len(indices)),
        "mismatch_indices": indices[:50].tolist(),
        "mismatch_indices_truncated": len(indices) > 50,
    }


def _decision_parity(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    expected_names: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    expected_names = expected_names or {}
    return {
        key: _row_parity(actual[key], expected[expected_names.get(key, key)])
        for key in ("predictions", "top3", "approved", "unknown", "confirmation")
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.models) != len(args.reference_logits):
        raise ValueError("models and reference logits must have the same length")
    if len(args.models) < 3:
        raise ValueError("confirmed classifier requires two base and confirmation models")
    policy_report = json.loads(args.policy_report.read_text(encoding="utf-8"))
    selected = policy_report["selected"]
    final = selected["full_development_calibration_for_package"]
    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = _load_rows(args.evaluation_records)
    if len(tensors) != len(rows):
        raise ValueError("evaluation tensors and records are not aligned")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    masks = _masks(args, rows, tensors.shape[-1])
    expected_logits = []
    for path in args.reference_logits:
        payload = np.load(path)
        if not np.array_equal(payload["targets"], targets):
            raise ValueError(f"reference logits targets differ: {path}")
        expected_logits.append(payload[args.reference_logit_key].astype(np.float32))
    reference_decisions = np.load(args.reference_decisions)
    providers: dict[str, Any] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for provider in args.providers:
        logits, elapsed = _provider_logits(
            args,
            provider=provider,
            tensors=tensors,
            masks=masks,
        )
        decisions = final_policy_decisions(
            (logits[0], logits[1]),
            logits[2:],
            base_view_names=tuple(args.base_view_names),
            first_view_weight=float(selected["first_view_weight"]),
            ranking_tie_break_bias_span=float(
                policy_report["selection"]["ranking_tie_break_bias_span"]
            ),
            approval_thresholds=np.asarray(final["approval_thresholds"], dtype=np.float32),
            top3_safety_threshold=float(final["top3_safety_threshold"]),
        )
        reference_state = {
            name: reference_decisions[name]
            for name in (
                "predictions",
                "top3",
                "final_approved",
                "final_unknown",
                "confirmation",
            )
        }
        providers[provider] = {
            "inference_sample_count": len(targets),
            "inference_elapsed_seconds": elapsed,
            "inference_ms_per_crop_all_models": elapsed * 1000.0 / len(targets),
            "models": [
                _model_parity(actual, expected) for actual, expected in zip(logits, expected_logits)
            ],
            "state_parity_to_pytorch": _decision_parity(
                decisions,
                reference_state,
                expected_names={
                    "approved": "final_approved",
                    "unknown": "final_unknown",
                },
            ),
        }
        arrays[provider] = decisions
        if args.arrays_output_dir is not None:
            args.arrays_output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.arrays_output_dir / f"{provider}-classifier-parity-arrays.npz",
                targets=targets,
                model_logits=np.stack(logits),
                **decisions,
            )
    if {"cpu", "cuda"}.issubset(arrays):
        providers["cpu_cuda_state_parity"] = _decision_parity(
            arrays["cpu"],
            arrays["cuda"],
        )
    state_parity_met = all(
        detail["equal"]
        for provider in args.providers
        for detail in providers[provider]["state_parity_to_pytorch"].values()
    )
    if "cpu_cuda_state_parity" in providers:
        state_parity_met = state_parity_met and all(
            detail["equal"] for detail in providers["cpu_cuda_state_parity"].values()
        )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_confirmed_classifier_onnx_parity",
        "sample_count": len(targets),
        "model_count": len(args.models),
        "providers": providers,
        "state_parity_met": state_parity_met,
        "promotion_ready": False,
        "promotion_blocker": (
            "full Worker latency and independent locked test pending"
            if state_parity_met
            else "classifier PyTorch/ONNX provider state parity failed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the confirmed classifier across PyTorch, CPU ONNX, and CUDA ONNX"
    )
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-logits", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-logit-key", default="normalized_bias0.000")
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--reference-decisions", type=Path, required=True)
    parser.add_argument("--base-view-names", nargs=2, default=("clutter_v2", "clutter_finetune_v2"))
    parser.add_argument("--providers", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda"))
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--margin-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--arrays-output-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
