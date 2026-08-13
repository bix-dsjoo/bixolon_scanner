from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..inference import OrtRunner
from ..package import load_model_package, sha256_file
from .fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
    wrap_inference_classifier,
)
from .models import require_torch
from .ten_shot_candidates import verify_experiment_lock


def _rank(logits: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(logits), axis=1, kind="stable")[:, :3]


def _confidence(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1)


def strict_classifier_parity(
    *,
    pytorch_logits: np.ndarray,
    cpu_logits: np.ndarray,
    cuda_logits: np.ndarray,
    temperature: float,
    approval_threshold: float,
    pytorch_onnx_tolerance: float,
    cross_provider_tolerance: float,
) -> dict[str, Any]:
    """Require numerical tolerance plus exact state and ordered Top-3 parity."""
    arrays = [np.asarray(value) for value in (pytorch_logits, cpu_logits, cuda_logits)]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("parity logits must have the same [batch, classes] shape")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("parity logits contain non-finite values")
    ranks = [_rank(value) for value in arrays]
    states = [_confidence(value, temperature) >= approval_threshold for value in arrays]
    checks = {
        "pytorch_cpu_tolerance": float(np.max(np.abs(arrays[0] - arrays[1])))
        <= pytorch_onnx_tolerance,
        "pytorch_cuda_tolerance": float(np.max(np.abs(arrays[0] - arrays[2])))
        <= pytorch_onnx_tolerance,
        "cpu_cuda_tolerance": float(np.max(np.abs(arrays[1] - arrays[2])))
        <= cross_provider_tolerance,
        "top1_equal": all(np.array_equal(ranks[0][:, :1], value[:, :1]) for value in ranks[1:]),
        "top3_set_and_order_equal": all(np.array_equal(ranks[0], value) for value in ranks[1:]),
        "final_state_equal": all(np.array_equal(states[0], value) for value in states[1:]),
    }
    mismatch_counts = {
        "top1": int(
            sum(np.any(ranks[0][:, :1] != value[:, :1], axis=1).sum() for value in ranks[1:])
        ),
        "top3_set_or_order": int(
            sum(np.any(ranks[0] != value, axis=1).sum() for value in ranks[1:])
        ),
        "final_state": int(sum(np.count_nonzero(states[0] != value) for value in states[1:])),
        "cpu_cuda_top3_set_or_order": int(np.any(ranks[1] != ranks[2], axis=1).sum()),
        "cpu_cuda_final_state": int(np.count_nonzero(states[1] != states[2])),
    }
    return {
        "schema_version": "1.0",
        "passes": all(checks.values()),
        "checks": checks,
        "mismatch_counts": mismatch_counts,
        "sample_count": arrays[0].shape[0],
        "maximum_absolute_difference": {
            "pytorch_cpu": float(np.max(np.abs(arrays[0] - arrays[1]))),
            "pytorch_cuda": float(np.max(np.abs(arrays[0] - arrays[2]))),
            "cpu_cuda": float(np.max(np.abs(arrays[1] - arrays[2]))),
        },
    }


def _pytorch_logits(model, tensors: np.ndarray, *, device, batch_size: int) -> np.ndarray:
    torch = require_torch()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(
                    tensors[start : start + batch_size],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(device)
            parts.append(model(batch).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _onnx_logits(
    model_path: Path,
    tensors: np.ndarray,
    *,
    provider: str,
    cuda_dll_dir: Path | None,
    batch_size: int,
) -> np.ndarray:
    runner = OrtRunner(model_path, provider, cuda_dll_dir)
    parts = []
    for start in range(0, len(tensors), batch_size):
        batch = np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
        parts.append(runner.run(["logits"], "pixel_values", batch)[0])
    return np.concatenate(parts).astype(np.float32)


def run_locked_parity(args: argparse.Namespace) -> dict[str, Any]:
    verify_experiment_lock(
        args.pretest_lock,
        config=args.config,
        manifest=args.manifest,
        manifest_metadata=args.manifest_metadata,
        checkpoint=args.checkpoint,
        calibration=args.calibration,
    )
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    package = load_model_package(args.package_dir)
    spec = adapter_spec_from_dict(checkpoint["adapter_spec"])
    model = build_ten_shot_classifier(
        backbone_kind=checkpoint["backbone_kind"],
        weights_path=args.backbone_weights,
        hub_repository=(
            f"facebookresearch/dinov3:{checkpoint['source_revision']}"
            if checkpoint.get("source_revision")
            else "facebookresearch/dinov3"
        ),
        spec=spec,
    )
    if checkpoint["architecture"] == "ten_shot_residual_cosine_challenger":
        model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    elif checkpoint["architecture"] == "ten_shot_residual_cosine":
        model.classifier.load_state_dict(compatible_proxy_state_dict(checkpoint["head_state_dict"]))
    else:
        raise ValueError("unsupported ten-shot checkpoint architecture")
    crop_value = config.get("inference", {}).get("center_crop_scale")
    crop_scale = None if crop_value is None else float(crop_value)
    inference = config.get("inference", {})
    model = wrap_inference_classifier(
        model,
        input_size=int(checkpoint["image_size"]),
        crop_scale=crop_scale,
        num_classes=int(checkpoint["num_classes"]),
        logit_quantum=(
            None if inference.get("logit_quantum") is None else float(inference["logit_quantum"])
        ),
        logit_phase=float(inference.get("logit_phase", 0.0)),
        tie_break_bias_span=float(inference.get("tie_break_bias_span", 0.0)),
        logit_divisor=float(inference.get("logit_divisor", 1.0)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    pytorch = _pytorch_logits(model, tensors, device=device, batch_size=args.batch_size)
    cpu = _onnx_logits(
        package.classifier_path,
        tensors,
        provider="cpu",
        cuda_dll_dir=None,
        batch_size=args.batch_size,
    )
    cuda = _onnx_logits(
        package.classifier_path,
        tensors,
        provider="cuda",
        cuda_dll_dir=args.cuda_dll_dir,
        batch_size=args.batch_size,
    )
    report = strict_classifier_parity(
        pytorch_logits=pytorch,
        cpu_logits=cpu,
        cuda_logits=cuda,
        temperature=float(calibration["temperature"]),
        approval_threshold=float(calibration["approval_threshold"]),
        pytorch_onnx_tolerance=float(config["evaluation"]["classifier_tolerance"]),
        cross_provider_tolerance=float(config["evaluation"]["cross_provider_tolerance"]),
    )
    report.update(
        {
            "pytorch_device": str(device),
            "onnx_providers": ["CPUExecutionProvider", "CUDAExecutionProvider"],
            "classifier_version": package.metadata.classifier.version,
            "classifier_checkpoint_sha256": sha256_file(args.checkpoint),
            "package_artifact_sha256": {
                "metadata.json": sha256_file(args.package_dir / "metadata.json"),
                package.metadata.classifier.filename: sha256_file(package.classifier_path),
            },
            "inference_center_crop_scale": crop_scale,
            "pretest_lock": str(args.pretest_lock),
        }
    )
    logits_dir = args.output.parent / "parity_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / "pytorch.npy", pytorch)
    np.save(logits_dir / "cpu_onnx.npy", cpu)
    np.save(logits_dir / "cuda_onnx.npy", cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run locked PyTorch/CPU ONNX/CUDA ONNX 10-shot parity"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--pretest-lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_locked_parity(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
