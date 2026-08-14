from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ....training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ....training.models import require_torch
from ...bread.ten_shot import (
    _apply_inference_logit_policy,
    _model_development_logits,
)


def fusion_candidates(
    logits_by_scale: dict[float, np.ndarray], targets: np.ndarray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = []

    def record(name: str, logits: np.ndarray, scales: list[float], weights: list[float]) -> None:
        values = _apply_inference_logit_policy(logits, config)
        correct = int(np.count_nonzero(values.argmax(axis=1) == targets))
        candidates.append(
            {
                "name": name,
                "scales": scales,
                "weights": weights,
                "correct": correct,
                "sample_count": len(targets),
                "top1_accuracy": correct / len(targets),
            }
        )

    for scale, logits in logits_by_scale.items():
        record(f"single_{scale:.3f}", logits, [scale], [1.0])
    for left, right in combinations(logits_by_scale, 2):
        for weight in np.linspace(0.1, 0.9, 9):
            record(
                f"pair_{left:.3f}_{right:.3f}_{weight:.1f}",
                logits_by_scale[left] * weight + logits_by_scale[right] * (1.0 - weight),
                [left, right],
                [float(weight), float(1.0 - weight)],
            )
    for scales in combinations(logits_by_scale, 3):
        record(
            "triple_" + "_".join(f"{value:.3f}" for value in scales),
            sum(logits_by_scale[scale] for scale in scales) / 3.0,
            list(scales),
            [1.0 / 3.0] * 3,
        )
    return sorted(candidates, key=lambda row: (row["top1_accuracy"], row["name"]), reverse=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=checkpoint["backbone_kind"],
        weights_path=args.weights,
        hub_repository=config["training"]["hub_repository"],
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device).eval()
    tensors = np.load(args.prepared / "evaluation_tensors.npy", mmap_mode="r")
    rows = [
        json.loads(line)
        for line in (args.prepared / "evaluation_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    logits_by_scale = {}
    for scale in args.scales:
        logits, _ = _model_development_logits(
            model,
            tensors,
            device=device,
            batch_size=args.batch_size,
            tta={"enabled": False},
            primary_crop_scale=None if scale == 1.0 else scale,
        )
        logits_by_scale[float(scale)] = logits
    candidates = fusion_candidates(logits_by_scale, targets, config)
    logits_output = getattr(args, "logits_output", None)
    if logits_output is not None:
        logits_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            logits_output,
            **{f"scale_{scale:.3f}": values for scale, values in logits_by_scale.items()},
            targets=targets,
        )
    report = {
        "evaluation": "classifier_center_crop_tta_diagnostic_only",
        "sample_count": len(targets),
        "candidate_count": len(candidates),
        "selected": candidates[0],
        "top_candidates": candidates[:20],
        "passes_top1_gate": candidates[0]["top1_accuracy"] >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe center-crop TTA for the bread classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
