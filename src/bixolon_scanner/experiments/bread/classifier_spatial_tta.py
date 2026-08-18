from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ...training.models import require_torch


def spatial_crop_bounds(
    height: int,
    width: int,
    *,
    scale: float,
    x_position: int,
    y_position: int,
) -> tuple[int, int, int, int]:
    if height < 1 or width < 1:
        raise ValueError("spatial crop requires positive dimensions")
    if not 0.5 <= scale <= 1.0:
        raise ValueError("spatial crop scale must be in [0.5, 1.0]")
    if x_position not in {-1, 0, 1} or y_position not in {-1, 0, 1}:
        raise ValueError("spatial crop positions must be -1, 0, or 1")
    crop_height = max(1, round(height * scale))
    crop_width = max(1, round(width * scale))
    top = round((height - crop_height) * (y_position + 1) / 2)
    left = round((width - crop_width) * (x_position + 1) / 2)
    return top, left, crop_height, crop_width


def _spatial_crop(torch, values, *, scale: float, x_position: int, y_position: int):
    top, left, crop_height, crop_width = spatial_crop_bounds(
        values.shape[-2],
        values.shape[-1],
        scale=scale,
        x_position=x_position,
        y_position=y_position,
    )
    return torch.nn.functional.interpolate(
        values[..., top : top + crop_height, left : left + crop_width],
        size=values.shape[-2:],
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def _metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-logits, axis=1, kind="stable")
    return {
        "top1_error_count": int(np.count_nonzero(order[:, 0] != targets)),
        "top3_miss_count": int(np.count_nonzero(~np.any(order[:, :3] == targets[:, None], axis=1))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=args.hub_repository,
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device).eval()
    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    if len(tensors) != len(targets):
        raise ValueError("evaluation tensors and records are not aligned")
    views = [
        (scale, x_position, y_position)
        for scale in args.scales
        for y_position in (-1, 0, 1)
        for x_position in (-1, 0, 1)
    ]
    outputs: dict[str, list[np.ndarray]] = {
        f"scale{scale:.3f}_x{x_position:+d}_y{y_position:+d}": []
        for scale, x_position, y_position in views
    }
    with torch.inference_mode():
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + args.batch_size], dtype=np.float32, copy=True)
            ).to(device)
            for scale, x_position, y_position in views:
                name = f"scale{scale:.3f}_x{x_position:+d}_y{y_position:+d}"
                cropped = _spatial_crop(
                    torch,
                    batch,
                    scale=scale,
                    x_position=x_position,
                    y_position=y_position,
                )
                outputs[name].append(model(cropped).float().cpu().numpy())
    logits = {name: np.concatenate(parts).astype(np.float32) for name, parts in outputs.items()}
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.logits_output, targets=targets, **logits)
    view_metrics = {name: _metrics(values, targets) for name, values in logits.items()}
    target_in_any_top3 = np.zeros(len(targets), dtype=bool)
    for values in logits.values():
        order = np.argsort(-values, axis=1, kind="stable")[:, :3]
        target_in_any_top3 |= np.any(order == targets[:, None], axis=1)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_spatial_tta_probe",
        "sample_count": len(targets),
        "scales": args.scales,
        "view_count": len(views),
        "view_metrics": view_metrics,
        "oracle_union_top3_miss_count": int(np.count_nonzero(~target_in_any_top3)),
        "limitations": [
            "The oracle union diagnoses available evidence and is not a deployable policy.",
            "No evaluation image identifiers or targets are inputs to spatial view creation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe label-agnostic spatial classifier crops")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.65, 0.75, 0.85))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--hub-repository",
        default="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
    )
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
