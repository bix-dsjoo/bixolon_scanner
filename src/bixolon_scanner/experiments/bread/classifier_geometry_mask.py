from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...pipeline.ports import Detection
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_neighbor_ownership_mask,
)
from ...training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ...training.models import require_torch


def neighbor_ownership_mask(
    *,
    image_width: int,
    image_height: int,
    boxes: Sequence[Sequence[float]],
    target_index: int,
    output_size: int,
    margin_ratio: float,
    distance_bias: float,
    shared_scale: bool,
) -> np.ndarray:
    detections = [Detection(*values, score=1.0) for values in boxes]
    return classifier_neighbor_ownership_mask(
        detections,
        target_index,
        image_width=image_width,
        image_height=image_height,
        output_size=output_size,
        margin_ratio=margin_ratio,
        distance_bias=distance_bias,
        shared_scale=shared_scale,
    )


def apply_background_mask(tensors: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return apply_classifier_background_masks(tensors, masks)


def _metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-logits, axis=1, kind="stable")
    return {
        "top1_error_count": int(np.count_nonzero(order[:, 0] != targets)),
        "top3_miss_count": int(np.count_nonzero(~np.any(order[:, :3] == targets[:, None], axis=1))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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
    predictions = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    manifest = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    if len(tensors) != len(rows):
        raise ValueError("evaluation tensors and records are not aligned")
    variants = [
        (shared_scale, distance_bias)
        for shared_scale in (False, True)
        for distance_bias in args.distance_biases
    ]
    masks = {}
    for shared_scale, distance_bias in variants:
        name = f"{'shared' if shared_scale else 'normalized'}_bias{distance_bias:.3f}"
        masks[name] = np.stack(
            [
                neighbor_ownership_mask(
                    image_width=int(manifest[int(row["image_id"])]["width"]),
                    image_height=int(manifest[int(row["image_id"])]["height"]),
                    boxes=predictions[int(row["image_id"])]["boxes_xyxy"],
                    target_index=int(row["detection_index"]),
                    output_size=tensors.shape[-1],
                    margin_ratio=args.margin_ratio,
                    distance_bias=distance_bias,
                    shared_scale=shared_scale,
                )
                for row in rows
            ]
        )
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in masks}
    with torch.inference_mode():
        for start in range(0, len(tensors), args.batch_size):
            batch = np.array(tensors[start : start + args.batch_size], dtype=np.float32, copy=True)
            for name, values in masks.items():
                masked = apply_background_mask(batch, values[start : start + len(batch)])
                pixels = torch.from_numpy(masked).to(device)
                outputs[name].append(model(pixels).float().cpu().numpy())
    logits = {name: np.concatenate(parts).astype(np.float32) for name, parts in outputs.items()}
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.logits_output, targets=targets, **logits)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_neighbor_ownership_mask_probe",
        "sample_count": len(targets),
        "margin_ratio": args.margin_ratio,
        "variants": {
            name: {
                **_metrics(values, targets),
                "masked_pixel_rate": float(masks[name].mean()),
                "masked_sample_count": int(np.count_nonzero(masks[name].any(axis=(1, 2)))),
            }
            for name, values in logits.items()
        },
        "geometry_only_mask": True,
        "evaluation_targets_used_for_mask": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suppress neighbor-owned pixels in classifier ROIs"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-biases", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--margin-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--hub-repository",
        default="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
    )
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
