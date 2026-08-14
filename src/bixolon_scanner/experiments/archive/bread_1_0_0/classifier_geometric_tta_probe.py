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


def _center_crop(torch, values, scale: float):
    height, width = values.shape[-2:]
    crop_height, crop_width = round(height * scale), round(width * scale)
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return torch.nn.functional.interpolate(
        values[..., top : top + crop_height, left : left + crop_width],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def _view(torch, values, name: str):
    if name == "base":
        return values
    if name == "hflip":
        return torch.flip(values, dims=(-1,))
    if name == "vflip":
        return torch.flip(values, dims=(-2,))
    if name == "rot90":
        return torch.rot90(values, 1, dims=(-2, -1))
    if name == "rot180":
        return torch.rot90(values, 2, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(values, 3, dims=(-2, -1))
    from torchvision.transforms.functional import InterpolationMode, rotate

    angle = float(name.removeprefix("rot"))
    return rotate(
        values,
        angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0.0,
    )


def _logits_by_view(model, tensors, *, names, torch, device, batch_size: int, crop_scale: float):
    results = {name: [] for name in names}
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            batch = _center_crop(torch, batch, crop_scale)
            for name in names:
                results[name].append(model(_view(torch, batch, name)).float().cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in results.items()}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device).eval()
    tensors = np.load(args.tensors, mmap_mode="r")
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    names = (
        "base",
        "hflip",
        "vflip",
        "rot90",
        "rot180",
        "rot270",
        "rot15",
        "rot-15",
        "rot30",
        "rot-30",
    )
    logits = _logits_by_view(
        model,
        tensors,
        names=names,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
        crop_scale=args.crop_scale,
    )
    candidates = []
    for size in range(1, 7):
        for selected in combinations(names, size):
            values = np.mean([logits[name] for name in selected], axis=0)
            accuracy = float((values.argmax(axis=1) == targets).mean())
            candidates.append(
                {"views": list(selected), "view_count": size, "top1_accuracy": accuracy}
            )
    candidates.sort(key=lambda row: (row["top1_accuracy"], -row["view_count"]), reverse=True)
    view_predictions = np.stack([logits[name].argmax(axis=1) for name in names])
    oracle = np.any(view_predictions == targets[None, :], axis=0)
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.logits_output, targets=targets, **logits)
    report = {
        "schema_version": "1.0",
        "evaluation": "geometric_tta_fixed_average_probe",
        "sample_count": len(targets),
        "selected": candidates[0],
        "top_candidates": candidates[:20],
        "view_oracle": {
            "top1_accuracy": float(oracle.mean()),
            "error_count": int(np.count_nonzero(~oracle)),
        },
        "passes_top1_gate": candidates[0]["top1_accuracy"] >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe geometric TTA on a classifier candidate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
