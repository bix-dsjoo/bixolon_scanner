from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ....runtime.onnx import prepare_rgb
from ....training.bread_dataset import audit_bread_dataset
from ....training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ....training.models import require_torch
from .pretrained_probe import _evaluation_samples

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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


def _metrics(logits: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-logits, axis=1, kind="stable")
    top1 = order[:, 0] == targets
    top3 = np.any(order[:, :3] == targets[:, None], axis=1)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        return {
            "sample_count": int(mask.sum()),
            "top1_accuracy": float(top1[mask].mean()),
            "top3_accuracy": float(top3[mask].mean()),
            "top1_error_count": int(np.count_nonzero(~top1[mask])),
        }

    return {
        "ALL": summarize(np.ones(len(targets), dtype=bool)),
        **{group.upper(): summarize(groups == group) for group in sorted(set(groups))},
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    root = args.dataset_root.resolve()
    _, metadata = audit_bread_dataset(root)
    samples = list(_evaluation_samples(root))
    targets = np.asarray([row[1] for row in samples], dtype=np.int64)
    groups = np.asarray([row[2] for row in samples])
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
    results = []
    for image_size in args.image_sizes:
        logits = []
        elapsed = 0.0
        with torch.inference_mode():
            for start in range(0, len(samples), args.batch_size):
                batch = np.asarray(
                    [
                        prepare_rgb(
                            row[0],
                            (image_size, image_size),
                            MEAN,
                            STD,
                            reducing_gap=1.0,
                        )
                        for row in samples[start : start + args.batch_size]
                    ],
                    dtype=np.float32,
                )
                pixels = torch.from_numpy(batch).to(device)
                pixels = _center_crop(torch, pixels, args.crop_scale)
                started = time.perf_counter()
                output = model(pixels)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - started
                logits.append(output.float().cpu().numpy())
        values = np.concatenate(logits)
        result = {
            "image_size": image_size,
            "metrics": _metrics(values, targets, groups),
            "encoder_and_head_mean_ms": elapsed * 1000.0 / len(samples),
            "batch_size": args.batch_size,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    selected = max(results, key=lambda row: row["metrics"]["ALL"]["top1_accuracy"])
    report = {
        "schema_version": "1.0",
        "evaluation": "classifier_input_resolution_probe",
        "promotion_status": "diagnostic_only",
        "dataset_version": metadata["dataset_version"],
        "training_contract": {
            "source_original_count": 200,
            "evaluation_images_used_for_training": False,
        },
        "crop_contract": "ground_truth_bbox_plus_5_percent_margin_box_resize",
        "selected": selected,
        "runs": results,
        "passes_top1_gate": selected["metrics"]["ALL"]["top1_accuracy"] >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe classifier input resolution")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-sizes", type=int, nargs="+", default=(224, 320, 384, 448))
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
