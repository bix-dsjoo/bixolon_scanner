from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.training.dinov3_objectness_detector import (
    build_dinov3_fasterrcnn,
)
from bixolon_scanner.training.models import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fresh DINOv3 Faster R-CNN for runtime v2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--maximum-queries", type=int, default=100)
    parser.add_argument("--containment-minimum", type=float, default=0.70)
    parser.add_argument("--containment-maximum-area-ratio", type=float, default=0.35)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    torch = require_torch()
    model = build_dinov3_fasterrcnn(
        args.weights,
        image_size=args.image_size,
        score_threshold=0.001,
        nms_threshold=0.4,
        detections_per_image=args.maximum_queries,
        fpn_channels=args.fpn_channels,
        unfreeze_last_stages=0,
        device="cpu",
    )
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    class _RuntimeWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # The unchanged runtime detector policy supplies 0.5/0.5 normalized
            # input. Recover RGB; Faster R-CNN applies DINOv3 normalization.
            rgb = pixel_values * 0.5 + 0.5
            output = self.model([rgb[0]])[0]
            boxes = output["boxes"]
            scores = output["scores"]
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            left_top = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
            right_bottom = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
            intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
            containment = intersection / area[:, None].clamp(min=1e-6)
            area_ratio = area[:, None] / area[None, :].clamp(min=1e-6)
            higher_or_equal = scores[None, :] >= scores[:, None]
            indexes = torch.arange(scores.shape[0], device=scores.device)
            diagonal = indexes[:, None] == indexes[None, :]
            suppressor = (
                (containment >= args.containment_minimum)
                & (area_ratio <= args.containment_maximum_area_ratio)
                & higher_or_equal
                & ~diagonal
            )
            keep = ~suppressor.any(dim=1)
            scores = torch.where(keep, scores, torch.full_like(scores, 1e-6))
            scores = scores.clamp(1e-6, 1.0 - 1e-6)
            objectness_logits = torch.log(scores / (1.0 - scores))
            background = torch.full(
                (scores.shape[0], 19),
                -20.0,
                dtype=objectness_logits.dtype,
                device=objectness_logits.device,
            )
            logits = torch.cat((objectness_logits.unsqueeze(-1), background), dim=-1)
            center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 / args.image_size
            center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 / args.image_size
            width = (boxes[:, 2] - boxes[:, 0]) / args.image_size
            height = (boxes[:, 3] - boxes[:, 1]) / args.image_size
            normalized_boxes = torch.stack((center_x, center_y, width, height), dim=-1)
            return logits.unsqueeze(0), normalized_boxes.unsqueeze(0)

    wrapper = _RuntimeWrapper().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, args.image_size, args.image_size),
        args.output,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    report = {
        "schema_version": "1.0",
        "architecture": "dinov3_convnext_tiny_fpn_fasterrcnn_runtime_v2",
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "official_weights": str(args.weights),
        "input_size": args.image_size,
        "maximum_queries": args.maximum_queries,
        "contained_fragment_filter": {
            "minimum_containment": args.containment_minimum,
            "maximum_area_ratio": args.containment_maximum_area_ratio,
            "action": "set detector score below unchanged runtime threshold",
        },
        "output": str(args.output),
        "runtime_detector_policy_normalization_preserved": {
            "mean": 0.5,
            "std": 0.5,
        },
        "internal_dinov3_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
