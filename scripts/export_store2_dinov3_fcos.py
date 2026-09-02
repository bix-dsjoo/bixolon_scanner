from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.training.dinov3_objectness_detector import build_dinov3_fcos
from bixolon_scanner.training.models import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fresh DINOv3 FCOS for runtime v2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--maximum-queries", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch = require_torch()
    from torchvision.ops import nms

    model = build_dinov3_fcos(
        args.weights,
        image_size=args.image_size,
        score_threshold=0.01,
        nms_threshold=0.4,
        detections_per_image=args.maximum_queries,
        topk_candidates=1000,
        fpn_channels=args.fpn_channels,
        unfreeze_last_stages=0,
        device="cpu",
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        sample_rgb = torch.zeros(3, args.image_size, args.image_size)
        image_list, _ = model.transform([sample_rgb], None)
        sample_features = list(model.backbone(image_list.tensors).values())
        level_anchor_counts = tuple(
            int(feature.shape[-2] * feature.shape[-1]) for feature in sample_features
        )
        anchors = model.anchor_generator(image_list, sample_features)[0]

    class _RuntimeWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = model.backbone
            self.head = model.head
            self.box_coder = model.box_coder
            self.register_buffer("anchors", anchors)
            self.register_buffer(
                "imagenet_mean",
                torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
            )
            self.register_buffer(
                "imagenet_std",
                torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
            )

        def forward(self, pixel_values):
            # Runtime policy keeps its historical 0.5/0.5 normalization. Recover
            # RGB here, then apply the DINOv3 ImageNet normalization internally.
            rgb = pixel_values * 0.5 + 0.5
            normalized = (rgb - self.imagenet_mean) / self.imagenet_std
            features = list(self.backbone(normalized).values())
            outputs = self.head(features)
            scores = torch.sqrt(
                torch.sigmoid(outputs["cls_logits"].squeeze(-1))
                * torch.sigmoid(outputs["bbox_ctrness"].squeeze(-1))
            )[0]
            level_scores = []
            level_indices = []
            offset = 0
            for anchor_count in level_anchor_counts:
                candidate_count = min(1000, anchor_count)
                candidate_scores, candidate_indices = torch.topk(
                    scores[offset : offset + anchor_count], candidate_count
                )
                level_scores.append(candidate_scores)
                level_indices.append(candidate_indices + offset)
                offset += anchor_count
            candidate_scores = torch.cat(level_scores)
            candidate_indices = torch.cat(level_indices)
            boxes = self.box_coder.decode(
                outputs["bbox_regression"][0, candidate_indices],
                self.anchors[candidate_indices],
            ).clamp(0.0, float(args.image_size))
            keep = nms(boxes, candidate_scores, 0.4)[: args.maximum_queries]
            boxes = boxes[keep]
            selected_scores = candidate_scores[keep].clamp(1e-6, 1.0 - 1e-6)
            objectness_logits = torch.log(selected_scores / (1.0 - selected_scores))
            background = torch.full(
                (keep.shape[0], 19),
                -20.0,
                dtype=objectness_logits.dtype,
                device=objectness_logits.device,
            )
            logits = torch.cat((objectness_logits.unsqueeze(-1), background), dim=-1)
            center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 / args.image_size
            center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 / args.image_size
            width = (boxes[:, 2] - boxes[:, 0]) / args.image_size
            height = (boxes[:, 3] - boxes[:, 1]) / args.image_size
            return logits.unsqueeze(0), torch.stack(
                (center_x, center_y, width, height), dim=-1
            ).unsqueeze(0)

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
        "architecture": "dinov3_convnext_tiny_fpn_fcos_runtime_v2",
        "checkpoint": str(args.checkpoint),
        "official_weights": str(args.weights),
        "input_size": args.image_size,
        "maximum_queries": args.maximum_queries,
        "fpn_level_anchor_counts": level_anchor_counts,
        "topk_candidates_per_level": 1000,
        "output": str(args.output),
        "runtime_detector_policy_normalization_preserved": {"mean": 0.5, "std": 0.5},
        "internal_dinov3_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
