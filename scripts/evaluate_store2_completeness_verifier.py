from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.training.dinov3_objectness_detector import (
    load_dinov3_convnext_tiny,
)
from bixolon_scanner.training.models import require_torch

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _containment_keep(boxes, scores):
    torch = require_torch()
    if len(boxes) < 2:
        return torch.ones(len(boxes), dtype=torch.bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    left_top = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    right_bottom = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    containment = intersection / area[:, None].clamp(min=1e-6)
    area_ratio = area[:, None] / area[None, :].clamp(min=1e-6)
    higher_or_equal = scores[None, :] >= scores[:, None]
    diagonal = torch.eye(len(boxes), dtype=torch.bool)
    suppressor = (containment >= 0.70) & (area_ratio <= 0.55) & higher_or_equal & ~diagonal
    return ~suppressor.any(dim=1)


def _truth_boxes(record: dict):
    torch = require_torch()
    return torch.as_tensor(
        [
            [
                row["bbox_xywh"][0],
                row["bbox_xywh"][1],
                row["bbox_xywh"][0] + row["bbox_xywh"][2],
                row["bbox_xywh"][1] + row["bbox_xywh"][3],
            ]
            for row in record["annotations"]
        ],
        dtype=torch.float32,
    ).reshape(-1, 4)


def _evaluate(records: list[dict], candidates: dict[int, list[dict]], threshold: float):
    torch = require_torch()
    from torchvision.ops import box_iou

    matched = 0
    false_positive = 0
    failures = []
    for record in records:
        rows = [
            row for row in candidates[int(record["image_id"])] if row["verifier_score"] >= threshold
        ]
        rows.sort(key=lambda row: row["detector_score"], reverse=True)
        boxes = torch.as_tensor([row["box"] for row in rows], dtype=torch.float32).reshape(-1, 4)
        truth = _truth_boxes(record)
        overlaps = box_iou(boxes, truth) if len(boxes) and len(truth) else None
        unmatched = set(range(len(truth)))
        unmatched_predictions = 0
        for index in range(len(boxes)):
            if not unmatched:
                unmatched_predictions += len(boxes) - index
                break
            best = max(unmatched, key=lambda target: float(overlaps[index, target]))
            if float(overlaps[index, best]) < 0.5:
                unmatched_predictions += 1
                continue
            unmatched.remove(best)
            matched += 1
        false_positive += unmatched_predictions
        if unmatched or unmatched_predictions:
            failures.append(
                {
                    "image_id": int(record["image_id"]),
                    "false_negative_count": len(unmatched),
                    "false_positive_count": unmatched_predictions,
                }
            )
    ground_truth_count = sum(len(row["annotations"]) for row in records)
    return {
        "verifier_threshold": threshold,
        "matched_count": matched,
        "false_negative_count": ground_truth_count - matched,
        "false_positive_count": false_positive,
        "total_error_count": ground_truth_count - matched + false_positive,
        "failure_image_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate broad proposals with DINO verifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--detector-threshold", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    torch = require_torch()

    class Verifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = load_dinov3_convnext_tiny(args.weights, device="cuda")
            self.classifier = torch.nn.Linear(768, 2)

        def forward(self, pixel_values):
            return self.classifier(self.backbone(pixel_values))

    verifier = Verifier().cuda().eval()
    verifier.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    records = _jsonl(args.manifest)
    raw_predictions = {int(row["image_id"]): row for row in _jsonl(args.predictions)}
    candidates: dict[int, list[dict]] = {}
    tensors = []
    flat_rows = []
    for record in records:
        image_id = int(record["image_id"])
        raw = raw_predictions[image_id]
        boxes = torch.as_tensor(raw["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(raw["scores"], dtype=torch.float32)
        eligible = scores >= args.detector_threshold
        boxes, scores = boxes[eligible], scores[eligible]
        keep = _containment_keep(boxes, scores)
        boxes, scores = boxes[keep], scores[keep]
        candidates[image_id] = []
        with Image.open(args.dataset_root / record["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for box, detector_score in zip(boxes.tolist(), scores.tolist(), strict=True):
                left, top, right, bottom = box
                crop = image.crop(
                    (
                        max(0, int(np.floor(left))),
                        max(0, int(np.floor(top))),
                        min(image.width, int(np.ceil(right))),
                        min(image.height, int(np.ceil(bottom))),
                    )
                ).resize((192, 192), Image.Resampling.BICUBIC)
                values = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
                tensors.append((values - MEAN) / STD)
                row = {
                    "image_id": image_id,
                    "box": box,
                    "detector_score": float(detector_score),
                }
                candidates[image_id].append(row)
                flat_rows.append(row)

    verifier_scores = []
    with torch.inference_mode():
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.from_numpy(
                np.asarray(tensors[start : start + args.batch_size], dtype=np.float32)
            ).cuda()
            logits = verifier(batch)
            verifier_scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    for row, score in zip(flat_rows, verifier_scores, strict=True):
        row["verifier_score"] = float(score)

    unique_scores = sorted(set(verifier_scores))
    thresholds = [0.0, 1.0]
    thresholds.extend(
        (left + right) * 0.5 for left, right in zip(unique_scores, unique_scores[1:], strict=False)
    )
    results = [_evaluate(records, candidates, threshold) for threshold in thresholds]
    results.sort(
        key=lambda row: (
            row["total_error_count"],
            row["false_positive_count"],
            row["false_negative_count"],
        )
    )
    report = {
        "schema_version": "1.0",
        "detector_threshold": args.detector_threshold,
        "containment": {"minimum": 0.70, "maximum_area_ratio": 0.55},
        "candidate_count": len(flat_rows),
        "best_candidates": results[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in flat_rows),
        encoding="utf-8",
    )
    print(json.dumps(results[:20], indent=2))


if __name__ == "__main__":
    main()
