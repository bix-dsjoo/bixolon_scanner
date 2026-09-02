from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

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


class _CandidateDataset:
    def __init__(self, cache: Path, rows: list[dict]) -> None:
        metadata = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        self.images = np.load(cache / metadata["array_filename"], mmap_mode="r")
        self.index = {int(key): int(value) for key, value in metadata["index"].items()}
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        torch = require_torch()
        row = self.rows[index]
        image = Image.fromarray(
            np.asarray(self.images[self.index[int(row["image_id"])]]), mode="RGB"
        )
        left, top, right, bottom = row["box"]
        crop = image.crop(
            (
                max(0, math.floor(left)),
                max(0, math.floor(top)),
                min(image.width, math.ceil(right)),
                min(image.height, math.ceil(bottom)),
            )
        ).resize((192, 192), Image.Resampling.BICUBIC)
        values = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((values - MEAN) / STD)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score detector candidates with DINO verifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-threshold", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch = require_torch()

    class Verifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = load_dinov3_convnext_tiny(args.weights, device="cuda")
            self.classifier = torch.nn.Linear(768, 2)

        def forward(self, pixel_values):
            return self.classifier(self.backbone(pixel_values))

    records = {int(row["image_id"]): row for row in _jsonl(args.manifest)}
    rows = []
    for raw in _jsonl(args.predictions):
        image_id = int(raw["image_id"])
        boxes = torch.as_tensor(raw["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(raw["scores"], dtype=torch.float32)
        eligible = scores >= args.detector_threshold
        boxes, scores = boxes[eligible], scores[eligible]
        keep = _containment_keep(boxes, scores)
        for box, score in zip(boxes[keep].tolist(), scores[keep].tolist(), strict=True):
            rows.append(
                {
                    "image_id": image_id,
                    "capture_session_id": records[image_id]["capture_session_id"],
                    "box": box,
                    "detector_score": float(score),
                }
            )
    dataset = _CandidateDataset(args.cache, rows)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = Verifier().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    scores = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch.cuda(non_blocking=True))
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    for row, score in zip(rows, scores, strict=True):
        row["verifier_score"] = float(score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
