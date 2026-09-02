from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from transformers import SamModel, SamProcessor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe SAM for store-2 annotation only")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam-vit-base")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    rows = [
        json.loads(line)
        for line in (args.prepared_root / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    row = next(item for item in rows if int(item["image_id"]) == args.image_id)
    source_path = args.source_root / row["original_image_path"]
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    statistics = row["statistics"]
    point_x = float(statistics["prompt_x"])
    point_y = float(statistics["prompt_y"])
    point = [[[point_x, point_y]]]
    half_extent = round(min(image.width, image.height) * 0.35)
    left = max(0, round(point_x) - half_extent)
    top = max(0, round(point_y) - half_extent)
    right = min(image.width, round(point_x) + half_extent)
    bottom = min(image.height, round(point_y) + half_extent)
    box = [[[left, top, right, bottom]]]

    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to("cuda").eval()
    inputs = processor(image, input_points=point, input_boxes=box, return_tensors="pt")
    original_sizes = inputs.pop("original_sizes")
    reshaped_input_sizes = inputs.pop("reshaped_input_sizes")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), original_sizes, reshaped_input_sizes
    )[0][0]
    scores = outputs.iou_scores.detach().cpu()[0, 0]
    diagnostics = []
    canvas = Image.new("RGB", (image.width * int(masks.shape[0]), image.height), "white")
    for index, mask_tensor in enumerate(masks):
        mask = mask_tensor.numpy() > 0
        labels, _ = ndimage.label(mask)
        point_label = labels[round(point[0][0][1]), round(point[0][0][0])]
        if point_label:
            mask = labels == point_label
        ys, xs = np.nonzero(mask)
        diagnostics.append(
            {
                "candidate": index,
                "predicted_iou": float(scores[index]),
                "area_ratio": float(mask.mean()),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                "touches_image_edge": bool(
                    xs.min() == 0
                    or ys.min() == 0
                    or xs.max() == image.width - 1
                    or ys.max() == image.height - 1
                ),
            }
        )
        overlay = np.asarray(image).copy()
        overlay[mask] = np.rint(
            overlay[mask].astype(np.float32) * 0.65
            + np.asarray([0, 255, 80], dtype=np.float32) * 0.35
        ).astype(np.uint8)
        panel = Image.fromarray(overlay)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 0, 0), width=8)
        draw.ellipse(
            (point[0][0][0] - 8, point[0][0][1] - 8, point[0][0][0] + 8, point[0][0][1] + 8),
            fill=(0, 0, 255),
        )
        draw.text(
            (20, 20),
            f"candidate={index} predicted_iou={float(scores[index]):.4f}",
            fill=(255, 0, 0),
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
        canvas.paste(panel, (index * image.width, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((2400, 1200), Image.Resampling.LANCZOS)
    canvas.save(args.output, quality=92)
    print(json.dumps({"output": str(args.output), "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
