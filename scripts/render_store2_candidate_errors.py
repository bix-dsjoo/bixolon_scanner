from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _iou(box: list[float], other: list[float]) -> float:
    intersection = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
        0.0, min(box[3], other[3]) - max(box[1], other[1])
    )
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    return intersection / max(area + other_area - intersection, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render broad candidate false positives")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {int(row["image_id"]): row for row in _jsonl(args.manifest)}
    candidates: dict[int, list[dict]] = defaultdict(list)
    for row in _jsonl(args.candidates):
        candidates[int(row["image_id"])].append(row)
    false_positives: dict[int, list[dict]] = defaultdict(list)
    true_positives: dict[int, list[dict]] = defaultdict(list)
    for image_id, record in records.items():
        truth = [
            [
                row["bbox_xywh"][0],
                row["bbox_xywh"][1],
                row["bbox_xywh"][0] + row["bbox_xywh"][2],
                row["bbox_xywh"][1] + row["bbox_xywh"][3],
            ]
            for row in record["annotations"]
        ]
        unmatched = set(range(len(truth)))
        for candidate in sorted(
            candidates[image_id], key=lambda row: row["detector_score"], reverse=True
        ):
            best = (
                max(unmatched, key=lambda index: _iou(candidate["box"], truth[index]))
                if unmatched
                else None
            )
            if best is not None and _iou(candidate["box"], truth[best]) >= 0.5:
                unmatched.remove(best)
                true_positives[image_id].append(candidate)
            else:
                false_positives[image_id].append(candidate)

    image_ids = sorted(false_positives)
    tile_width = 640
    tile_height = 520
    columns = 2
    sheet = Image.new(
        "RGB",
        (columns * tile_width, ((len(image_ids) + columns - 1) // columns) * tile_height),
        "#181818",
    )
    font = ImageFont.load_default(size=18)
    for tile_index, image_id in enumerate(image_ids):
        record = records[image_id]
        with Image.open(args.dataset_root / record["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(image)
        line_width = max(7, round(max(image.size) / 600))
        for annotation in record["annotations"]:
            x, y, width, height = annotation["bbox_xywh"]
            draw.rectangle((x, y, x + width, y + height), outline="#3ee67a", width=line_width)
        for row in true_positives[image_id]:
            draw.rectangle(row["box"], outline="#4cc9f0", width=line_width)
        for row in false_positives[image_id]:
            draw.rectangle(row["box"], outline="#ff3b30", width=line_width * 2)
        image.thumbnail((tile_width - 20, tile_height - 65), Image.Resampling.LANCZOS)
        column = tile_index % columns
        row_index = tile_index // columns
        sheet.paste(
            image,
            (
                column * tile_width + (tile_width - image.width) // 2,
                row_index * tile_height + 55,
            ),
        )
        scores = ", ".join(
            f"d={row['detector_score']:.3f} v={row['verifier_score']:.3f}"
            for row in false_positives[image_id]
        )
        ImageDraw.Draw(sheet).text(
            (column * tile_width + 10, row_index * tile_height + 10),
            f"image {image_id}  FP {len(false_positives[image_id])}  {scores}",
            fill="white",
            font=font,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(args.output)


if __name__ == "__main__":
    main()
