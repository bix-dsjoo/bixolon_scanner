from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Store 2 detector error overlays")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {int(row["image_id"]): row for row in _jsonl(args.manifest)}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    false_positives: dict[int, list[dict]] = defaultdict(list)
    false_negatives: dict[int, list[dict]] = defaultdict(list)
    for row in report["false_positives"]:
        false_positives[int(row["image_id"])].append(row)
    for row in report["false_negatives"]:
        false_negatives[int(row["image_id"])].append(row)
    image_ids = sorted(set(false_positives) | set(false_negatives))
    tile_width = 520
    tile_height = 420
    columns = 3
    rows = (len(image_ids) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#181818")
    font = ImageFont.load_default(size=18)
    for tile_index, image_id in enumerate(image_ids):
        record = records[image_id]
        with Image.open(args.dataset_root / record["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(image)
        line_width = max(8, round(max(image.size) / 500))
        for annotation in record["annotations"]:
            x, y, width, height = annotation["bbox_xywh"]
            draw.rectangle((x, y, x + width, y + height), outline="#3ee67a", width=line_width)
        for row_value in false_positives.get(image_id, []):
            draw.rectangle(row_value["box"], outline="#ff3b30", width=line_width * 2)
        for row_value in false_negatives.get(image_id, []):
            draw.rectangle(row_value["box"], outline="#ffd60a", width=line_width * 2)
        image.thumbnail((tile_width - 20, tile_height - 55), Image.Resampling.LANCZOS)
        x_offset = tile_index % columns * tile_width + (tile_width - image.width) // 2
        y_offset = tile_index // columns * tile_height + 42
        sheet.paste(image, (x_offset, y_offset))
        label = (
            f"image {image_id}  FP {len(false_positives.get(image_id, []))}  "
            f"FN {len(false_negatives.get(image_id, []))}"
        )
        ImageDraw.Draw(sheet).text(
            (tile_index % columns * tile_width + 10, tile_index // columns * tile_height + 10),
            label,
            fill="white",
            font=font,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
