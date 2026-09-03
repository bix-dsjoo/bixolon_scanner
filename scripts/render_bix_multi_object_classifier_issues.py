from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BIX multi-object classifier issues")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scope", default="detector_boxes_runtime_preprocess")
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include every non-approved ROI in addition to wrong approvals and Top-3 misses",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    issues = [
        row
        for row in _rows(args.predictions)
        if row["scope"] == args.scope
        and (
            (row["approved"] and not row["top1_correct"])
            or not row["top3_hit"]
            or (args.include_unknown and not row["approved"])
        )
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in issues:
        grouped[str(row["image_path"])].append(row)
    if not grouped:
        raise ValueError("no wrong APPROVED or Top-3 miss rows found")

    cell_width, cell_height = 600, 580
    tiles: list[Image.Image] = []
    for image_path, image_rows in grouped.items():
        with Image.open(args.dataset_root / image_path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        display = original.copy()
        display.thumbnail((cell_width, 410), Image.Resampling.LANCZOS)
        scale_x = display.width / original.width
        scale_y = display.height / original.height
        draw = ImageDraw.Draw(display)
        labels = []
        for row in image_rows:
            box = row["bbox_xyxy"]
            wrong_approved = row["approved"] and not row["top1_correct"]
            top3_miss = not row["top3_hit"]
            color = (185, 45, 45) if wrong_approved or top3_miss else (38, 72, 96)
            draw.rectangle(
                (
                    box[0] * scale_x,
                    box[1] * scale_y,
                    box[2] * scale_x,
                    box[3] * scale_y,
                ),
                outline=color,
                width=6,
            )
            if wrong_approved:
                kind = "WRONG APPROVED"
            elif top3_miss:
                kind = "TOP-3 MISS"
            elif row["top1_correct"]:
                kind = "UNKNOWN / TOP-1 CORRECT"
            else:
                kind = "UNKNOWN / TOP-1 WRONG"
            labels.append(
                f"{kind}: {row['target_class_id']} -> {row['predicted_class_id']} "
                f"[{', '.join(row['top3'])}]"
            )
        tile = Image.new("RGB", (cell_width, cell_height), "white")
        tile.paste(display, ((cell_width - display.width) // 2, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((10, 416), f"image {image_rows[0]['image_id']} / {image_path}", fill="black")
        for index, label in enumerate(labels[:7]):
            tile_draw.text((10, 438 + index * 20), label, fill="black")
        tiles.append(tile)

    columns = 2
    row_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * row_count), (235, 235, 235))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * cell_width, (index // columns) * cell_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(
        json.dumps(
            {
                "issue_count": len(issues),
                "image_count": len(tiles),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
