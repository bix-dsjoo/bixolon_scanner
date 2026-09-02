from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Store 2 wrong APPROVED crops")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {int(row["image_id"]): row for row in _jsonl(args.manifest)}
    trace = _jsonl(args.trace)
    tiles = []
    for row in trace:
        diagnostics = row.get("matched_classifier_diagnostics", [])
        bad = [
            value
            for value in diagnostics
            if value["final_status"] == "APPROVED" and not value["classifier_top1_correct"]
        ]
        if not bad:
            continue
        decision_boxes = [value["bbox"] for value in row["decision"]["segmentations"]]
        with Image.open(args.dataset_root / manifest[int(row["image_id"])]["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            for value in bad:
                box = decision_boxes[int(value["detection_index"])]
                left = int(box["x"])
                top = int(box["y"])
                right = left + int(box["width"])
                bottom = top + int(box["height"])
                crop = image.crop((left, top, right, bottom))
                crop.thumbnail((300, 240), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (320, 300), "white")
                tile.paste(crop, ((320 - crop.width) // 2, 8))
                draw = ImageDraw.Draw(tile)
                draw.text(
                    (8, 252),
                    f"image {row['image_id']}  {value['target_class_id']} -> ",
                    fill="black",
                )
                draw.text((8, 272), value["classifier_top1_class_id"], fill=(190, 0, 0))
                tiles.append(tile)
    if not tiles:
        raise ValueError("trace has no wrong APPROVED classifier crops")
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 300), (225, 225, 225))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 320, (index // columns) * 300))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(json.dumps({"wrong_approved_count": len(tiles), "output": str(args.output)}))


if __name__ == "__main__":
    main()
