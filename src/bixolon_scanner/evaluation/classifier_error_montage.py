from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def render_error_montage(
    dataset_root: Path,
    manifest_path: Path,
    trace_path: Path,
    output_path: Path,
) -> int:
    paths = {
        int(row["image_id"]): dataset_root / str(row["image_path"])
        for row in (
            json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        )
    }
    errors: list[tuple[dict, dict, dict]] = []
    for row in (json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()):
        for diagnostic in row.get("matched_classifier_diagnostics", []):
            if diagnostic["classifier_top1_correct"]:
                continue
            segmentation = row["decision"]["segmentations"][diagnostic["detection_index"]]
            errors.append((row, diagnostic, segmentation))

    tile_width, tile_height, columns = 320, 360, 4
    row_count = max(1, (len(errors) + columns - 1) // columns)
    canvas = Image.new("RGB", (tile_width * columns, tile_height * row_count), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (row, diagnostic, segmentation) in enumerate(errors):
        with Image.open(paths[int(row["image_id"])]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        box = segmentation["bbox"]
        x1, y1 = int(box["x"]), int(box["y"])
        crop = image.crop((x1, y1, x1 + int(box["width"]), y1 + int(box["height"])))
        crop.thumbnail((300, 290), Image.Resampling.LANCZOS)
        tile_x = (index % columns) * tile_width
        tile_y = (index // columns) * tile_height
        canvas.paste(crop, (tile_x + (tile_width - crop.width) // 2, tile_y + 4))
        text_y = tile_y + 300
        draw.text(
            (tile_x + 8, text_y),
            f"image {row['image_id']} score {diagnostic['approval_score']:.3f}",
            fill="black",
        )
        draw.text(
            (tile_x + 8, text_y + 16),
            (
                f"GT {diagnostic['target_class_id']} -> "
                f"PRED {diagnostic['classifier_top1_class_id']}"
            ),
            fill="red",
        )
        draw.text(
            (tile_x + 8, text_y + 32),
            (
                f"top2 {diagnostic['classifier_top2_class_id']} "
                f"sim {diagnostic['retrieval_top1_similarity']:.3f}"
            ),
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return len(errors)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render raw Scanner classifier errors")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    error_count = render_error_montage(args.dataset_root, args.manifest, args.trace, args.output)
    print(json.dumps({"error_count": error_count, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
