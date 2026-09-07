from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#f3f4f6")
    sample = image.copy()
    sample.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(sample, ((size[0] - sample.width) // 2, (size[1] - sample.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render classifier error comparisons")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    records = {int(row["image_id"]): row for row in _jsonl(args.evaluation_manifest)}
    support = {}
    for row in _jsonl(args.support_manifest):
        support.setdefault(int(row["category_id"]), row)

    errors = report.get("errors", report.get("target_remaining_errors"))
    if errors is None:
        raise ValueError("report has neither errors nor target_remaining_errors")
    cell_width, cell_height = 930, 310
    sheet = Image.new("RGB", (cell_width, cell_height * len(errors)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, error in enumerate(errors):
        record = records[int(error["image_id"])]
        annotation = next(
            row
            for row in record["annotations"]
            if int(row["annotation_id"]) == int(error["annotation_id"])
        )
        x, y, width, height = annotation["bbox_xywh"]
        with Image.open(args.evaluation_root / record["image_path"]) as opened:
            target = (
                ImageOps.exif_transpose(opened).convert("RGB").crop((x, y, x + width, y + height))
            )
        expected = int(error["expected"])
        predicted = int(error["predicted"])
        with Image.open(args.support_root / support[expected]["image_path"]) as opened:
            expected_image = ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(args.support_root / support[predicted]["image_path"]) as opened:
            predicted_image = ImageOps.exif_transpose(opened).convert("RGB")

        top = index * cell_height
        sheet.paste(_fit(target, (290, 250)), (10, top + 45))
        sheet.paste(_fit(expected_image, (290, 250)), (320, top + 45))
        sheet.paste(_fit(predicted_image, (290, 250)), (620, top + 45))
        draw.rectangle((0, top, cell_width, top + 38), fill="#111827")
        draw.text(
            (12, top + 12),
            f"image {error['image_id']} ann {error['annotation_id']} | target | expected {expected} | predicted {predicted}",
            fill="white",
            font=font,
        )
        draw.text((20, top + 48), "TARGET", fill="#111827", font=font)
        draw.text((330, top + 48), f"EXPECTED {expected}", fill="#111827", font=font)
        draw.text((630, top + 48), f"PREDICTED {predicted}", fill="#111827", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
