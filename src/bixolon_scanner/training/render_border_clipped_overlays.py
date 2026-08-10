from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from ..imaging import decode_image
from ..inference import build_onnx_adapters
from ..package import load_model_package
from .evaluate_difficulty import _load_records
from .render_detection_overlays import _contact_sheet, _font, _label, _xywh


GT_COLOR = (40, 220, 80)
PRED_COLOR = (0, 180, 255)
CLIPPED_COLOR = (255, 0, 200)
SAFE_BORDER_COLOR = (255, 230, 0)


def render(args: argparse.Namespace) -> list[Path]:
    with args.details.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    target_paths = {
        Path(row["image"]).resolve()
        for row in rows
        if row["error_type"] == "RECAPTURE"
        and "DETECTOR_BORDER_CLIPPED" in row["reason_codes"].split("|")
    }
    package = load_model_package(args.package_dir)
    detector, _, _ = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    records = {
        record["image_path"].resolve(): record for record in _load_records(args.dataset_root)
    }
    output_paths: list[Path] = []
    for image_path in sorted(target_paths):
        record = records[image_path]
        decoded = decode_image(
            image_path.read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        result = detector.detect(decoded)
        with Image.open(image_path) as source:
            source.seek(0)
            canvas = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        line_width = max(5, round(min(width, height) / 350))
        font = _font(max(22, round(min(width, height) / 85)))
        border_x = width * package.metadata.quality.border_margin_ratio
        border_y = height * package.metadata.quality.border_margin_ratio
        draw.rectangle(
            (border_x, border_y, width - border_x, height - border_y),
            outline=SAFE_BORDER_COLOR,
            width=line_width,
        )

        for annotation in record["annotations"]:
            box = _xywh([float(value) for value in annotation["bbox"]])
            draw.rectangle(box, outline=GT_COLOR, width=line_width)
            class_id = f"bread_{int(annotation['category_id']):02d}"
            _label(draw, (box[0] + line_width, box[1] + line_width), f"GT {class_id}", GT_COLOR, font)

        clipped_count = 0
        for detection in result.detections:
            sides: list[str] = []
            if detection.x1 <= border_x:
                sides.append("LEFT")
            if detection.y1 <= border_y:
                sides.append("TOP")
            if detection.x2 >= width - border_x:
                sides.append("RIGHT")
            if detection.y2 >= height - border_y:
                sides.append("BOTTOM")
            clipped = bool(sides)
            clipped_count += int(clipped)
            color = CLIPPED_COLOR if clipped else PRED_COLOR
            box = (detection.x1, detection.y1, detection.x2, detection.y2)
            draw.rectangle(box, outline=color, width=line_width * 2 if clipped else line_width)
            text = (
                f"BORDER CLIPPED {'+'.join(sides)} score={detection.score:.3f}"
                if clipped
                else f"PRED score={detection.score:.3f}"
            )
            label_y = max(0.0, box[1] - font.size - line_width * 3)
            _label(draw, (box[0] + line_width, label_y), text, color, font)
        if clipped_count == 0:
            raise RuntimeError(f"no clipped detection found for {image_path}")

        legend_font = _font(max(20, round(min(width, height) / 95)))
        legend = [
            (CLIPPED_COLOR, "MAGENTA: bbox that triggered RECAPTURE"),
            (SAFE_BORDER_COLOR, f"YELLOW: safe boundary ({package.metadata.quality.border_margin_ratio:.3%})"),
            (GT_COLOR, "GREEN: ground-truth bbox"),
            (PRED_COLOR, "BLUE: other predicted bbox"),
        ]
        padding = line_width * 2
        legend_height = (legend_font.size + padding) * len(legend) + padding
        legend_width = max(draw.textlength(text, font=legend_font) for _, text in legend) + padding * 3
        draw.rectangle((0, 0, legend_width, legend_height), fill=(0, 0, 0))
        for row_index, (color, text) in enumerate(legend):
            y = padding + row_index * (legend_font.size + padding)
            draw.rectangle((padding, y, padding * 2, y + legend_font.size), fill=color)
            draw.text((padding * 2.5, y), text, fill=color, font=legend_font)

        output_path = args.output_dir / f"{image_path.stem}_border_clipped.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="JPEG", quality=92, optimize=True)
        output_paths.append(output_path)

    _contact_sheet(output_paths, args.output_dir / "border_clipped_contact_sheet.jpg")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DETECTOR_BORDER_CLIPPED overlays")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    paths = render(parser.parse_args())
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
