from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..imaging import decode_image, image_original_size
from ..inference import Detection, build_onnx_adapters
from ..package import load_model_package
from .evaluate_difficulty import _load_records, _match_detections


GT_MATCHED = (40, 220, 80)
GT_MISSED = (255, 45, 45)
PRED_MATCHED = (0, 180, 255)
PRED_FALSE_POSITIVE = (255, 170, 0)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = position
    box = draw.textbbox((x, y), text, font=font, stroke_width=1)
    draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))


def _xywh(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x, y, x + width, y + height


def _render_overlay(
    record: dict[str, Any],
    detections: list[Detection],
    matches: dict[int, tuple[int, float]],
    missed_gt: set[int],
    output_path: Path,
) -> None:
    with Image.open(record["image_path"]) as source:
        source.seek(0)
        canvas = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(5, round(min(width, height) / 350))
    font = _font(max(22, round(min(width, height) / 85)))

    for gt_index, annotation in enumerate(record["annotations"]):
        missed = gt_index in missed_gt
        color = GT_MISSED if missed else GT_MATCHED
        box = _xywh([float(value) for value in annotation["bbox"]])
        draw.rectangle(box, outline=color, width=line_width * 2 if missed else line_width)
        class_id = f"bread_{int(annotation['category_id']):02d}"
        _label(
            draw,
            (box[0] + line_width, box[1] + line_width),
            f"GT {class_id} {'MISSED' if missed else 'MATCH'}",
            color,
            font,
        )

    for detection_index, detection in enumerate(detections):
        match = matches.get(detection_index)
        color = PRED_MATCHED if match else PRED_FALSE_POSITIVE
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(box, outline=color, width=line_width * 2 if match is None else line_width)
        text = (
            f"PRED score={detection.score:.3f} IoU={match[1]:.3f}"
            if match
            else f"PRED FALSE POSITIVE score={detection.score:.3f}"
        )
        label_y = max(0.0, box[1] - font.size - line_width * 3)
        _label(draw, (box[0] + line_width, label_y), text, color, font)

    legend = [
        (GT_MISSED, "RED: missed ground truth"),
        (PRED_FALSE_POSITIVE, "ORANGE: false-positive prediction"),
        (GT_MATCHED, "GREEN: matched ground truth"),
        (PRED_MATCHED, "BLUE: matched prediction"),
    ]
    legend_font = _font(max(20, round(min(width, height) / 95)))
    padding = line_width * 2
    legend_height = (legend_font.size + padding) * len(legend) + padding
    legend_width = max(draw.textlength(text, font=legend_font) for _, text in legend) + padding * 3
    draw.rectangle((0, 0, legend_width, legend_height), fill=(0, 0, 0))
    for row, (color, text) in enumerate(legend):
        y = padding + row * (legend_font.size + padding)
        draw.rectangle((padding, y, padding * 2, y + legend_font.size), fill=color)
        draw.text((padding * 2.5, y), text, fill=color, font=legend_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92, optimize=True)


def _contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumb_width, thumb_height = 900, 1200
    margin = 24
    title_height = 58
    sheet = Image.new(
        "RGB",
        (thumb_width * 2 + margin * 3, (thumb_height + title_height) * 2 + margin * 3),
        (25, 25, 25),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(30)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        column, row = index % 2, index // 2
        x = margin + column * (thumb_width + margin)
        y = margin + row * (thumb_height + title_height + margin)
        draw.text((x, y), path.stem, fill=(255, 255, 255), font=font)
        paste_x = x + (thumb_width - image.width) // 2
        paste_y = y + title_height + (thumb_height - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90, optimize=True)


def render(args: argparse.Namespace) -> list[Path]:
    with args.details.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    target_paths = {
        Path(row["image"]).resolve()
        for row in rows
        if row["error_type"] in {"DETECTOR_MISSED_GT", "DETECTOR_FALSE_POSITIVE"}
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
        record = records.get(image_path)
        if record is None:
            raise ValueError(f"detail image is not in dataset: {image_path}")
        image = decode_image(
            image_path.read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        result = detector.detect(image)
        detections = sorted(result.detections, key=lambda item: (item.y1, item.x1))
        matches, missed_gt = _match_detections(
            detections, record["annotations"], args.match_iou_threshold
        )
        if image_original_size(image) == (0, 0):
            raise RuntimeError("decoded image has invalid original size")
        output_path = args.output_dir / f"{image_path.stem}_overlay.jpg"
        _render_overlay(record, detections, matches, missed_gt, output_path)
        output_paths.append(output_path)
    _contact_sheet(output_paths, args.output_dir / "detection_errors_contact_sheet.jpg")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render missed/false-positive bbox overlays")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    paths = render(parser.parse_args())
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
