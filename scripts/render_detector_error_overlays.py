from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from bixolon_scanner.evaluation.detector import _iou, _xywh_to_xyxy
from bixolon_scanner.evaluation.onnx_detector import _fuse_rotation_predictions
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.onnx import nms


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _select(prediction: dict, score_threshold: float, nms_threshold: float) -> list[Detection]:
    return nms(
        [
            Detection(*box, float(score))
            for box, score in zip(prediction["boxes_xyxy"], prediction["scores"], strict=True)
            if float(score) >= score_threshold
        ],
        nms_threshold,
    )


def _match(record: dict, detections: list[Detection], match_iou: float):
    gt_boxes = [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]]
    unmatched_gt = set(range(len(gt_boxes)))
    matched = []
    false_positives = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2])
        candidates = [(index, _iou(box, gt_boxes[index])) for index in unmatched_gt]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= match_iou:
                unmatched_gt.remove(index)
                matched.append((detection, index, overlap))
                continue
        false_positives.append(detection)
    return gt_boxes, matched, false_positives, sorted(unmatched_gt)


def _draw_box(draw: ImageDraw.ImageDraw, box, color: str, width: int, label: str) -> None:
    draw.rectangle(tuple(float(value) for value in box), outline=color, width=width)
    x1, y1 = int(box[0]), int(box[1])
    text_box = draw.textbbox((x1, y1), label)
    draw.rectangle(text_box, fill=color)
    draw.text((x1, y1), label, fill="white", font=ImageFont.load_default())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render detector FP/FN overlays")
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion-iou", type=float, default=0.4)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.35)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--matched-manifest-output", type=Path)
    args = parser.parse_args()

    records = _rows(args.evaluation_manifest)
    sources = [_rows(path) for path in args.predictions]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    summary = []
    matched_records = []
    for record, grouped in zip(records, zip(*sources), strict=True):
        prediction = {key: [] for key in ("boxes_xyxy", "scores", "class_ids", "top3_class_ids")}
        view_ids = []
        for source_index, row in enumerate(grouped):
            count = len(row["scores"])
            view_ids.extend([source_index] * count)
            for key in prediction:
                prediction[key].extend(row[key])
        if len(sources) > 1:
            prediction = _fuse_rotation_predictions(
                prediction,
                view_ids,
                view_count=len(sources),
                minimum_support=len(sources),
                iou_threshold=args.fusion_iou,
                score_mode="max",
            )
        detections = _select(prediction, args.score_threshold, args.nms_threshold)
        gt_boxes, matched, false_positives, false_negatives = _match(
            record, detections, args.match_iou
        )
        matched_record = dict(record)
        matched_record["annotations"] = []
        for detection, gt_index, _overlap in matched:
            gt = record["annotations"][gt_index]
            matched_record["annotations"].append(
                {
                    "annotation_id": int(gt["annotation_id"]),
                    "category_id": int(gt["category_id"]),
                    "bbox_xywh": [
                        float(detection.x1),
                        float(detection.y1),
                        float(detection.x2 - detection.x1),
                        float(detection.y2 - detection.y1),
                    ],
                }
            )
        matched_records.append(matched_record)
        if not false_positives and not false_negatives:
            continue
        source_path = args.evaluation_root / record["image_path"]
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(image)
        line_width = max(5, round(min(image.size) / 400))
        for detection, _gt_index, overlap in matched:
            _draw_box(
                draw,
                (detection.x1, detection.y1, detection.x2, detection.y2),
                "#19a974",
                line_width,
                f"MATCH {detection.score:.3f} IoU {overlap:.2f}",
            )
        for detection in false_positives:
            _draw_box(
                draw,
                (detection.x1, detection.y1, detection.x2, detection.y2),
                "#e5484d",
                line_width,
                f"FP {detection.score:.3f}",
            )
        for gt_index in false_negatives:
            category = int(record["annotations"][gt_index]["category_id"])
            _draw_box(draw, gt_boxes[gt_index], "#3e63dd", line_width, f"FN class {category}")
        draw.rectangle((0, 0, image.width, 58), fill="#111827")
        draw.text(
            (16, 14),
            f"{Path(record['image_path']).name} | FP {len(false_positives)} | FN {len(false_negatives)}",
            fill="white",
            font=ImageFont.load_default(),
        )
        output_path = args.output_dir / f"{int(record['image_id']):03d}.jpg"
        image.save(output_path, quality=92)
        summary.append(
            {
                "image_id": int(record["image_id"]),
                "image_path": str(record["image_path"]),
                "false_positive_count": len(false_positives),
                "false_negative_count": len(false_negatives),
                "overlay": output_path.name,
            }
        )
        thumbnail = image.copy()
        thumbnail.thumbnail((720, 540), Image.Resampling.LANCZOS)
        rendered.append((thumbnail, summary[-1]))

    columns = 2
    cell_width, cell_height = 740, 590
    rows = max(1, (len(rendered) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#f3f4f6")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (thumbnail, row) in enumerate(rendered):
        x = (index % columns) * cell_width + 10
        y = (index // columns) * cell_height + 40
        sheet.paste(thumbnail, (x, y))
        sheet_draw.text(
            (x, 12 + (index // columns) * cell_height),
            f"image {row['image_id']} | red FP {row['false_positive_count']} | blue FN {row['false_negative_count']}",
            fill="#111827",
            font=ImageFont.load_default(),
        )
    sheet_path = args.output_dir / "contact-sheet.jpg"
    sheet.save(sheet_path, quality=92)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "score_threshold": args.score_threshold,
                "fusion_iou": args.fusion_iou,
                "nms_threshold": args.nms_threshold,
                "error_image_count": len(summary),
                "false_positive_count": sum(row["false_positive_count"] for row in summary),
                "false_negative_count": sum(row["false_negative_count"] for row in summary),
                "images": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.matched_manifest_output:
        args.matched_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.matched_manifest_output.write_text(
            "".join(json.dumps(row) + "\n" for row in matched_records),
            encoding="utf-8",
        )
    print(sheet_path.resolve())


if __name__ == "__main__":
    main()
