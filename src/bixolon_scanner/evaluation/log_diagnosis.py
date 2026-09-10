"""Explain log detection errors without changing confirmed annotations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..configuration import load_json_config
from ..training.three_bakery_data import read_jsonl, write_json


def diagnose(manifest: Path, measurement: Path, output: Path) -> dict:
    records = {r["image_id"]: r for r in read_jsonl(manifest)}
    rows = [r for r in read_jsonl(measurement / "responses.jsonl") if r["repetition"] == 0]
    if {r["image_id"] for r in rows} != records.keys():
        raise ValueError("diagnosis requires a complete measured image set")
    output.mkdir(parents=True, exist_ok=True)
    issues, totals = [], Counter()
    for row in rows:
        metrics = row["metrics"]
        totals.update(
            {
                k: metrics[k]
                for k in (
                    "ground_truth_count",
                    "correct_approved_count",
                    "wrong_approved_count",
                    "wrong_class_approved_count",
                    "unmatched_approved_count",
                    "missed_count",
                    "extra_count",
                )
            }
        )
        if metrics["complete_image"]:
            continue
        record = records[row["image_id"]]
        issues.append({"image_id": row["image_id"], "metrics": metrics})
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        scale = min(1.0, 1400 / image.width)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
        for index, target in enumerate(record["annotations"]):
            x, y, w, h = target["bbox_xywh"]
            box = [v * scale for v in (x, y, x + w, y + h)]
            draw.rectangle(box, outline="#00d685", width=3)
            draw.text(
                (box[0] + 3, box[1] + 3),
                f"GT{index + 1} C{target['category_id']:02d}",
                fill="#003c24",
                stroke_width=2,
                stroke_fill="white",
                font=font,
            )
        for seg, item in zip(row["response"]["segmentations"], metrics["items"], strict=True):
            x, y = seg["bbox"]["x"], seg["bbox"]["y"]
            w, h = seg["bbox"]["width"], seg["bbox"]["height"]
            box = [v * scale for v in (x, y, x + w, y + h)]
            wrong = item["status"] == "APPROVED" and (
                item["target_class_id"] != item["predicted_class_id"]
            )
            color = "#e00027" if wrong else "#f59700" if item["status"] != "APPROVED" else "#4285ff"
            draw.rectangle(box, outline=color, width=5 if wrong else 2)
            label = (
                ("WRONG " if wrong else "") + item["status"] + " " + str(item["predicted_class_id"])
            )
            draw.text(
                (box[0] + 3, max(0, box[3] - 23)),
                label,
                fill=color,
                stroke_width=2,
                stroke_fill="white",
                font=font,
            )
        image.save(output / f"log-{row['image_id']:03d}.jpg", quality=92)
    report = load_json_config(measurement / "report.json")
    result = {
        "totals": dict(totals),
        "failure_images": issues,
        "repetitions": report["repetitions"],
        "annotation_policy": "Confirmed category and visible bbox retained unchanged.",
    }
    write_json(output / "diagnosis.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.manifest, args.measurement, args.output)
    print(result["totals"])


if __name__ == "__main__":
    main()
