from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .calibration import softmax
from .rpc_worker_gate import _iou, postprocess_worker_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.output_dir / "runs" / "full" / f"seed{config['experiment']['seeds'][0]}"
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    archive = np.load(run_dir / "selection_predictions.npz")
    classifier_probabilities = softmax(archive["logits"], float(calibration["temperature"]))
    context_report = json.loads(
        (run_dir / "context-rejector" / "report.json").read_text(encoding="utf-8")
    )["models"]["logistic"]["policy"]
    context_scores = np.load(run_dir / "context-rejector" / "logistic_scores.npz")["selection"]
    approved_unmatched_ids = set(
        archive["sample_ids"][
            (archive["targets"] < 0)
            & (
                classifier_probabilities.max(axis=1)
                >= float(context_report["classifier_threshold"])
            )
            & (context_scores >= float(context_report["quality_threshold"]))
        ]
        .astype(str)
        .tolist()
    )
    detector_dir = args.output_dir / "detector"
    threshold = json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))[
        "selected_score_threshold"
    ]
    options = dict(config["detector"], score_threshold=threshold)
    records = [
        json.loads(line)
        for line in (detector_dir / "manifest" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions = {
        row["sample_key"]: row
        for row in (
            json.loads(line)
            for line in (detector_dir / "predictions" / "val_oof.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    rows = []
    for record in records:
        if record["role"] != "selection":
            continue
        result = postprocess_worker_gate(
            record,
            predictions[f"{record['source']}:{record['image_id']}"],
            options,
        )
        detections = result["detections"]
        for index in result["unmatched_detection_indices"]:
            box = detections[int(index)]["bbox_xyxy"]
            overlaps = [
                _iou(box, detection["bbox_xyxy"])
                for other, detection in enumerate(detections)
                if other != int(index)
            ]
            rows.append(
                {
                    "level": record["level"],
                    "score": float(detections[int(index)]["score"]),
                    "max_detection_iou": max(overlaps, default=0.0),
                    "detection_count": len(detections),
                    "classifier_approved": (
                        f"val:{record['image_id']}:det{index}" in approved_unmatched_ids
                    ),
                    "sample_id": f"val:{record['image_id']}:det{index}",
                    "image_path": str(record["image_path"]),
                    "bbox_xyxy": [float(value) for value in box],
                    "ground_truth_boxes": [
                        [
                            float(annotation["bbox_xywh"][0]),
                            float(annotation["bbox_xywh"][1]),
                            float(annotation["bbox_xywh"][0]) + float(annotation["bbox_xywh"][2]),
                            float(annotation["bbox_xywh"][1]) + float(annotation["bbox_xywh"][3]),
                        ]
                        for annotation in record["annotations"]
                    ],
                }
            )
    overlap = np.asarray([row["max_detection_iou"] for row in rows])
    score = np.asarray([row["score"] for row in rows])
    report = {
        "count": len(rows),
        "score_quantiles": np.quantile(score, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "overlap_quantiles": np.quantile(overlap, [0, 0.1, 0.5, 0.9, 0.95, 0.99, 1]).tolist(),
        "overlap_threshold_counts": {
            str(value): int((overlap >= value).sum())
            for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        },
        "by_level": {
            level: sum(row["level"] == level for row in rows)
            for level in ("easy", "medium", "hard")
        },
        "classifier_approved": {},
    }
    approved = [row for row in rows if row["classifier_approved"]]
    approved_overlap = np.asarray([row["max_detection_iou"] for row in approved])
    report["classifier_approved"] = {
        "count": len(approved),
        "overlap_quantiles": np.quantile(
            approved_overlap, [0, 0.1, 0.5, 0.9, 0.95, 0.99, 1]
        ).tolist(),
        "overlap_threshold_counts": {
            str(value): int((approved_overlap >= value).sum())
            for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        },
    }
    if args.contact_sheet is not None:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required with --contact-sheet")
        selected = sorted(
            approved,
            key=lambda row: (float(row["score"]), row["sample_id"]),
            reverse=True,
        )[:20]
        cell_width, cell_height = 480, 390
        sheet = Image.new("RGB", (cell_width * 4, cell_height * 5), "white")
        for position, row in enumerate(selected):
            with Image.open(args.dataset_root / row["image_path"]) as source:
                image = source.convert("RGB")
            scale = min(cell_width / image.width, 350 / image.height)
            resized = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.BILINEAR,
            )
            draw = ImageDraw.Draw(resized)
            for box in row["ground_truth_boxes"]:
                draw.rectangle([value * scale for value in box], outline="lime", width=3)
            draw.rectangle([value * scale for value in row["bbox_xyxy"]], outline="red", width=5)
            cell = Image.new("RGB", (cell_width, cell_height), "white")
            cell.paste(resized, ((cell_width - resized.width) // 2, 0))
            label = ImageDraw.Draw(cell)
            label.text(
                (8, 355),
                f"{row['sample_id']} score={row['score']:.3f} overlap={row['max_detection_iou']:.3f}",
                fill="black",
            )
            sheet.paste(cell, ((position % 4) * cell_width, (position // 4) * cell_height))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet, quality=92)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
