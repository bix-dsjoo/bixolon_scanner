from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def box_area_ratio(box: list[float], *, image_width: int, image_height: int) -> float:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    image_area = image_width * image_height
    return width * height / image_area if image_area else 0.0


def reject_oversized_predictions(
    predictions: list[dict[str, Any]],
    dimensions: dict[int, tuple[int, int]],
    *,
    maximum_box_area_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < maximum_box_area_ratio <= 1.0:
        raise ValueError("maximum box area ratio must be in (0, 1]")
    outputs = []
    diagnostics = []
    for row in predictions:
        image_id = int(row["image_id"])
        image_width, image_height = dimensions[image_id]
        keep = []
        removed = []
        for index, box in enumerate(row["boxes_xyxy"]):
            ratio = box_area_ratio(box, image_width=image_width, image_height=image_height)
            if ratio <= maximum_box_area_ratio:
                keep.append(index)
            else:
                removed.append({"prediction_index": index, "box_area_ratio": ratio})
        output = {
            key: (
                [value[index] for index in keep]
                if isinstance(value, list) and len(value) == len(row["boxes_xyxy"])
                else value
            )
            for key, value in row.items()
        }
        outputs.append(output)
        if removed:
            diagnostics.append({"image_id": image_id, "removed": removed})
    return outputs, diagnostics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dimensions_from_manifest(path: Path) -> dict[int, tuple[int, int]]:
    return {
        int(row["image_id"]): (int(row["width"]), int(row["height"]))
        for row in _read_jsonl(path)
        if row.get("record_type") == "detection"
    }


def _dimensions_from_coco(path: Path) -> dict[int, tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {int(row["id"]): (int(row["width"]), int(row["height"])) for row in payload["images"]}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    predictions = _read_jsonl(args.predictions)
    dimensions = (
        _dimensions_from_manifest(args.manifest)
        if args.manifest is not None
        else _dimensions_from_coco(args.coco)
    )
    outputs, diagnostics = reject_oversized_predictions(
        predictions,
        dimensions,
        maximum_box_area_ratio=args.maximum_box_area_ratio,
    )
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outputs),
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_oversized_frame_rejection",
        "selection_scope": "rejected locked test is development data for the successor",
        "maximum_box_area_ratio": args.maximum_box_area_ratio,
        "image_count": len(outputs),
        "input_prediction_count": sum(len(row["boxes_xyxy"]) for row in predictions),
        "output_prediction_count": sum(len(row["boxes_xyxy"]) for row in outputs),
        "removed_prediction_count": sum(len(row["removed"]) for row in diagnostics),
        "images_without_predictions": [
            int(row["image_id"]) for row in outputs if not row["boxes_xyxy"]
        ],
        "diagnostics": diagnostics,
        "locked_test_status": "consumed_by_rejected_v2_and_reclassified_as_development",
        "promotion_evidence": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject tray or frame-sized detector boxes")
    parser.add_argument("--predictions", type=Path, required=True)
    dimensions = parser.add_mutually_exclusive_group(required=True)
    dimensions.add_argument("--manifest", type=Path)
    dimensions.add_argument("--coco", type=Path)
    parser.add_argument("--maximum-box-area-ratio", type=float, default=0.3)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
