from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from bixolon_scanner.training.models import require_torch


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tensor_rows(row: dict):
    torch = require_torch()
    return (
        torch.as_tensor(row["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4),
        torch.as_tensor(row["scores"], dtype=torch.float32),
    )


def _nms(boxes, scores, threshold: float):
    from torchvision.ops import nms

    if not len(boxes):
        return boxes, scores
    keep = nms(boxes, scores, threshold)
    return boxes[keep], scores[keep]


def _ground_truth(record: dict):
    torch = require_torch()
    return torch.as_tensor(
        [
            [
                row["bbox_xywh"][0],
                row["bbox_xywh"][1],
                row["bbox_xywh"][0] + row["bbox_xywh"][2],
                row["bbox_xywh"][1] + row["bbox_xywh"][3],
            ]
            for row in record["annotations"]
        ],
        dtype=torch.float32,
    ).reshape(-1, 4)


def _evaluate(records: list[dict], predictions: dict[int, tuple]) -> dict:
    from torchvision.ops import box_iou

    matched = 0
    false_positive = 0
    failure_images = []
    for record in records:
        image_id = int(record["image_id"])
        boxes, scores = predictions[image_id]
        truth = _ground_truth(record)
        overlaps = box_iou(boxes, truth) if len(boxes) and len(truth) else None
        unmatched_truth = set(range(len(truth)))
        unmatched_prediction = []
        for prediction_index in range(len(boxes)):
            if not unmatched_truth:
                unmatched_prediction.extend(range(prediction_index, len(boxes)))
                break
            best = max(
                unmatched_truth,
                key=lambda index: float(overlaps[prediction_index, index]),
            )
            if float(overlaps[prediction_index, best]) < 0.5:
                unmatched_prediction.append(prediction_index)
                continue
            unmatched_truth.remove(best)
            matched += 1
        false_positive += len(unmatched_prediction)
        if unmatched_truth or unmatched_prediction:
            failure_images.append(
                {
                    "image_id": image_id,
                    "false_negative_count": len(unmatched_truth),
                    "false_positive_count": len(unmatched_prediction),
                    "prediction_count": len(boxes),
                    "ground_truth_count": len(truth),
                }
            )
    ground_truth = sum(len(record["annotations"]) for record in records)
    return {
        "matched_count": matched,
        "false_negative_count": ground_truth - matched,
        "false_positive_count": false_positive,
        "total_error_count": ground_truth - matched + false_positive,
        "failure_image_count": len(failure_images),
        "failure_images": failure_images,
    }


def _fuse(
    primary_row: dict,
    secondary_row: dict,
    *,
    primary_threshold: float,
    secondary_threshold: float,
    agreement_iou: float,
    nms_threshold: float,
    mode: str,
):
    torch = require_torch()
    from torchvision.ops import box_iou

    primary_boxes, primary_scores = _tensor_rows(primary_row)
    secondary_boxes, secondary_scores = _tensor_rows(secondary_row)
    primary_mask = primary_scores >= primary_threshold
    secondary_mask = secondary_scores >= secondary_threshold
    primary_boxes, primary_scores = primary_boxes[primary_mask], primary_scores[primary_mask]
    secondary_boxes, secondary_scores = (
        secondary_boxes[secondary_mask],
        secondary_scores[secondary_mask],
    )
    primary_boxes, primary_scores = _nms(primary_boxes, primary_scores, nms_threshold)
    secondary_boxes, secondary_scores = _nms(secondary_boxes, secondary_scores, nms_threshold)

    if mode == "union":
        # Source confidence scales differ. Keep primary ordering, then allow a
        # secondary-only proposal to recover an object missed by the primary.
        mapped_secondary = 0.5 + 0.49 * secondary_scores
        boxes = torch.cat((primary_boxes, secondary_boxes))
        scores = torch.cat((primary_scores, mapped_secondary))
        return _nms(boxes, scores, nms_threshold)

    if not len(primary_boxes) or not len(secondary_boxes):
        if mode == "verified":
            return primary_boxes[:0], primary_scores[:0]
        return primary_boxes, primary_scores

    overlaps = box_iou(primary_boxes, secondary_boxes)
    best_iou, best_index = overlaps.max(dim=1)
    agreed = best_iou >= agreement_iou
    if mode == "verified":
        return primary_boxes[agreed], primary_scores[agreed]
    if mode == "replace":
        output = primary_boxes.clone()
        output[agreed] = secondary_boxes[best_index[agreed]]
        return _nms(output, primary_scores, nms_threshold)
    if mode == "average":
        output = primary_boxes.clone()
        output[agreed] = (primary_boxes[agreed] + secondary_boxes[best_index[agreed]]) / 2
        return _nms(output, primary_scores, nms_threshold)
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fresh detector fusion")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--secondary-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _read_jsonl(args.manifest)
    primary = {int(row["image_id"]): row for row in _read_jsonl(args.primary_predictions)}
    secondary = {int(row["image_id"]): row for row in _read_jsonl(args.secondary_predictions)}
    candidates = []
    grid = itertools.product(
        ("union", "verified", "replace", "average"),
        (0.30, 0.34, 0.40, 0.50, 0.60, 0.62, 0.69, 0.735, 0.80, 0.85, 0.90),
        (0.05, 0.08, 0.10, 0.12, 0.20, 0.30, 0.40, 0.50, 0.60),
        (0.30, 0.40, 0.50, 0.60, 0.70),
    )
    for mode, primary_threshold, secondary_threshold, agreement_iou in grid:
        predictions = {
            int(record["image_id"]): _fuse(
                primary[int(record["image_id"])],
                secondary[int(record["image_id"])],
                primary_threshold=primary_threshold,
                secondary_threshold=secondary_threshold,
                agreement_iou=agreement_iou,
                nms_threshold=0.4,
                mode=mode,
            )
            for record in records
        }
        metrics = _evaluate(records, predictions)
        candidates.append(
            {
                "mode": mode,
                "primary_threshold": primary_threshold,
                "secondary_threshold": secondary_threshold,
                "agreement_iou": agreement_iou,
                **metrics,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["total_error_count"],
            row["false_positive_count"],
            row["false_negative_count"],
        )
    )
    # Diagnostic upper bound only: GT object counts are never used as a deployable
    # rule. This determines whether a separately trained exact-count head could
    # in principle recover every object from the available fresh proposals.
    oracle_count_candidates = []
    for source, nms_threshold, secondary_scale in itertools.product(
        ("primary", "union"),
        (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        (0.25, 0.5, 0.75, 1.0),
    ):
        predictions = {}
        for record in records:
            image_id = int(record["image_id"])
            primary_boxes, primary_scores = _tensor_rows(primary[image_id])
            if source == "union":
                secondary_boxes, secondary_scores = _tensor_rows(secondary[image_id])
                boxes = require_torch().cat((primary_boxes, secondary_boxes))
                scores = require_torch().cat((primary_scores, secondary_scores * secondary_scale))
            else:
                boxes, scores = primary_boxes, primary_scores
            boxes, scores = _nms(boxes, scores, nms_threshold)
            count = len(record["annotations"])
            predictions[image_id] = (boxes[:count], scores[:count])
        metrics = _evaluate(records, predictions)
        oracle_count_candidates.append(
            {
                "source": source,
                "nms_threshold": nms_threshold,
                "secondary_scale": secondary_scale,
                **metrics,
            }
        )
    oracle_count_candidates.sort(
        key=lambda row: (
            row["total_error_count"],
            row["false_positive_count"],
            row["false_negative_count"],
        )
    )
    report = {
        "schema_version": "1.0",
        "candidate_count": len(candidates),
        "best_candidates": candidates[:50],
        "oracle_count_is_diagnostic_only": True,
        "oracle_count_best_candidates": oracle_count_candidates[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["best_candidates"][:20], indent=2))
    print(json.dumps(report["oracle_count_best_candidates"][:10], indent=2))


if __name__ == "__main__":
    main()
