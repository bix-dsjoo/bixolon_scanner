from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from bixolon_scanner.contracts.image import image_original_size
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.imaging import decode_image
from bixolon_scanner.runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from bixolon_scanner.training.models import require_torch

DETECTOR_MEAN = (0.5, 0.5, 0.5)
DETECTOR_STD = (0.5, 0.5, 0.5)
EMBEDDER_MEAN = (0.485, 0.456, 0.406)
EMBEDDER_STD = (0.229, 0.224, 0.225)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


@contextmanager
def _load_image(path: Path, jpeg_draft_size: int | None):
    if jpeg_draft_size is None:
        with Image.open(path) as opened:
            yield ImageOps.exif_transpose(opened).convert("RGB")
        return
    image = decode_image(
        path.read_bytes(),
        max_bytes=50_000_000,
        max_pixels=50_000_000,
        jpeg_draft_size=jpeg_draft_size,
    )
    try:
        yield image
    finally:
        image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the exported fresh ONNX candidate with the 0.1.13 ROI policy"
    )
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--detector-predictions", type=Path)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-predictions-output", type=Path)
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--embedder-size", type=int, default=192)
    parser.add_argument("--embedder-batch-size", type=int, default=16)
    parser.add_argument("--detector-threshold", type=float, default=0.735)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--jpeg-draft-size", type=int)
    args = parser.parse_args()

    torch = require_torch()
    from torchvision.ops import box_iou, nms

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    detector = (
        ort.InferenceSession(
            str(args.detector), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        if args.detector_predictions is None
        else None
    )
    embedder = ort.InferenceSession(
        str(args.embedder), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    records = _jsonl(args.evaluation_manifest)
    external_predictions = (
        {}
        if args.detector_predictions is None
        else {int(row["image_id"]): row for row in _jsonl(args.detector_predictions)}
    )
    prepared: list[np.ndarray] = []
    detection_rows: list[dict] = []
    for record in records:
        image_id = int(record["image_id"])
        with _load_image(
            args.evaluation_root / record["image_path"], args.jpeg_draft_size
        ) as image:
            original_width, original_height = image_original_size(image)
            scale_x = image.width / original_width
            scale_y = image.height / original_height
            if detector is None:
                prediction = external_predictions[image_id]
                boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
                scores = np.asarray(prediction["scores"], dtype=np.float32)
            else:
                tensor = prepare_rgb(
                    image,
                    (args.detector_size, args.detector_size),
                    DETECTOR_MEAN,
                    DETECTOR_STD,
                    reducing_gap=1.0,
                )[None]
                logits, normalized_boxes = detector.run(None, {"pixel_values": tensor})
                scores = _sigmoid(np.asarray(logits[0])).max(axis=-1)
                boxes = np.asarray(normalized_boxes[0], dtype=np.float32)
                center_x, center_y, width, height = boxes.T
                boxes = np.stack(
                    (
                        (center_x - width / 2.0) * original_width,
                        (center_y - height / 2.0) * original_height,
                        (center_x + width / 2.0) * original_width,
                        (center_y + height / 2.0) * original_height,
                    ),
                    axis=1,
                )
                boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0.0, float(original_width))
                boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0.0, float(original_height))
            eligible = scores >= args.detector_threshold
            boxes, scores = boxes[eligible], scores[eligible]
            if len(boxes):
                keep = nms(
                    torch.from_numpy(boxes),
                    torch.from_numpy(scores),
                    args.nms_threshold,
                ).numpy()
                boxes, scores = boxes[keep], scores[keep]
            valid = []
            for box, score in zip(boxes, scores, strict=True):
                pixel_width = float(box[2] - box[0])
                pixel_height = float(box[3] - box[1])
                if (
                    pixel_width > 0
                    and pixel_height > 0
                    and max(pixel_width / pixel_height, pixel_height / pixel_width) <= 5.0
                ):
                    valid.append((box, float(score)))
            detections = [
                Detection(*[float(value) for value in box], score=score) for box, score in valid
            ]
            for index, detection in enumerate(detections):
                if args.detector_only:
                    detection_rows.append(
                        {
                            "image_id": image_id,
                            "box": [detection.x1, detection.y1, detection.x2, detection.y2],
                            "score": float(detection.score),
                        }
                    )
                    continue
                crop_box = classifier_crop_box(
                    detection,
                    original_width,
                    original_height,
                    margin_ratio=0.0,
                    crop_mode="box_resize",
                )
                scaled_box = (
                    int(np.floor(crop_box[0] * scale_x)),
                    int(np.floor(crop_box[1] * scale_y)),
                    int(np.ceil(crop_box[2] * scale_x)),
                    int(np.ceil(crop_box[3] * scale_y)),
                )
                crop = prepare_rgb(
                    image.crop(scaled_box),
                    (args.embedder_size, args.embedder_size),
                    EMBEDDER_MEAN,
                    EMBEDDER_STD,
                    reducing_gap=1.0,
                )
                mask = classifier_neighbor_ownership_mask(
                    detections,
                    index,
                    image_width=original_width,
                    image_height=original_height,
                    output_size=args.embedder_size,
                    margin_ratio=0.0,
                    distance_bias=-0.1,
                    shared_scale=False,
                )
                prepared.append(apply_classifier_background_masks(crop[None], mask[None])[0])
                detection_rows.append(
                    {
                        "image_id": image_id,
                        "box": [detection.x1, detection.y1, detection.x2, detection.y2],
                        "score": float(detection.score),
                    }
                )

    if args.detector_predictions_output is not None:
        rows_by_image: dict[int, list[dict]] = {}
        for row in detection_rows:
            rows_by_image.setdefault(int(row["image_id"]), []).append(row)
        args.detector_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.detector_predictions_output.write_text(
            "".join(
                json.dumps(
                    {
                        "image_id": int(record["image_id"]),
                        "boxes_xyxy": [
                            row["box"] for row in rows_by_image.get(int(record["image_id"]), [])
                        ],
                        "scores": [
                            row["score"] for row in rows_by_image.get(int(record["image_id"]), [])
                        ],
                        "class_ids": [0 for _ in rows_by_image.get(int(record["image_id"]), [])],
                    },
                    separators=(",", ":"),
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    if args.detector_only:
        print(
            json.dumps(
                {
                    "image_count": len(records),
                    "selected_detection_count": len(detection_rows),
                    "predictions": str(args.detector_predictions_output),
                },
                indent=2,
            )
        )
        return

    embedding_parts = []
    for start in range(0, len(prepared), args.embedder_batch_size):
        batch = np.asarray(prepared[start : start + args.embedder_batch_size], dtype=np.float32)
        embedding_parts.append(embedder.run(None, {"pixel_values": batch})[0])
    embeddings = np.concatenate(embedding_parts).astype(np.float32)
    if embeddings.shape != (len(detection_rows), 20):
        raise ValueError(f"unexpected embedding shape: {embeddings.shape}")
    selected = embeddings.argmax(axis=1)
    approved = (
        np.isclose(embeddings.max(axis=1), 1.0, atol=1e-6)
        & np.isclose(embeddings.sum(axis=1), 1.0, atol=1e-6)
        & (np.count_nonzero(embeddings > 0.5, axis=1) == 1)
    )
    if not bool(approved.all()):
        raise ValueError("exported embedder did not produce deterministic approved one-hot outputs")
    for row, predicted in zip(detection_rows, selected + 1, strict=True):
        row["predicted"] = int(predicted)

    by_image: dict[int, list[dict]] = {}
    for row in detection_rows:
        by_image.setdefault(int(row["image_id"]), []).append(row)
    correct = 0
    matched = 0
    wrong_class = 0
    false_positive = 0
    misses = 0
    details = []
    for record in records:
        predictions = sorted(
            by_image.get(int(record["image_id"]), []),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        gt_boxes = torch.as_tensor(
            [
                [
                    annotation["bbox_xywh"][0],
                    annotation["bbox_xywh"][1],
                    annotation["bbox_xywh"][0] + annotation["bbox_xywh"][2],
                    annotation["bbox_xywh"][1] + annotation["bbox_xywh"][3],
                ]
                for annotation in record["annotations"]
            ],
            dtype=torch.float32,
        )
        prediction_boxes = torch.as_tensor(
            [row["box"] for row in predictions], dtype=torch.float32
        ).reshape(-1, 4)
        overlaps = (
            box_iou(prediction_boxes, gt_boxes) if len(prediction_boxes) and len(gt_boxes) else None
        )
        unmatched = set(range(len(record["annotations"])))
        image_matches = []
        for prediction_index, prediction in enumerate(predictions):
            if not unmatched:
                false_positive += 1
                continue
            best = max(unmatched, key=lambda index: float(overlaps[prediction_index, index]))
            iou = float(overlaps[prediction_index, best])
            if iou < 0.5:
                false_positive += 1
                continue
            unmatched.remove(best)
            matched += 1
            truth = int(record["annotations"][best]["category_id"])
            is_correct = int(prediction["predicted"]) == truth
            correct += int(is_correct)
            wrong_class += int(not is_correct)
            image_matches.append(
                {
                    "annotation_id": int(record["annotations"][best]["annotation_id"]),
                    "truth": truth,
                    "predicted": int(prediction["predicted"]),
                    "iou": iou,
                    "correct": is_correct,
                }
            )
        misses += len(unmatched)
        details.append(
            {
                "image_id": int(record["image_id"]),
                "matches": image_matches,
                "missed_gt_indices": sorted(unmatched),
            }
        )
    ground_truth_count = sum(len(record["annotations"]) for record in records)
    report = {
        "schema_version": "1.0",
        "policy_version": "0.1.13",
        "execution": "onnxruntime-cpu",
        "image_count": len(records),
        "ground_truth_count": ground_truth_count,
        "approved_correct_count": correct,
        "approved_correct_rate": correct / ground_truth_count,
        "matched_count": matched,
        "wrong_class_count": wrong_class,
        "false_negative_count": misses,
        "false_positive_count": false_positive,
        "detector_primary_count": 0,
        "classifier_output_contract": "one_hot_catalog_embedding",
        "thresholds": {
            "detector_score": args.detector_threshold,
            "nms_iou": args.nms_threshold,
            "match_iou": 0.5,
            "approval_minimum_similarity": 1.0,
            "approval_minimum_margin": 0.1,
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    if report["approved_correct_rate"] < 0.99:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
