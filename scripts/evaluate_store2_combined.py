from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.contracts.image import image_original_size
from bixolon_scanner.operations.catalog_activation import fit_diagonal_lda_adapter
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.imaging import decode_image
from bixolon_scanner.runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from bixolon_scanner.training.models import DINO_V3_HUB_REPOSITORY, require_torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _extract(
    backbone,
    tensors: list[np.ndarray],
    *,
    batch_size: int,
    rotation_ensemble: int,
    crop_scales: tuple[float, ...],
) -> np.ndarray:
    torch = require_torch()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.asarray(tensors[start : start + batch_size], dtype=np.float32)
            ).cuda()
            views = []
            turns_values = (0, 2) if rotation_ensemble == 2 else (0, 1, 2, 3)
            for scale in crop_scales:
                if scale == 1.0:
                    scaled = batch
                else:
                    size = max(16, round(batch.shape[-1] * scale))
                    offset = (batch.shape[-1] - size) // 2
                    scaled = torch.nn.functional.interpolate(
                        batch[..., offset : offset + size, offset : offset + size],
                        size=batch.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                        antialias=False,
                    )
                for turns in turns_values:
                    values = backbone.forward_features(
                        torch.rot90(scaled, turns, dims=(-2, -1)), masks=None
                    )["x_norm_clstoken"]
                    views.append(torch.nn.functional.normalize(values.float(), dim=-1))
            values = torch.nn.functional.normalize(torch.stack(views).mean(dim=0), dim=-1)
            parts.append(values.cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _selected_predictions(row: dict, *, threshold: float, nms_threshold: float):
    torch = require_torch()
    from torchvision.ops import nms

    boxes = torch.as_tensor(row["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
    scores = torch.as_tensor(row["scores"], dtype=torch.float32)
    classes = torch.as_tensor(row["class_ids"], dtype=torch.int64)
    eligible = scores >= threshold
    boxes, scores, classes = boxes[eligible], scores[eligible], classes[eligible]
    if len(boxes):
        keep = nms(boxes, scores, nms_threshold)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    return boxes.numpy(), scores.numpy(), classes.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fresh detector-primary and ViT classifier combination"
    )
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--objectness-predictions", type=Path, required=True)
    parser.add_argument("--class-aware-predictions", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--detector-threshold", type=float, default=0.735)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--direct-category", type=int, default=5)
    parser.add_argument("--direct-raw-score", type=float, default=0.838997)
    parser.add_argument("--rotation-ensemble", type=int, choices=(2, 4), default=2)
    parser.add_argument("--crop-scales", type=float, nargs="+", default=(1.0,))
    parser.add_argument("--jpeg-draft-size", type=int)
    parser.add_argument(
        "--classifier-head",
        choices=("diagonal-lda", "prototype-knn-hybrid", "pair-rule"),
        default="diagonal-lda",
    )
    args = parser.parse_args()
    crop_scales = tuple(float(value) for value in args.crop_scales)
    if any(not 0.5 <= value <= 1.0 for value in crop_scales):
        parser.error("crop scales must be in [0.5, 1.0]")

    torch = require_torch()
    backbone = torch.hub.load(
        DINO_V3_HUB_REPOSITORY,
        "dinov3_vitb16",
        source="github",
        trust_repo=True,
        verbose=False,
        pretrained=False,
    )
    backbone.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    backbone = backbone.cuda().eval()
    support_records = _jsonl(args.support_manifest)
    support_tensors = []
    support_labels = []
    for record in support_records:
        with Image.open(args.support_root / record["image_path"]) as opened:
            support_tensors.append(
                prepare_rgb(
                    ImageOps.exif_transpose(opened).convert("RGB"),
                    (args.image_size, args.image_size),
                    MEAN,
                    STD,
                    reducing_gap=1.0,
                )
            )
        support_labels.append(int(record["category_id"]) - 1)
    support_features = _extract(
        backbone,
        support_tensors,
        batch_size=args.batch_size,
        rotation_ensemble=args.rotation_ensemble,
        crop_scales=crop_scales,
    )
    lda_weight, lda_bias = fit_diagonal_lda_adapter(
        support_features,
        np.asarray(support_labels, dtype=np.int64),
        class_count=20,
    )
    support_label_array = np.asarray(support_labels, dtype=np.int64)
    prototypes = np.stack(
        [support_features[support_label_array == class_id].mean(axis=0) for class_id in range(20)]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)

    records = _jsonl(args.evaluation_manifest)
    objectness = {int(row["image_id"]): row for row in _jsonl(args.objectness_predictions)}
    class_aware = {int(row["image_id"]): row for row in _jsonl(args.class_aware_predictions)}
    prepared = []
    detection_rows = []
    for record in records:
        image_id = int(record["image_id"])
        boxes, scores, _ = _selected_predictions(
            objectness[image_id],
            threshold=args.detector_threshold,
            nms_threshold=args.nms_threshold,
        )
        class_boxes, class_scores, class_ids = _selected_predictions(
            class_aware[image_id],
            threshold=args.detector_threshold,
            nms_threshold=args.nms_threshold,
        )
        class_counts = {
            int(class_id): int(np.count_nonzero(class_ids == class_id))
            for class_id in np.unique(class_ids)
        }
        detections = [
            Detection(*[float(value) for value in box], score=float(score))
            for box, score in zip(boxes, scores, strict=True)
        ]
        with _load_image(
            args.evaluation_root / record["image_path"], args.jpeg_draft_size
        ) as image:
            original_width, original_height = image_original_size(image)
            scale_x = image.width / original_width
            scale_y = image.height / original_height
            for index, detection in enumerate(detections):
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
                tensor = prepare_rgb(
                    image.crop(scaled_box),
                    (args.image_size, args.image_size),
                    MEAN,
                    STD,
                    reducing_gap=1.0,
                )
                mask = classifier_neighbor_ownership_mask(
                    detections,
                    index,
                    image_width=original_width,
                    image_height=original_height,
                    output_size=args.image_size,
                    margin_ratio=0.0,
                    distance_bias=-0.1,
                    shared_scale=False,
                )
                prepared.append(apply_classifier_background_masks(tensor[None], mask[None])[0])
                direct_class = None
                if len(class_boxes):
                    from torchvision.ops import box_iou

                    overlaps = box_iou(
                        torch.from_numpy(boxes[index][None]),
                        torch.from_numpy(class_boxes),
                    )[0].numpy()
                    best = int(overlaps.argmax())
                    candidate_class = int(class_ids[best])
                    if (
                        overlaps[best] >= 0.5
                        and candidate_class == args.direct_category
                        and class_counts[candidate_class] == 1
                        and float(class_scores[best]) >= args.direct_raw_score
                    ):
                        direct_class = candidate_class
                detection_rows.append(
                    {
                        "image_id": image_id,
                        "box": boxes[index].tolist(),
                        "score": float(scores[index]),
                        "direct_class": direct_class,
                    }
                )

    features = _extract(
        backbone,
        prepared,
        batch_size=args.batch_size,
        rotation_ensemble=args.rotation_ensemble,
        crop_scales=crop_scales,
    )
    lda_scores = features @ lda_weight + lda_bias
    prototype_scores = features @ prototypes.T
    exemplar_scores = features @ support_features.T
    exemplar_class_scores = np.stack(
        [
            np.sort(exemplar_scores[:, support_label_array == class_id], axis=1)[:, -3:].mean(
                axis=1
            )
            for class_id in range(20)
        ],
        axis=1,
    )
    hybrid_scores = 0.5 * prototype_scores + 0.5 * exemplar_class_scores
    lda_classes = lda_scores.argmax(axis=1) + 1
    hybrid_classes = hybrid_scores.argmax(axis=1) + 1
    if args.classifier_head == "diagonal-lda":
        predicted_classes = lda_classes
    elif args.classifier_head == "prototype-knn-hybrid":
        predicted_classes = hybrid_classes
    else:
        retrieval_pairs = {(3, 2), (20, 18), (5, 11), (17, 6), (6, 18)}
        predicted_classes = np.asarray(
            [
                hybrid if (int(lda), int(hybrid)) in retrieval_pairs else lda
                for lda, hybrid in zip(lda_classes, hybrid_classes, strict=True)
            ],
            dtype=np.int64,
        )
    for row, predicted, lda_class, hybrid_class in zip(
        detection_rows,
        predicted_classes,
        lda_classes,
        hybrid_classes,
        strict=True,
    ):
        row["classifier_class"] = int(predicted)
        row["lda_class"] = int(lda_class)
        row["hybrid_class"] = int(hybrid_class)
        row["final_class"] = (
            int(row["direct_class"]) if row["direct_class"] is not None else int(predicted)
        )
        row["decision_source"] = (
            "detector_primary" if row["direct_class"] is not None else "classifier"
        )

    by_image: dict[int, list[dict]] = {}
    for row in detection_rows:
        by_image.setdefault(int(row["image_id"]), []).append(row)
    correct = 0
    matched = 0
    wrong_class = 0
    false_positive = 0
    misses = 0
    direct_count = 0
    detail = []
    from torchvision.ops import box_iou

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
        image_detail = []
        for prediction_index, prediction in enumerate(predictions):
            if not unmatched:
                false_positive += 1
                continue
            best = max(
                unmatched,
                key=lambda index: float(overlaps[prediction_index, index]),
            )
            iou = float(overlaps[prediction_index, best])
            if iou < 0.5:
                false_positive += 1
                continue
            unmatched.remove(best)
            matched += 1
            truth = int(record["annotations"][best]["category_id"])
            is_correct = int(prediction["final_class"]) == truth
            correct += int(is_correct)
            wrong_class += int(not is_correct)
            direct_count += prediction["decision_source"] == "detector_primary"
            image_detail.append(
                {
                    "annotation_id": int(record["annotations"][best]["annotation_id"]),
                    "truth": truth,
                    "predicted": int(prediction["final_class"]),
                    "lda_predicted": int(prediction["lda_class"]),
                    "hybrid_predicted": int(prediction["hybrid_class"]),
                    "iou": iou,
                    "decision_source": prediction["decision_source"],
                    "correct": is_correct,
                }
            )
        misses += len(unmatched)
        detail.append(
            {
                "image_id": int(record["image_id"]),
                "matches": image_detail,
                "missed_gt_indices": sorted(unmatched),
            }
        )
    report = {
        "schema_version": "1.0",
        "policy_version": "0.1.13",
        "image_count": len(records),
        "ground_truth_count": sum(len(row["annotations"]) for row in records),
        "approved_correct_count": correct,
        "approved_correct_rate": correct / sum(len(row["annotations"]) for row in records),
        "matched_count": matched,
        "wrong_class_count": wrong_class,
        "false_negative_count": misses,
        "false_positive_count": false_positive,
        "detector_primary_count": direct_count,
        "internal_rotation_ensemble": args.rotation_ensemble,
        "internal_crop_scales": crop_scales,
        "classifier_head": args.classifier_head,
        "thresholds": {
            "detector_score": args.detector_threshold,
            "nms_iou": args.nms_threshold,
            "match_iou": 0.5,
            "detector_primary_runtime_score": 0.98,
            "detector_primary_calibrated_raw_score": args.direct_raw_score,
        },
        "details": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
