from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..pipeline.ports import Detection
from ..runtime.imaging import decode_image, image_original_size
from ..runtime.onnx import OrtRunner, nms, prepare_rgb, sigmoid
from .bread_dataset_identity import resolve_coco_image
from .detector import (
    _allowed_aspect_ratio,
    _iou,
    _metrics_grid,
    _xywh_to_xyxy,
    select_release_threshold_candidate,
)


def raw_outputs_to_prediction(
    logits: np.ndarray,
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, list[Any]]:
    """Collapse class logits to object scores and convert normalized boxes to pixels."""
    values = np.asarray(logits, dtype=np.float32)
    normalized_boxes = np.asarray(boxes, dtype=np.float32)
    if values.ndim == 1:
        scores = sigmoid(values)
        ranks = np.zeros((len(values), 1), dtype=np.int64)
    elif values.ndim == 2:
        scores = sigmoid(values).max(axis=-1)
        ranks = np.argsort(-values, axis=-1, kind="stable")
    else:
        raise ValueError("detector logits must have shape [queries] or [queries, classes]")
    if normalized_boxes.shape != (len(scores), 4):
        raise ValueError("detector boxes must have shape [queries, 4]")

    converted_boxes: list[list[float]] = []
    converted_scores: list[float] = []
    converted_class_ids: list[int] = []
    converted_top3: list[list[int]] = []
    for score, box, rank in zip(scores, normalized_boxes, ranks):
        cx, cy, width, height = [float(value) for value in box]
        x1 = max(0.0, (cx - width * 0.5) * image_width)
        y1 = max(0.0, (cy - height * 0.5) * image_height)
        x2 = min(float(image_width), (cx + width * 0.5) * image_width)
        y2 = min(float(image_height), (cy + height * 0.5) * image_height)
        if x2 > x1 and y2 > y1:
            converted_boxes.append([x1, y1, x2, y2])
            converted_scores.append(float(score))
            converted_class_ids.append(int(rank[0]))
            converted_top3.append([int(value) for value in rank[:3]])
    return {
        "boxes_xyxy": converted_boxes,
        "scores": converted_scores,
        "class_ids": converted_class_ids,
        "top3_class_ids": converted_top3,
    }


def _restore_rotated_boxes(
    boxes: list[list[float]], *, turns: int, original_width: int, original_height: int
) -> list[list[float]]:
    if turns == 0:
        return boxes
    restored = []
    for x1, y1, x2, y2 in boxes:
        if turns == 1:
            box = [original_width - y2, x1, original_width - y1, x2]
        elif turns == 2:
            box = [
                original_width - x2,
                original_height - y2,
                original_width - x1,
                original_height - y1,
            ]
        elif turns == 3:
            box = [y1, original_height - x2, y2, original_height - x1]
        else:
            raise ValueError("rotation turns must be between zero and three")
        restored.append([float(value) for value in box])
    return restored


def _filter_rotation_support(
    prediction: dict[str, list[Any]],
    view_ids: list[int],
    *,
    minimum_support: int,
    iou_threshold: float,
    minimum_score: float = 0.0,
) -> dict[str, list[Any]]:
    eligible = [
        index for index, score in enumerate(prediction["scores"]) if float(score) >= minimum_score
    ]
    if minimum_support <= 1:
        return {key: [values[index] for index in eligible] for key, values in prediction.items()}
    boxes = {
        index: np.asarray(prediction["boxes_xyxy"][index], dtype=np.float32) for index in eligible
    }
    keep = []
    for index, box in boxes.items():
        supporting_views = {
            view_ids[other]
            for other, candidate in boxes.items()
            if view_ids[other] != view_ids[index] and _iou(box, candidate) >= iou_threshold
        }
        supporting_views.add(view_ids[index])
        if len(supporting_views) >= minimum_support:
            keep.append(index)
    return {key: [values[index] for index in keep] for key, values in prediction.items()}


def _preselect_view_prediction(
    prediction: dict[str, list[Any]],
    *,
    minimum_score: float,
    nms_iou_threshold: float,
    max_object_aspect_ratio: float | None,
    maximum_candidates: int | None = None,
) -> dict[str, list[Any]]:
    candidates = [
        (index, Detection(*box, score))
        for index, (box, score) in enumerate(
            zip(prediction["boxes_xyxy"], prediction["scores"], strict=True)
        )
        if score >= minimum_score and _allowed_aspect_ratio(box, max_object_aspect_ratio)
    ]
    if maximum_candidates:
        candidates = sorted(candidates, key=lambda item: item[1].score, reverse=True)[
            :maximum_candidates
        ]
    index_by_detection = {detection: index for index, detection in candidates}
    keep = [
        index_by_detection[detection]
        for detection in nms(
            [detection for _, detection in candidates],
            nms_iou_threshold,
        )
    ]
    return {key: [values[index] for index in keep] for key, values in prediction.items()}


def _fuse_rotation_predictions(
    prediction: dict[str, list[Any]],
    view_ids: list[int],
    *,
    view_count: int,
    minimum_support: int,
    iou_threshold: float,
    score_mode: str,
) -> dict[str, list[Any]]:
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    assigned: set[int] = set()
    fused = {key: [] for key in prediction}
    fused["rotation_support_count"] = []
    fused["rotation_score_standard_deviation"] = []
    fused["rotation_box_dispersion"] = []
    for seed_value in np.argsort(-scores, kind="stable"):
        seed = int(seed_value)
        if seed in assigned:
            continue
        members = [seed]
        assigned.add(seed)
        for view_id in sorted(set(view_ids)):
            if view_id == view_ids[seed]:
                continue
            candidates = [
                index
                for index in range(len(scores))
                if index not in assigned
                and view_ids[index] == view_id
                and _iou(boxes[seed], boxes[index]) >= iou_threshold
            ]
            if candidates:
                selected = max(candidates, key=lambda index: float(scores[index]))
                members.append(selected)
                assigned.add(selected)
        if len(members) < minimum_support:
            continue
        member_scores = scores[members]
        weights = member_scores / member_scores.sum().clip(min=1e-12)
        fused_box = (boxes[members] * weights[:, None]).sum(axis=0)
        fused["boxes_xyxy"].append(fused_box.astype(float).tolist())
        if score_mode == "max":
            fused_score = float(member_scores.max())
        elif score_mode == "mean":
            fused_score = float(member_scores.sum() / view_count)
        elif score_mode == "geometric":
            fused_score = float(
                np.exp(np.log(member_scores.clip(min=1e-12)).mean()) * len(members) / view_count
            )
        else:
            raise ValueError(f"unsupported rotation score fusion: {score_mode}")
        fused["scores"].append(fused_score)
        fused["rotation_support_count"].append(len(members))
        fused["rotation_score_standard_deviation"].append(float(member_scores.std()))
        fused_area = max(
            float((fused_box[2] - fused_box[0]) * (fused_box[3] - fused_box[1])),
            1e-12,
        )
        fused["rotation_box_dispersion"].append(
            float(np.linalg.norm(boxes[members].std(axis=0)) / np.sqrt(fused_area))
        )
        representative = int(members[int(member_scores.argmax())])
        for key in prediction:
            if key not in {"boxes_xyxy", "scores"}:
                fused[key].append(prediction[key][representative])
    return fused


def load_records(
    dataset_root: Path,
    annotation_name: str,
    *,
    annotation_path: Path | None = None,
) -> list[dict[str, Any]]:
    dataset_root = dataset_root.resolve()
    explicit_annotation = annotation_path is not None
    annotation_path = (
        annotation_path.resolve()
        if explicit_annotation
        else dataset_root / "annotations" / annotation_name
    )
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    declared_category_ids = [int(row["id"]) for row in payload.get("categories", [])]
    annotation_category_ids = [int(row["category_id"]) for row in payload["annotations"]]
    category_ids = declared_category_ids or annotation_category_ids
    # Canonical bread annotations use 1-based IDs, while the generated D-FINE
    # COCO splits use 0-based IDs. Normalize both inputs to the canonical form.
    category_id_offset = 1 if category_ids and min(category_ids) == 0 else 0
    annotations: dict[int, list[dict[str, Any]]] = {}
    for row in payload["annotations"]:
        annotations.setdefault(int(row["image_id"]), []).append(
            {
                "bbox_xywh": [float(value) for value in row["bbox"]],
                "category_id": int(row["category_id"]) + category_id_offset,
            }
        )
    records = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        path = resolve_coco_image(
            dataset_root,
            annotation_path,
            str(image["file_name"]),
        )
        records.append(
            {
                "image_id": int(image["id"]),
                "image_path": path,
                "expected_image_status": str(image.get("status", "ANNOTATED")),
                "annotations": annotations.get(int(image["id"]), []),
            }
        )
    return records


def detector_classification_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    max_object_aspect_ratio: float | None,
) -> dict[str, float | int | None]:
    matched = 0
    top1_correct = 0
    top3_correct = 0
    for record, prediction in zip(records, predictions):
        candidates = [
            (index, Detection(*box, score))
            for index, (box, score) in enumerate(
                zip(prediction["boxes_xyxy"], prediction["scores"])
            )
            if score >= score_threshold and _allowed_aspect_ratio(box, max_object_aspect_ratio)
        ]
        index_by_detection = {detection: index for index, detection in candidates}
        selected = nms([detection for _, detection in candidates], nms_iou_threshold)
        remaining = set(range(len(record["annotations"])))
        for detection in selected:
            prediction_index = index_by_detection[detection]
            box = np.asarray(
                [detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32
            )
            overlaps = [
                (gt_index, _iou(box, _xywh_to_xyxy(record["annotations"][gt_index]["bbox_xywh"])))
                for gt_index in remaining
            ]
            if not overlaps:
                continue
            gt_index, overlap = max(overlaps, key=lambda item: item[1])
            if overlap < match_iou_threshold:
                continue
            remaining.remove(gt_index)
            target = int(record["annotations"][gt_index]["category_id"]) - 1
            matched += 1
            top1_correct += int(prediction["class_ids"][prediction_index] == target)
            top3_correct += int(target in prediction["top3_class_ids"][prediction_index])
    return {
        "matched_sample_count": matched,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / matched if matched else None,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / matched if matched else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(
        args.dataset_root.resolve(),
        args.annotation,
        annotation_path=getattr(args, "annotation_path", None),
    )
    if args.expected_status is not None:
        records = [
            record for record in records if record["expected_image_status"] == args.expected_status
        ]
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    predictions = []
    query_count: int | None = None
    for record in records:
        jpeg_draft_size = getattr(args, "jpeg_draft_size", None)
        if jpeg_draft_size is None:
            with Image.open(record["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            image = decode_image(
                record["image_path"].read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=jpeg_draft_size,
            )
        width, height = image_original_size(image)
        rotation_tta = getattr(args, "rotation_tta", 1)
        turns_values = (
            (0,) if rotation_tta == 1 else ((0, 2) if rotation_tta == 2 else (0, 1, 2, 3))
        )
        prediction = {
            "boxes_xyxy": [],
            "scores": [],
            "class_ids": [],
            "top3_class_ids": [],
        }
        view_ids = []
        class_logits = []
        try:
            for turns in turns_values:
                rotated = image if turns == 0 else image.rotate(90 * turns, expand=True)
                try:
                    rotated_width, rotated_height = image_original_size(rotated)
                    input_mean = getattr(args, "mean", (0.0, 0.0, 0.0))
                    input_std = getattr(args, "std", (1.0, 1.0, 1.0))
                    tensor = prepare_rgb(
                        rotated,
                        (args.input_height, args.input_width),
                        tuple(float(value) for value in input_mean),
                        tuple(float(value) for value in input_std),
                        reducing_gap=1.0,
                    )[None]
                    logits, boxes = runner.run(
                        [args.logits_output, args.boxes_output], args.input_name, tensor
                    )
                finally:
                    if rotated is not image:
                        rotated.close()
                logits = np.asarray(logits)[0]
                boxes = np.asarray(boxes)[0]
                query_count = len(logits) if query_count is None else query_count
                view_prediction = raw_outputs_to_prediction(
                    logits,
                    boxes,
                    image_width=rotated_width,
                    image_height=rotated_height,
                )
                if getattr(args, "include_class_logits", False):
                    view_prediction["class_logits"] = np.asarray(logits, dtype=np.float32).tolist()
                view_prediction = _preselect_view_prediction(
                    view_prediction,
                    minimum_score=args.min_score_threshold,
                    nms_iou_threshold=args.nms_threshold,
                    max_object_aspect_ratio=args.max_object_aspect_ratio,
                    maximum_candidates=getattr(args, "rotation_max_candidates_per_view", None),
                )
                view_prediction["boxes_xyxy"] = _restore_rotated_boxes(
                    view_prediction["boxes_xyxy"],
                    turns=turns,
                    original_width=width,
                    original_height=height,
                )
                count = len(view_prediction["scores"])
                view_ids.extend([turns] * count)
                for key in prediction:
                    prediction[key].extend(view_prediction[key])
                if getattr(args, "include_class_logits", False):
                    class_logits.extend(view_prediction["class_logits"])
        finally:
            image.close()
        if getattr(args, "include_class_logits", False):
            prediction["class_logits"] = class_logits
        rotation_score_fusion = getattr(args, "rotation_score_fusion", "none")
        if rotation_score_fusion == "none":
            prediction = _filter_rotation_support(
                prediction,
                view_ids,
                minimum_support=getattr(args, "rotation_min_support", 1),
                iou_threshold=getattr(args, "rotation_support_iou", 0.5),
                minimum_score=args.min_score_threshold,
            )
        else:
            prediction = _fuse_rotation_predictions(
                prediction,
                view_ids,
                view_count=len(turns_values),
                minimum_support=getattr(args, "rotation_min_support", 1),
                iou_threshold=getattr(args, "rotation_support_iou", 0.5),
                score_mode=rotation_score_fusion,
            )
        prediction["image_id"] = record["image_id"]
        predictions.append(prediction)

    thresholds = np.linspace(
        args.min_score_threshold,
        args.max_score_threshold,
        args.threshold_steps,
        dtype=np.float64,
    )
    candidates = _metrics_grid(
        records,
        predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=query_count or args.max_queries,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
    )
    selected = select_release_threshold_candidate(candidates, args.target_recall)
    class_metrics = detector_classification_metrics(
        records,
        predictions,
        score_threshold=float(selected["score_threshold"]),
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
    )
    report = {
        "evaluation": "onnx_detector_threshold_selection",
        "model": args.model.name,
        "annotation": args.annotation,
        "expected_status_filter": args.expected_status,
        "provider": args.provider,
        "input_normalization": {
            "mean": [float(value) for value in getattr(args, "mean", (0.0, 0.0, 0.0))],
            "std": [float(value) for value in getattr(args, "std", (1.0, 1.0, 1.0))],
        },
        "jpeg_draft_size": getattr(args, "jpeg_draft_size", None),
        "rotation_tta": getattr(args, "rotation_tta", 1),
        "rotation_min_support": getattr(args, "rotation_min_support", 1),
        "rotation_max_candidates_per_view": getattr(args, "rotation_max_candidates_per_view", None),
        "rotation_score_fusion": getattr(args, "rotation_score_fusion", "none"),
        "threshold_policy": "recall_floor_then_precision_on_development_set",
        "target_recall": args.target_recall,
        "target_recall_satisfied": float(selected["recall"]) >= args.target_recall,
        "selected_score_threshold": selected["score_threshold"],
        "metrics": selected,
        "detector_classification_on_matched": class_metrics,
        "query_count": query_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an ONNX detector threshold")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", default="multi_object_instances.json")
    parser.add_argument(
        "--annotation-path",
        type=Path,
        help="Use an explicit COCO annotation while resolving file_name under --dataset-root",
    )
    parser.add_argument("--expected-status")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--provider",
        choices=("cuda", "cpu", "openvino", "openvino_gpu"),
        default="cuda",
    )
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("R", "G", "B"),
        help="External RGB normalization mean; use 0.5 0.5 0.5 for exported SSDLite",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help="External RGB normalization std; use 0.5 0.5 0.5 for exported SSDLite",
    )
    parser.add_argument("--jpeg-draft-size", type=int)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--rotation-tta", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rotation-min-support", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--rotation-support-iou", type=float, default=0.5)
    parser.add_argument("--rotation-max-candidates-per-view", type=int)
    parser.add_argument(
        "--rotation-score-fusion",
        choices=("none", "max", "mean", "geometric"),
        default="none",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--max-object-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--min-score-threshold", type=float, default=0.01)
    parser.add_argument("--max-score-threshold", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=197)
    parser.add_argument(
        "--include-class-logits",
        action="store_true",
        help="Include raw per-query class logits in diagnostic prediction output",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
