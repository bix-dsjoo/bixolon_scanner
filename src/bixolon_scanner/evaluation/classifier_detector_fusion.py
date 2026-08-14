from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline.ports import Detection
from ..runtime.onnx import box_iou, nms


def _normalized(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - values.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return centered / np.maximum(scale, 1e-12)


def _margins(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def fusion_candidates(
    dino_logits: np.ndarray, detector_logits: np.ndarray
) -> dict[str, np.ndarray]:
    if dino_logits.shape != detector_logits.shape or dino_logits.ndim != 2:
        raise ValueError("fusion logits must have matching [samples, classes] shapes")
    dino = _normalized(dino_logits)
    detector = _normalized(detector_logits)
    dino_top1 = dino.argmax(axis=1)
    detector_top1 = detector.argmax(axis=1)
    dino_margin = _margins(dino)
    detector_margin = _margins(detector)
    candidates: dict[str, np.ndarray] = {"dino_only": dino_top1}
    for weight in np.linspace(0.0, 1.0, 41):
        candidates[f"zblend_{weight:.3f}"] = (dino * weight + detector * (1.0 - weight)).argmax(
            axis=1
        )
    for dino_threshold in np.linspace(0.0, 2.5, 26):
        for detector_threshold in np.linspace(0.0, 2.5, 26):
            use_detector = (dino_margin <= dino_threshold) & (detector_margin >= detector_threshold)
            candidates[f"margin_gate_{dino_threshold:.2f}_{detector_threshold:.2f}"] = np.where(
                use_detector, detector_top1, dino_top1
            )
    return candidates


def cross_validated_selection(
    candidates: dict[str, np.ndarray], targets: np.ndarray, folds: np.ndarray
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one fusion candidate is required")
    if targets.shape != folds.shape:
        raise ValueError("targets and folds must have matching shapes")
    fold_results = []
    held_out_predictions = np.full(len(targets), -1, dtype=np.int64)
    for fold in sorted(set(int(value) for value in folds)):
        training = folds != fold
        held_out = folds == fold
        selected_name = max(
            sorted(candidates),
            key=lambda name: int(np.count_nonzero(candidates[name][training] == targets[training])),
        )
        predictions = candidates[selected_name]
        held_out_predictions[held_out] = predictions[held_out]
        fold_results.append(
            {
                "fold": fold,
                "selected": selected_name,
                "selection_correct": int(
                    np.count_nonzero(predictions[training] == targets[training])
                ),
                "selection_sample_count": int(training.sum()),
                "held_out_correct": int(
                    np.count_nonzero(predictions[held_out] == targets[held_out])
                ),
                "held_out_sample_count": int(held_out.sum()),
            }
        )
    correct = int(np.count_nonzero(held_out_predictions == targets))
    return {
        "correct": correct,
        "sample_count": len(targets),
        "top1_accuracy": correct / len(targets),
        "folds": fold_results,
    }


def _allowed_aspect_ratio(box: list[float], maximum: float) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width > 0 and height > 0 and max(width / height, height / width) <= maximum


def joined_logits(
    *,
    annotations: dict[str, Any],
    detector_predictions: list[dict[str, Any]],
    classifier_records: list[dict[str, Any]],
    classifier_logits: np.ndarray,
    score_threshold: float,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    maximum_aspect_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    detector_by_key: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for prediction in detector_predictions:
        if "class_logits" not in prediction:
            raise ValueError("detector prediction is missing class_logits")
        index_by_detection: dict[Detection, int] = {}
        candidates = []
        for index, (box, score) in enumerate(zip(prediction["boxes_xyxy"], prediction["scores"])):
            if score < score_threshold or not _allowed_aspect_ratio(box, maximum_aspect_ratio):
                continue
            detection = Detection(*box, float(score))
            candidates.append(detection)
            index_by_detection[detection] = index
        selected = nms(candidates, nms_iou_threshold)
        image_id = int(prediction["image_id"])
        ground_truth = annotations_by_image[image_id]
        remaining = set(range(len(ground_truth)))
        for detection in sorted(selected, key=lambda value: value.score, reverse=True):
            overlaps = []
            for index in remaining:
                x, y, width, height = (float(value) for value in ground_truth[index]["bbox"])
                target_box = Detection(x, y, x + width, y + height, 1.0)
                overlaps.append((index, box_iou(detection, target_box)))
            if not overlaps:
                continue
            target_index, overlap = max(overlaps, key=lambda row: row[1])
            if overlap < match_iou_threshold:
                continue
            remaining.remove(target_index)
            target = int(ground_truth[target_index]["category_id"]) - 1
            raw_index = index_by_detection[detection]
            detector_by_key[(image_id, target)].append(
                np.asarray(prediction["class_logits"][raw_index], dtype=np.float32)
            )

    joined_classifier = []
    joined_detector = []
    joined_targets = []
    joined_folds = []
    used: dict[tuple[int, int], int] = defaultdict(int)
    for record, logits in zip(classifier_records, classifier_logits):
        key = (int(record["image_id"]), int(record["target"]))
        index = used[key]
        used[key] += 1
        if index >= len(detector_by_key[key]):
            continue
        joined_classifier.append(np.asarray(logits, dtype=np.float32))
        joined_detector.append(detector_by_key[key][index])
        joined_targets.append(key[1])
        joined_folds.append(int(record["fold"]))
    return (
        np.asarray(joined_classifier, dtype=np.float32),
        np.asarray(joined_detector, dtype=np.float32),
        np.asarray(joined_targets, dtype=np.int64),
        np.asarray(joined_folds, dtype=np.int64),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    annotations = json.loads(args.annotation.read_text(encoding="utf-8-sig"))
    detector_predictions = [
        json.loads(line)
        for line in args.detector_predictions.read_text(encoding="utf-8").splitlines()
        if line
    ]
    classifier_records = [
        json.loads(line)
        for line in args.classifier_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    classifier_logits = np.load(args.classifier_logits)
    dino, detector, targets, folds = joined_logits(
        annotations=annotations,
        detector_predictions=detector_predictions,
        classifier_records=classifier_records,
        classifier_logits=classifier_logits,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        maximum_aspect_ratio=args.maximum_aspect_ratio,
    )
    candidates = fusion_candidates(dino, detector)
    cross_validated = cross_validated_selection(candidates, targets, folds)
    dino_top1 = dino.argmax(axis=1)
    detector_order = np.argsort(-detector, axis=1, kind="stable")
    dino_correct = dino_top1 == targets
    detector_top1_correct = detector_order[:, 0] == targets
    detector_top3_correct = np.any(detector_order[:, :3] == targets[:, None], axis=1)
    best_in_sample_name = max(
        sorted(candidates),
        key=lambda name: int(np.count_nonzero(candidates[name] == targets)),
    )
    best_in_sample_correct = int(np.count_nonzero(candidates[best_in_sample_name] == targets))
    report = {
        "evaluation": "classifier_detector_class_fusion_diagnostic_only",
        "promotion_status": "diagnostic_only",
        "joined_sample_count": len(targets),
        "missing_classifier_samples": len(classifier_records) - len(targets),
        "candidate_count": len(candidates),
        "dino_top1": {
            "correct": int(dino_correct.sum()),
            "accuracy": float(dino_correct.mean()),
        },
        "detector_top1": {
            "correct": int(detector_top1_correct.sum()),
            "accuracy": float(detector_top1_correct.mean()),
        },
        "oracle_either_top1": {
            "correct": int(np.count_nonzero(dino_correct | detector_top1_correct)),
            "accuracy": float(np.mean(dino_correct | detector_top1_correct)),
        },
        "oracle_dino_top1_or_detector_top3": {
            "correct": int(np.count_nonzero(dino_correct | detector_top3_correct)),
            "accuracy": float(np.mean(dino_correct | detector_top3_correct)),
        },
        "best_in_sample_policy": {
            "name": best_in_sample_name,
            "correct": best_in_sample_correct,
            "accuracy": best_in_sample_correct / len(targets),
        },
        "cross_validated_policy": cross_validated,
        "passes_top1_gate": cross_validated["top1_accuracy"] >= 0.99,
        "limitations": [
            "Uses ground-truth matching and development labels, so it cannot be promoted.",
            "The cross-validated policy is a feasibility diagnostic, not an independent locked test.",
            "Missing detector matches are excluded and must be counted as errors in a Worker gate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe DINO and detector class-logit fusion")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--classifier-records", type=Path, required=True)
    parser.add_argument("--classifier-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.485)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--maximum-aspect-ratio", type=float, default=5.0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
