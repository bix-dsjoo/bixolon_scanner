from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..contracts.model_package import DetectorMetadata
from ..pipeline.ports import Detection, DetectionResult
from .imaging import image_original_size, redraft_image, restore_original_resolution
from .onnx import (
    OnnxClassifier,
    OrtRunner,
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
    sigmoid,
)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_ious(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    left = np.asarray(box, dtype=np.float64)
    right = np.asarray(boxes, dtype=np.float64)
    upper_left = np.maximum(left[:2], right[:, :2])
    lower_right = np.minimum(left[2:], right[:, 2:])
    intersection = np.prod(np.maximum(0.0, lower_right - upper_left), axis=1)
    left_area = np.prod(np.maximum(0.0, left[2:] - left[:2]))
    right_area = np.prod(np.maximum(0.0, right[:, 2:] - right[:, :2]), axis=1)
    union = left_area + right_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0.0,
    )


def _pairwise_ious(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if not len(values):
        return np.empty((0, 0), dtype=np.float64)
    upper_left = np.maximum(values[:, None, :2], values[None, :, :2])
    lower_right = np.minimum(values[:, None, 2:], values[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, lower_right - upper_left), axis=2)
    areas = np.prod(np.maximum(0.0, values[:, 2:] - values[:, :2]), axis=1)
    union = areas[:, None] + areas[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0.0,
    )


def _nms_candidate_indices(candidates: list[dict[str, Any]], threshold: float) -> list[int]:
    if not candidates:
        return []
    order = sorted(
        range(len(candidates)), key=lambda index: candidates[index]["score"], reverse=True
    )
    overlaps = _pairwise_ious(
        np.asarray([candidate["box"] for candidate in candidates], dtype=np.float32)
    )
    selected = []
    suppressed = np.zeros(len(candidates), dtype=bool)
    for index in order:
        if suppressed[index]:
            continue
        selected.append(index)
        suppressed |= overlaps[index] > threshold
        suppressed[index] = False
    return selected


def _area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _maximum_aspect_ratio_extremity(prediction: dict[str, Any]) -> float:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float64)
    if not len(boxes):
        return 1.0
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    valid = (widths > 0.0) & (heights > 0.0)
    if not np.any(valid):
        return 1.0
    ratios = widths[valid] / heights[valid]
    return float(np.max(np.maximum(ratios, 1.0 / ratios)))


def _filter_prediction_by_area(
    prediction: dict[str, Any], *, image_area: float, maximum_area_ratio: float
) -> dict[str, Any]:
    kept = [
        index
        for index, box in enumerate(prediction["boxes_xyxy"])
        if _area(np.asarray(box, dtype=np.float32)) / image_area <= maximum_area_ratio
    ]
    output = dict(prediction)
    for key in ("boxes_xyxy", "scores", "class_ids", "support_counts"):
        if key in prediction:
            output[key] = [prediction[key][index] for index in kept]
    return output


def _intersection(left: np.ndarray, right: np.ndarray) -> float:
    return max(
        0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))
    ) * max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))


def detector_output_to_prediction(
    logits: np.ndarray,
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float32)
    normalized = np.asarray(boxes, dtype=np.float32)
    scores = sigmoid(values).max(axis=-1)
    classes = np.argmax(values, axis=-1)
    converted_boxes = []
    converted_scores = []
    converted_classes = []
    for score, class_id, box in zip(scores, classes, normalized):
        center_x, center_y, width, height = [float(value) for value in box]
        x1 = max(0.0, (center_x - width * 0.5) * image_width)
        y1 = max(0.0, (center_y - height * 0.5) * image_height)
        x2 = min(float(image_width), (center_x + width * 0.5) * image_width)
        y2 = min(float(image_height), (center_y + height * 0.5) * image_height)
        if x2 > x1 and y2 > y1:
            converted_boxes.append([x1, y1, x2, y2])
            converted_scores.append(float(score))
            converted_classes.append(int(class_id))
    return {
        "boxes_xyxy": converted_boxes,
        "scores": converted_scores,
        "class_ids": converted_classes,
    }


def fuse_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    model_weights: list[float],
    score_thresholds: list[float],
    pre_nms_iou_threshold: float,
    maximum_candidates_per_model: int,
    cluster_iou_threshold: float,
) -> dict[str, Any]:
    candidates = []
    for source_id, (row, threshold) in enumerate(zip(rows, score_thresholds)):
        source = [
            {
                "box": np.asarray(box, dtype=np.float32),
                "score": float(score),
                "class_id": int(class_id),
                "source_id": source_id,
            }
            for box, score, class_id in zip(row["boxes_xyxy"], row["scores"], row["class_ids"])
            if float(score) >= threshold
        ]
        source.sort(key=lambda item: (-item["score"], *item["box"].tolist()))
        if pre_nms_iou_threshold >= 1.0:
            selected = source[:maximum_candidates_per_model]
        else:
            selected = []
            for candidate in source:
                if all(
                    _iou(candidate["box"], current["box"]) <= pre_nms_iou_threshold
                    for current in selected
                ):
                    selected.append(candidate)
                    if len(selected) == maximum_candidates_per_model:
                        break
        candidates.extend(selected)
    weights = np.asarray(model_weights, dtype=np.float64)
    candidates.sort(
        key=lambda item: (
            -item["score"] * float(weights[item["source_id"]]),
            item["source_id"],
            *item["box"].tolist(),
        )
    )
    clusters: list[dict[str, Any]] = []
    fused_boxes = np.empty((len(candidates), 4), dtype=np.float32)
    source_masks = np.zeros(len(candidates), dtype=np.uint64)
    for candidate in candidates:
        best_index = None
        cluster_count = len(clusters)
        source_bit = np.uint64(1 << int(candidate["source_id"]))
        if cluster_count:
            overlaps = _box_ious(candidate["box"], fused_boxes[:cluster_count])
            overlaps[(source_masks[:cluster_count] & source_bit) != 0] = -1.0
            eligible = overlaps >= cluster_iou_threshold
            if eligible.any():
                best_index = int(np.argmax(np.where(eligible, overlaps, -1.0)))
        if best_index is None:
            member_weight = max(candidate["score"], 1e-9) * weights[candidate["source_id"]]
            clusters.append(
                {
                    "members": [candidate],
                    "weighted_box_sum": candidate["box"].astype(np.float64) * member_weight,
                    "total_box_weight": member_weight,
                }
            )
            fused_boxes[cluster_count] = candidate["box"]
            source_masks[cluster_count] = source_bit
        else:
            cluster = clusters[best_index]
            cluster["members"].append(candidate)
            source_masks[best_index] |= source_bit
            member_weight = max(candidate["score"], 1e-9) * weights[candidate["source_id"]]
            cluster["weighted_box_sum"] += candidate["box"] * member_weight
            cluster["total_box_weight"] += member_weight
            fused_boxes[best_index] = (
                cluster["weighted_box_sum"] / cluster["total_box_weight"]
            ).astype(np.float32)
    outputs = []
    for cluster, box in zip(clusters, fused_boxes[: len(clusters)]):
        members = cluster["members"]
        source_ids = sorted({int(member["source_id"]) for member in members})
        class_scores: Counter[int] = Counter()
        for member in members:
            class_scores[int(member["class_id"])] += float(member["score"]) * float(
                weights[int(member["source_id"])]
            )
        outputs.append(
            {
                "box": box.tolist(),
                "score": max(float(member["score"]) for member in members),
                "class_id": int(
                    max(class_scores, key=lambda class_id: (class_scores[class_id], -class_id))
                ),
                "support_count": len(source_ids),
            }
        )
    outputs.sort(key=lambda item: (-item["score"], -item["support_count"], *item["box"]))
    return {
        "boxes_xyxy": [item["box"] for item in outputs],
        "scores": [item["score"] for item in outputs],
        "class_ids": [item["class_id"] for item in outputs],
        "support_counts": [item["support_count"] for item in outputs],
    }


def _containment_select(
    prediction: dict[str, Any],
    *,
    score_threshold: float,
    iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
) -> dict[str, Any]:
    candidates = [
        {
            "box": np.asarray(box, dtype=np.float32),
            "score": float(score),
            "class_id": int(class_id),
        }
        for box, score, class_id in zip(
            prediction["boxes_xyxy"], prediction["scores"], prediction["class_ids"]
        )
        if float(score) >= score_threshold
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    kept = []
    for index, candidate in enumerate(candidates):
        candidate_area = _area(candidate["box"])
        if candidate_area <= 0.0:
            continue
        stronger_inside = [
            other
            for other in candidates[:index]
            if _area(other["box"]) > 0.0
            and _intersection(candidate["box"], other["box"]) / _area(other["box"])
            >= containment_threshold
        ]
        suppressed = False
        for current in kept:
            current_area = _area(current["box"])
            intersection = _intersection(current["box"], candidate["box"])
            candidate_inside = intersection / candidate_area >= containment_threshold
            current_inside = (
                current_area > 0.0 and intersection / current_area >= containment_threshold
            )
            same_class = current["class_id"] == candidate["class_id"]
            if (
                _iou(current["box"], candidate["box"]) > iou_threshold
                or candidate_inside
                or current_inside
                and (same_class or len(stronger_inside) >= group_minimum)
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return {
        "boxes_xyxy": [item["box"].tolist() for item in kept],
        "scores": [item["score"] for item in kept],
        "class_ids": [item["class_id"] for item in kept],
    }


def _group_suppress(
    candidates: list[dict[str, Any]], *, containment_threshold: float, group_minimum: int
) -> list[dict[str, Any]]:
    if not group_minimum:
        return candidates
    boxes = np.asarray([candidate["box"] for candidate in candidates], dtype=np.float64)
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    upper_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    lower_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, lower_right - upper_left), axis=2)
    containment = np.divide(
        intersection,
        areas[None, :],
        out=np.zeros_like(intersection),
        where=areas[None, :] > 0.0,
    )
    union = areas[:, None] + areas[None, :] - intersection
    overlaps = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    scores = np.asarray([candidate["score"] for candidate in candidates], dtype=np.float64)
    output = []
    for index, candidate in enumerate(candidates):
        inner_indices = np.flatnonzero(
            (np.arange(len(candidates)) != index)
            & (areas > 0.0)
            & (areas < areas[index] * 0.8)
            & (containment[index] >= containment_threshold)
        )
        order = sorted(inner_indices.tolist(), key=lambda inner: scores[inner], reverse=True)
        selected: list[int] = []
        for inner_index in order:
            if all(overlaps[inner_index, current] <= 0.3 for current in selected):
                selected.append(inner_index)
                if len(selected) == group_minimum:
                    break
        distinct_count = len(selected)
        if distinct_count < group_minimum:
            output.append(candidate)
    return output


def _available_prediction(raw: dict[str, Any], rule: Any) -> dict[str, Any]:
    candidates = [
        {
            "box": np.asarray(box, dtype=np.float32),
            "score": float(score),
            "class_id": int(class_id),
        }
        for box, score, class_id in zip(raw["boxes_xyxy"], raw["scores"], raw["class_ids"])
        if float(score) >= rule.availability_score_threshold
    ]
    candidates = _group_suppress(
        candidates,
        containment_threshold=rule.availability_containment_threshold,
        group_minimum=rule.availability_group_minimum,
    )
    selected = [
        candidates[index]
        for index in _nms_candidate_indices(candidates, rule.availability_nms_iou_threshold)
    ]
    return {
        "boxes_xyxy": [item["box"].tolist() for item in selected],
        "scores": [item["score"] for item in selected],
        "class_ids": [item["class_id"] for item in selected],
    }


def is_ambiguous(raw: dict[str, Any], base: dict[str, Any], rules: list[Any]) -> bool:
    for rule in rules:
        available = _available_prediction(raw, rule)
        selected_count = len(base["scores"])
        extra_count = len(available["scores"]) - selected_count
        next_score = float(available["scores"][selected_count]) if extra_count > 0 else -1.0
        extra_matches = (
            extra_count == rule.extra_candidate_count
            if rule.extra_count_mode == "exact"
            else extra_count >= rule.extra_candidate_count
        )
        if (
            selected_count >= rule.minimum_selected_count
            and extra_matches
            and next_score >= rule.next_score_threshold_inclusive
        ):
            return True
    return False


def _predictions_agree(
    primary: dict[str, Any], recovery: dict[str, Any], *, iou_threshold: float
) -> bool:
    primary_boxes = np.asarray(primary["boxes_xyxy"], dtype=np.float32)
    recovery_boxes = np.asarray(recovery["boxes_xyxy"], dtype=np.float32)
    if len(primary_boxes) != len(recovery_boxes):
        return False
    if not len(primary_boxes):
        return True
    overlaps = np.asarray(
        [_box_ious(box, recovery_boxes) for box in primary_boxes], dtype=np.float64
    )
    adjacency = [np.flatnonzero(row >= iou_threshold).tolist() for row in overlaps]
    matched_primary = [-1] * len(recovery_boxes)

    def augment(primary_index: int, seen: set[int]) -> bool:
        for recovery_index in adjacency[primary_index]:
            if recovery_index in seen:
                continue
            seen.add(recovery_index)
            previous = matched_primary[recovery_index]
            if previous == -1 or augment(previous, seen):
                matched_primary[recovery_index] = primary_index
                return True
        return False

    return all(augment(index, set()) for index in range(len(primary_boxes)))


def consensus_agreement_count(
    base: dict[str, Any], rows_by_filename: dict[str, dict[str, Any]], policy: Any | None
) -> int:
    if policy is None:
        return 1
    agreement_count = 1
    for member_policy in policy.policies:
        member = _containment_select(
            rows_by_filename[member_policy.member_filename],
            score_threshold=member_policy.score_threshold,
            iou_threshold=member_policy.nms_iou_threshold,
            containment_threshold=member_policy.containment_threshold,
            group_minimum=member_policy.group_minimum,
        )
        agreement_count += _predictions_agree(
            base, member, iou_threshold=policy.agreement_iou_threshold
        )
    return agreement_count


def consensus_is_ambiguous(
    base: dict[str, Any], rows_by_filename: dict[str, dict[str, Any]], policy: Any | None
) -> bool:
    if policy is None:
        return False
    return (
        consensus_agreement_count(base, rows_by_filename, policy)
        < policy.minimum_agreeing_policy_count
    )


def _candidate_context(
    base_boxes: np.ndarray, candidate_box: np.ndarray, *, duplicate_iou: float
) -> tuple[list[list[float]], int]:
    kept = (
        base_boxes[_box_ious(candidate_box, base_boxes) < duplicate_iou]
        if len(base_boxes)
        else np.zeros((0, 4), dtype=np.float32)
    )
    boxes = [*kept.tolist(), candidate_box.tolist()]
    return boxes, len(boxes) - 1


def _class_verified_select(
    base: dict[str, Any],
    raw: dict[str, Any],
    entries: list[dict[str, Any]],
    policy: Any,
) -> dict[str, Any]:
    base_boxes = np.asarray(base["boxes_xyxy"], dtype=np.float32)
    raw_boxes = np.asarray(raw["boxes_xyxy"], dtype=np.float32)
    entries_by_index = {int(row["proposal_index"]): row for row in entries}
    mapped = []
    for box in base_boxes:
        overlaps = _box_ious(box, raw_boxes)
        raw_index = int(np.argmax(overlaps))
        if float(overlaps[raw_index]) < policy.base_match_iou:
            raise ValueError("base detector box cannot be mapped to the proposal union")
        entry = entries_by_index.get(raw_index)
        if (
            entry is not None
            and raw["support_counts"][raw_index] >= policy.candidate_minimum_support
        ):
            mapped.append(entry)
    by_class: dict[int, dict[str, Any]] = {}
    for entry in mapped:
        class_id = int(entry["predicted_class"])
        current = by_class.get(class_id)
        if current is None or entry["detector_score"] > current["detector_score"]:
            by_class[class_id] = entry
    selected = list(by_class.values())
    used = {int(row["proposal_index"]) for row in selected}
    for base_entry in list(selected):
        base_area = _area(base_entry["box"])
        if base_area <= 0.0:
            continue
        current_classes = {int(row["predicted_class"]) for row in selected}
        alternatives = []
        novel = []
        for candidate in entries:
            if int(candidate["proposal_index"]) in used:
                continue
            if _iou(candidate["box"], base_entry["box"]) < policy.group_relation_iou:
                continue
            if _area(candidate["box"]) > base_area * policy.group_area_ratio:
                continue
            if candidate["predicted_class"] == base_entry["predicted_class"]:
                if (
                    candidate["class_margin"]
                    >= base_entry["class_margin"] * policy.group_margin_ratio
                ):
                    alternatives.append(candidate)
            elif (
                candidate["predicted_class"] not in current_classes
                and candidate["class_margin"] >= policy.group_novel_margin
                and candidate["detector_score"] >= policy.group_minimum_score
            ):
                novel.append(candidate)
        if not alternatives or not novel:
            continue
        alternative = max(alternatives, key=lambda row: row["detector_score"])
        compatible = [row for row in novel if _iou(row["box"], alternative["box"]) < 0.5]
        if not compatible:
            continue
        novel_entry = max(compatible, key=lambda row: row["detector_score"])
        selected.remove(base_entry)
        selected.extend((alternative, novel_entry))
        used.discard(int(base_entry["proposal_index"]))
        used.update((int(alternative["proposal_index"]), int(novel_entry["proposal_index"])))
    current_classes = {int(row["predicted_class"]) for row in selected}
    independent = [
        row
        for row in entries
        if int(row["proposal_index"]) not in used
        and int(row["predicted_class"]) not in current_classes
        and row["class_margin"] >= policy.independent_margin
        and row["detector_score"] >= policy.independent_minimum_score
        and (
            not selected
            or max(_iou(row["box"], other["box"]) for other in selected)
            < policy.independent_maximum_iou
        )
    ]
    if independent:
        best_class = int(max(independent, key=lambda row: row["class_margin"])["predicted_class"])
        same_class = [row for row in independent if row["predicted_class"] == best_class]
        selected.append(max(same_class, key=lambda row: _area(row["box"])))
    selected.sort(key=lambda row: row["detector_score"], reverse=True)
    return {
        "boxes_xyxy": [row["box"].tolist() for row in selected],
        "scores": [float(row["detector_score"]) for row in selected],
        "class_ids": [int(row["predicted_class"]) for row in selected],
    }


class BreadZeroErrorDetector:
    def __init__(
        self,
        model_paths: list[Path],
        metadata: DetectorMetadata,
        classifier: OnnxClassifier,
        provider: str,
        cuda_dll_dir: Path | None = None,
    ):
        if metadata.ensemble is None:
            raise ValueError("Bread zero-error detector requires ensemble metadata")
        if classifier.metadata.neighbor_mask_inference is None:
            raise ValueError("proposal verification requires neighbor-mask classifier metadata")
        if len(classifier.metadata.neighbor_mask_inference.views) != 1:
            raise ValueError("proposal verification requires exactly one neighbor-mask view")
        self.metadata = metadata
        self.ensemble = metadata.ensemble
        self.classifier = classifier
        self.runners = [
            OrtRunner(
                path,
                provider,
                cuda_dll_dir,
                enable_cuda_graph=self.ensemble.cuda_graph_execution,
                cuda_graph_output_shapes={
                    self.metadata.logits_output: (
                        1,
                        self.metadata.max_queries,
                        len(classifier.metadata.labels),
                    ),
                    self.metadata.boxes_output: (1, self.metadata.max_queries, 4),
                },
            )
            for path in model_paths
        ]
        self.executor = (
            ThreadPoolExecutor(max_workers=len(self.runners))
            if self.ensemble.parallel_execution
            else None
        )
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        for runner in self.runners:
            runner.run(
                [self.metadata.logits_output, self.metadata.boxes_output],
                self.metadata.input_name,
                dummy,
            )

    def _run_models(self, tensor: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        def run(runner: OrtRunner) -> tuple[np.ndarray, np.ndarray]:
            logits, boxes = runner.run(
                [self.metadata.logits_output, self.metadata.boxes_output],
                self.metadata.input_name,
                tensor,
            )
            return np.asarray(logits)[0], np.asarray(boxes)[0]

        if self.executor is None:
            return [run(runner) for runner in self.runners]
        return list(self.executor.map(run, self.runners))

    def _proposal_scores(
        self,
        image: np.ndarray | Image.Image,
        base: dict[str, Any],
        raw: dict[str, Any],
        indices: list[int],
        *,
        image_width: int,
        image_height: int,
        restore_source_for_crops: bool = True,
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            pil_image = restore_original_resolution(image) if restore_source_for_crops else image
            pixel_width, pixel_height = image.size
            if pil_image is not image:
                pixel_width, pixel_height = pil_image.size
            scale_x = pixel_width / image_width
            scale_y = pixel_height / image_height
        else:
            pil_image = Image.fromarray(image, mode="RGB")
            scale_x = scale_y = 1.0
        tensors = []
        masks = []
        base_boxes = np.asarray(base["boxes_xyxy"], dtype=np.float32)
        policy = self.ensemble.class_verified_selector
        classifier_metadata = self.classifier.metadata
        view = classifier_metadata.neighbor_mask_inference.views[0]
        try:
            for index in indices:
                box = np.asarray(raw["boxes_xyxy"][index], dtype=np.float32)
                detection = Detection(
                    *box, float(raw["scores"][index]), int(raw["class_ids"][index])
                )
                x1, y1, x2, y2 = classifier_crop_box(
                    detection,
                    image_width,
                    image_height,
                    margin_ratio=classifier_metadata.crop_margin_ratio,
                    crop_mode=classifier_metadata.crop_mode,
                )
                crop = pil_image.crop(
                    (
                        int(np.floor(x1 * scale_x)),
                        int(np.floor(y1 * scale_y)),
                        int(np.ceil(x2 * scale_x)),
                        int(np.ceil(y2 * scale_y)),
                    )
                )
                tensors.append(
                    prepare_rgb(
                        crop,
                        classifier_metadata.input_size,
                        classifier_metadata.mean,
                        classifier_metadata.std,
                        reducing_gap=classifier_metadata.resize_reducing_gap,
                    )
                )
                mask_boxes, target_index = _candidate_context(
                    base_boxes, box, duplicate_iou=policy.candidate_duplicate_iou
                )
                masks.append(
                    classifier_neighbor_ownership_mask(
                        [Detection(*values, 1.0) for values in mask_boxes],
                        target_index,
                        image_width=image_width,
                        image_height=image_height,
                        output_size=classifier_metadata.input_size[0],
                        margin_ratio=classifier_metadata.crop_margin_ratio,
                        distance_bias=view.distance_bias,
                        shared_scale=view.shared_scale,
                    )
                )
        finally:
            if pil_image is not image:
                pil_image.close()
        batch = apply_classifier_background_masks(
            np.stack(tensors).astype(np.float32), np.stack(masks)
        )
        parts = []
        for start in range(0, len(batch), policy.classifier_batch_size):
            (scores,) = self.classifier.runner.run(
                [classifier_metadata.logits_output],
                classifier_metadata.input_name,
                batch[start : start + policy.classifier_batch_size],
            )
            parts.append(np.asarray(scores, dtype=np.float32))
        return np.concatenate(parts)

    def _predict(
        self,
        image: np.ndarray | Image.Image,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        tensor = prepare_rgb(
            image,
            self.metadata.input_size,
            self.metadata.mean,
            self.metadata.std,
            reducing_gap=self.metadata.resize_reducing_gap,
        )[None]
        raw_outputs = self._run_models(tensor)
        rows = [
            detector_output_to_prediction(
                logits, boxes, image_width=image_width, image_height=image_height
            )
            for logits, boxes in raw_outputs
        ]
        members = self.ensemble.members
        fusion = self.ensemble.fusion
        raw = fuse_prediction_rows(
            rows,
            model_weights=[member.weight for member in members],
            score_thresholds=[member.score_threshold for member in members],
            pre_nms_iou_threshold=fusion.pre_nms_iou_threshold,
            maximum_candidates_per_model=fusion.maximum_candidates_per_model,
            cluster_iou_threshold=fusion.cluster_iou_threshold,
        )
        base_policy = self.ensemble.base_selection
        selected = _containment_select(
            raw,
            score_threshold=base_policy.score_threshold,
            iou_threshold=base_policy.nms_iou_threshold,
            containment_threshold=base_policy.containment_threshold,
            group_minimum=base_policy.group_minimum,
        )
        return rows, raw, selected

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        if isinstance(image, Image.Image):
            image_width, image_height = image_original_size(image)
        else:
            image_height, image_width = image.shape[:2]
        rows, raw, selected = self._predict(
            image, image_width=image_width, image_height=image_height
        )
        maximum_area = self.ensemble.maximum_box_area_ratio
        image_area = float(image_width * image_height)
        rows = [
            _filter_prediction_by_area(row, image_area=image_area, maximum_area_ratio=maximum_area)
            for row in rows
        ]
        raw = _filter_prediction_by_area(
            raw, image_area=image_area, maximum_area_ratio=maximum_area
        )
        base_policy = self.ensemble.base_selection
        selected = _containment_select(
            raw,
            score_threshold=base_policy.score_threshold,
            iou_threshold=base_policy.nms_iou_threshold,
            containment_threshold=base_policy.containment_threshold,
            group_minimum=base_policy.group_minimum,
        )
        members = self.ensemble.members
        rows_by_filename = {member.filename: row for member, row in zip(members, rows)}
        consensus_policy = self.ensemble.policy_consensus
        agreement_count = consensus_agreement_count(selected, rows_by_filename, consensus_policy)
        consensus_ambiguous = (
            consensus_policy is not None
            and agreement_count < consensus_policy.minimum_agreeing_policy_count
        )
        union_ambiguous = is_ambiguous(raw, selected, self.ensemble.ambiguity_union)
        requires_class_verification = consensus_ambiguous or union_ambiguous
        refinement = self.ensemble.draft_refinement
        if (
            refinement is not None
            and consensus_ambiguous
            and len(selected["scores"])
            <= refinement.consensus_ambiguity_bypass_maximum_selected_count
            and min(selected["scores"], default=0.0)
            >= refinement.consensus_ambiguity_bypass_minimum_selected_score
        ):
            requires_class_verification = False
        if (
            refinement is not None
            and union_ambiguous
            and not consensus_ambiguous
            and consensus_policy is not None
            and agreement_count == len(consensus_policy.policies) + 1
            and len(selected["scores"])
            <= refinement.unanimous_ambiguity_bypass_maximum_selected_count
            and min(selected["scores"], default=0.0)
            >= refinement.unanimous_ambiguity_bypass_minimum_selected_score
        ):
            requires_class_verification = False
        verification_image = image
        owned_images: list[Image.Image] = []
        restore_source_for_crops = not (
            refinement is not None
            and isinstance(image, Image.Image)
            and image.size != image_original_size(image)
        )
        should_refine = requires_class_verification and (
            refinement is None
            or len(selected["scores"]) <= refinement.ambiguity_refinement_maximum_selected_count
        )
        if refinement is not None:
            should_refine = should_refine or (
                agreement_count <= refinement.maximum_agreeing_policy_count
                and len(selected["scores"]) >= refinement.minimum_selected_count
                and _maximum_aspect_ratio_extremity(selected)
                >= refinement.minimum_selected_box_aspect_ratio_extremity
            )
        if should_refine and isinstance(image, Image.Image):
            previous_selected_count = len(selected["scores"])
            refined = (
                redraft_image(image, refinement.draft_size)
                if refinement is not None
                else restore_original_resolution(image)
            )
            if refined is not image:
                owned_images.append(refined)
                verification_image = refined
                restore_source_for_crops = refinement is None
                rows, raw, selected = self._predict(
                    refined, image_width=image_width, image_height=image_height
                )
                rows = [
                    _filter_prediction_by_area(
                        row, image_area=image_area, maximum_area_ratio=maximum_area
                    )
                    for row in rows
                ]
                raw = _filter_prediction_by_area(
                    raw, image_area=image_area, maximum_area_ratio=maximum_area
                )
                selected = _containment_select(
                    raw,
                    score_threshold=base_policy.score_threshold,
                    iou_threshold=base_policy.nms_iou_threshold,
                    containment_threshold=base_policy.containment_threshold,
                    group_minimum=base_policy.group_minimum,
                )
                rows_by_filename = {member.filename: row for member, row in zip(members, rows)}
                consensus_ambiguous = consensus_is_ambiguous(
                    selected, rows_by_filename, consensus_policy
                )
                union_ambiguous = is_ambiguous(raw, selected, self.ensemble.ambiguity_union)
                requires_class_verification = consensus_ambiguous or union_ambiguous
                if refinement is not None:
                    requires_full_resolution = (
                        refinement.full_resolution_on_selected_count_change
                        and len(selected["scores"]) != previous_selected_count
                    ) or (
                        union_ambiguous
                        and len(selected["scores"])
                        <= refinement.full_resolution_unresolved_ambiguity_maximum_selected_count
                    )
                    if requires_full_resolution:
                        restored = restore_original_resolution(image)
                        if restored is not image:
                            owned_images.append(restored)
                            verification_image = restored
                            restore_source_for_crops = True
                            rows, raw, selected = self._predict(
                                restored,
                                image_width=image_width,
                                image_height=image_height,
                            )
                            rows = [
                                _filter_prediction_by_area(
                                    row,
                                    image_area=image_area,
                                    maximum_area_ratio=maximum_area,
                                )
                                for row in rows
                            ]
                            raw = _filter_prediction_by_area(
                                raw,
                                image_area=image_area,
                                maximum_area_ratio=maximum_area,
                            )
                            selected = _containment_select(
                                raw,
                                score_threshold=base_policy.score_threshold,
                                iou_threshold=base_policy.nms_iou_threshold,
                                containment_threshold=base_policy.containment_threshold,
                                group_minimum=base_policy.group_minimum,
                            )
                            rows_by_filename = {
                                member.filename: row for member, row in zip(members, rows)
                            }
                            requires_class_verification = consensus_is_ambiguous(
                                selected, rows_by_filename, consensus_policy
                            ) or is_ambiguous(raw, selected, self.ensemble.ambiguity_union)
        try:
            if requires_class_verification:
                policy = self.ensemble.class_verified_selector
                indices = [
                    index
                    for index, (score, support) in enumerate(
                        zip(raw["scores"], raw["support_counts"])
                    )
                    if score >= policy.candidate_minimum_score
                    and support >= policy.candidate_minimum_support
                ]
                scores = self._proposal_scores(
                    verification_image,
                    selected,
                    raw,
                    indices,
                    image_width=image_width,
                    image_height=image_height,
                    restore_source_for_crops=restore_source_for_crops,
                )
                entries = []
                for index, values in zip(indices, scores):
                    order = np.argsort(-values, kind="stable")
                    entries.append(
                        {
                            "proposal_index": index,
                            "box": np.asarray(raw["boxes_xyxy"][index], dtype=np.float32),
                            "detector_score": float(raw["scores"][index]),
                            "support_count": int(raw["support_counts"][index]),
                            "predicted_class": int(order[0]),
                            "class_margin": float(values[order[0]] - values[order[1]]),
                        }
                    )
                selected = _class_verified_select(selected, raw, entries, policy)
        finally:
            for owned_image in owned_images:
                owned_image.close()
        kept = [
            index
            for index, box in enumerate(selected["boxes_xyxy"])
            if _area(np.asarray(box, dtype=np.float32)) / (image_width * image_height)
            <= maximum_area
        ]
        detections = [
            Detection(
                *selected["boxes_xyxy"][index],
                float(selected["scores"][index]),
                int(selected["class_ids"][index]),
            )
            for index in kept
        ]
        return DetectionResult(detections=detections)
