from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ..contracts.errors import ModelExecutionError, ProviderInitializationError
from ..contracts.model_package import (
    ClassifierMetadata,
    CountVerifierMetadata,
    DetectorMetadata,
    ModelPackage,
)
from ..pipeline.ports import ClassificationResult, Detection, DetectionResult
from .imaging import image_original_size
from .onnx_session import OrtRunner, select_provider


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _softmax_rows(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)


def _nms(
    detections: list[Detection],
    threshold: float,
    containment_threshold: float | None = None,
    class_aware_containment: bool = False,
) -> list[Detection]:
    ordered = sorted(detections, key=lambda detection: detection.score, reverse=True)
    kept: list[Detection] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        remaining: list[Detection] = []
        current_area = max(0.0, current.x2 - current.x1) * max(0.0, current.y2 - current.y1)
        for candidate in ordered:
            ix1 = max(current.x1, candidate.x1)
            iy1 = max(current.y1, candidate.y1)
            ix2 = min(current.x2, candidate.x2)
            iy2 = min(current.y2, candidate.y2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            candidate_area = max(0.0, candidate.x2 - candidate.x1) * max(
                0.0, candidate.y2 - candidate.y1
            )
            union = current_area + candidate_area - intersection
            smaller_area = min(current_area, candidate_area)
            contained = (
                containment_threshold is not None
                and smaller_area > 0.0
                and intersection / smaller_area >= containment_threshold
                and (
                    not class_aware_containment
                    or current.class_id is not None
                    and current.class_id == candidate.class_id
                )
            )
            if (union <= 0.0 or intersection / union <= threshold) and not contained:
                remaining.append(candidate)
        ordered = remaining
    return kept


def _box_iou(left: Detection, right: Detection) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


sigmoid = _sigmoid
nms = _nms
box_iou = _box_iou


def _prepare_rgb(
    image: np.ndarray | Image.Image,
    size: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    reducing_gap: float | None = None,
) -> np.ndarray:
    source = image if isinstance(image, Image.Image) else Image.fromarray(image, mode="RGB")
    pil = source.resize(
        (size[1], size[0]),
        Image.Resampling.BILINEAR,
        reducing_gap=reducing_gap,
    )
    tensor = np.asarray(pil, dtype=np.float32)
    tensor /= np.float32(255.0)
    if any(value != 0.0 for value in mean):
        tensor -= np.asarray(mean, dtype=np.float32)
    if any(value != 1.0 for value in std):
        tensor /= np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1)))


prepare_rgb = _prepare_rgb


def classifier_neighbor_ownership_mask(
    detections: list[Detection],
    target_index: int,
    *,
    image_width: int,
    image_height: int,
    output_size: int,
    margin_ratio: float,
    distance_bias: float,
    shared_scale: bool,
) -> np.ndarray:
    if not 0 <= target_index < len(detections):
        raise ValueError("target detection index is outside the detection list")
    if output_size < 1 or margin_ratio < 0.0 or distance_bias < 0.0:
        raise ValueError("mask size, margin, and distance bias must be non-negative")
    target = detections[target_index]
    target_width = target.x2 - target.x1
    target_height = target.y2 - target.y1
    if target_width <= 0.0 or target_height <= 0.0:
        raise ValueError("target detection box is empty")
    crop_x1 = max(0.0, target.x1 - target_width * margin_ratio)
    crop_y1 = max(0.0, target.y1 - target_height * margin_ratio)
    crop_x2 = min(float(image_width), target.x2 + target_width * margin_ratio)
    crop_y2 = min(float(image_height), target.y2 + target_height * margin_ratio)
    x = crop_x1 + (np.arange(output_size) + 0.5) * (crop_x2 - crop_x1) / output_size
    y = crop_y1 + (np.arange(output_size) + 0.5) * (crop_y2 - crop_y1) / output_size
    grid_x, grid_y = np.meshgrid(x, y)
    target_center_x = (target.x1 + target.x2) / 2.0
    target_center_y = (target.y1 + target.y2) / 2.0
    target_distance = ((grid_x - target_center_x) / max(target_width / 2.0, 1e-12)) ** 2 + (
        (grid_y - target_center_y) / max(target_height / 2.0, 1e-12)
    ) ** 2
    mask = np.zeros((output_size, output_size), dtype=bool)
    for index, other in enumerate(detections):
        if index == target_index:
            continue
        other_width = other.x2 - other.x1
        other_height = other.y2 - other.y1
        if other_width <= 0.0 or other_height <= 0.0:
            continue
        inside = (
            (grid_x >= other.x1)
            & (grid_x <= other.x2)
            & (grid_y >= other.y1)
            & (grid_y <= other.y2)
        )
        width_scale = target_width if shared_scale else other_width
        height_scale = target_height if shared_scale else other_height
        other_distance = (
            (grid_x - (other.x1 + other.x2) / 2.0) / max(width_scale / 2.0, 1e-12)
        ) ** 2 + ((grid_y - (other.y1 + other.y2) / 2.0) / max(height_scale / 2.0, 1e-12)) ** 2
        mask |= inside & (other_distance + distance_bias < target_distance)
    return mask


def apply_classifier_background_masks(batch: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if batch.ndim != 4 or masks.shape != (len(batch), batch.shape[2], batch.shape[3]):
        raise ValueError("classifier batch and neighbor masks are not aligned")
    output = batch.copy()
    borders = np.concatenate(
        (
            batch[:, :, 0, :],
            batch[:, :, -1, :],
            batch[:, :, 1:-1, 0],
            batch[:, :, 1:-1, -1],
        ),
        axis=2,
    )
    background = np.median(borders, axis=2)
    for channel in range(batch.shape[1]):
        output[:, channel] = np.where(
            masks,
            background[:, channel, None, None],
            batch[:, channel],
        )
    return output


def classifier_crop_box(
    detection: Detection,
    image_width: int,
    image_height: int,
    *,
    margin_ratio: float,
    crop_mode: str,
) -> tuple[int, int, int, int]:
    margin_x = (detection.x2 - detection.x1) * margin_ratio
    margin_y = (detection.y2 - detection.y1) * margin_ratio
    x1 = max(0.0, detection.x1 - margin_x)
    y1 = max(0.0, detection.y1 - margin_y)
    x2 = min(float(image_width), detection.x2 + margin_x)
    y2 = min(float(image_height), detection.y2 + margin_y)
    if crop_mode == "square_context":
        side = min(max(x2 - x1, y2 - y1), float(image_width), float(image_height))
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        x1 = min(max(0.0, center_x - side * 0.5), image_width - side)
        y1 = min(max(0.0, center_y - side * 0.5), image_height - side)
        x2 = x1 + side
        y2 = y1 + side
    elif crop_mode != "box_resize":
        raise ValueError(f"unsupported classifier crop mode: {crop_mode}")
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(image_width, int(np.ceil(x2))),
        min(image_height, int(np.ceil(y2))),
    )


class OnnxDetector:
    def __init__(
        self,
        model_path: Path,
        metadata: DetectorMetadata,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(model_path, provider, cuda_dll_dir)
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            dummy,
        )

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        if isinstance(image, Image.Image):
            original_width, original_height = image_original_size(image)
        else:
            original_height, original_width = image.shape[:2]
        tensor = _prepare_rgb(
            image,
            self.metadata.input_size,
            self.metadata.mean,
            self.metadata.std,
            reducing_gap=self.metadata.resize_reducing_gap,
        )[None]
        logits, boxes = self.runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            tensor,
        )
        logits = np.asarray(logits)[0]
        boxes = np.asarray(boxes)[0]
        if logits.ndim == 1:
            scores = _sigmoid(logits)
        else:
            scores = _sigmoid(logits).max(axis=-1)
        selected_indices = np.flatnonzero(scores >= self.metadata.score_threshold)
        raw_saturated = len(selected_indices) >= self.metadata.max_queries

        def convert(indices) -> list[Detection]:
            converted: list[Detection] = []
            for index in indices:
                cx, cy, width, height = [float(value) for value in boxes[index]]
                x1 = max(0.0, (cx - width / 2.0) * original_width)
                y1 = max(0.0, (cy - height / 2.0) * original_height)
                x2 = min(float(original_width), (cx + width / 2.0) * original_width)
                y2 = min(float(original_height), (cy + height / 2.0) * original_height)
                if x2 > x1 and y2 > y1:
                    pixel_width = x2 - x1
                    pixel_height = y2 - y1
                    aspect_limit = getattr(self.metadata, "max_object_aspect_ratio", None)
                    if (
                        aspect_limit is not None
                        and max(pixel_width / pixel_height, pixel_height / pixel_width)
                        > aspect_limit
                    ):
                        continue
                    class_aware = getattr(self.metadata, "nms_class_aware_containment", False)
                    class_id = (
                        int(np.argmax(logits[index])) if class_aware and logits.ndim == 2 else None
                    )
                    converted.append(Detection(x1, y1, x2, y2, float(scores[index]), class_id))
            return converted

        containment_threshold = getattr(self.metadata, "nms_containment_threshold", None)
        class_aware_containment = getattr(self.metadata, "nms_class_aware_containment", False)
        detections = _nms(
            convert(selected_indices),
            self.metadata.nms_iou_threshold,
            containment_threshold,
            class_aware_containment,
        )
        uncertain_candidate_count = 0
        uncertain_candidate_scores: list[float] = []
        if self.metadata.uncertainty_score_threshold is not None:
            shadow_indices = np.flatnonzero(scores >= self.metadata.uncertainty_score_threshold)
            shadow = _nms(
                convert(shadow_indices),
                self.metadata.nms_iou_threshold,
                containment_threshold,
                class_aware_containment,
            )
            for candidate in shadow:
                if candidate.score >= self.metadata.score_threshold:
                    continue
                candidate_area_ratio = (
                    (candidate.x2 - candidate.x1)
                    * (candidate.y2 - candidate.y1)
                    / float(original_width * original_height)
                )
                if candidate_area_ratio < self.metadata.uncertainty_min_area_ratio:
                    continue
                overlaps = [_box_iou(candidate, accepted) for accepted in detections]
                if not overlaps or max(overlaps) < self.metadata.uncertainty_match_iou_threshold:
                    uncertain_candidate_count += 1
                    uncertain_candidate_scores.append(candidate.score)
        return DetectionResult(
            detections,
            raw_saturated,
            uncertain_candidate_count=uncertain_candidate_count,
            uncertain_candidate_scores=tuple(sorted(uncertain_candidate_scores, reverse=True)),
        )


class OnnxClassifier:
    def __init__(
        self,
        model_path: Path,
        metadata: ClassifierMetadata,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(model_path, provider, cuda_dll_dir)
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        for batch_size in self.metadata.warmup_batch_sizes:
            dummy = np.zeros((batch_size, 3, height, width), dtype=np.float32)
            staged = self.metadata.staged_inference
            neighbor_mask = self.metadata.neighbor_mask_inference
            if staged is None:
                multiplier = len(neighbor_mask.views) if neighbor_mask is not None else 1
                values = np.concatenate([dummy] * multiplier, axis=0)
                self.runner.run([self.metadata.logits_output], self.metadata.input_name, values)
            else:
                affine = self._view_affine(staged.first_view, batch_size)
                self.runner.run_inputs(
                    [self.metadata.logits_output],
                    {
                        self.metadata.input_name: dummy,
                        staged.affine_input_name: affine,
                    },
                )

    def _view_affine(self, name: str, batch_size: int) -> np.ndarray:
        staged = self.metadata.staged_inference
        if staged is None:
            raise ModelExecutionError
        matrices = {view.name: view.affine for view in staged.views}
        return np.repeat(np.asarray(matrices[name], dtype=np.float32)[None], batch_size, axis=0)

    def _run_view(self, batch: np.ndarray, indices: np.ndarray, name: str) -> np.ndarray:
        return self._run_views(batch, indices, [name])[0]

    def _run_views(self, batch: np.ndarray, indices: np.ndarray, names: list[str]) -> np.ndarray:
        staged = self.metadata.staged_inference
        if staged is None:
            raise ModelExecutionError
        selected_once = np.asarray(batch[indices], dtype=np.float32)
        selected = np.concatenate([selected_once] * len(names), axis=0)
        matrices = np.concatenate([self._view_affine(name, len(indices)) for name in names], axis=0)
        (logits,) = self.runner.run_inputs(
            [self.metadata.logits_output],
            {
                self.metadata.input_name: selected,
                staged.affine_input_name: matrices,
            },
        )
        values = np.asarray(logits, dtype=np.float32)
        return values.reshape(len(names), len(indices), values.shape[-1])

    def _aggregate_ranking_views(self, values: np.ndarray, method: str) -> np.ndarray:
        if method == "mean_logits":
            return values.mean(axis=0)
        probabilities = np.stack(
            [_softmax_rows(view, self.metadata.temperature) for view in values]
        )
        if method == "mean_probability":
            return probabilities.mean(axis=0)
        if method == "maximum_probability":
            return probabilities.max(axis=0)
        orders = np.argsort(-values, axis=2, kind="stable")
        ranks = np.empty_like(orders)
        rows = np.arange(values.shape[1])[:, None]
        class_ranks = np.arange(values.shape[2])[None, :]
        for view_index in range(values.shape[0]):
            ranks[view_index, rows, orders[view_index]] = class_ranks
        if method == "reciprocal_rank":
            return (1.0 / (ranks + 1.0)).mean(axis=0)
        if method == "top3_vote":
            return (ranks < 3).mean(axis=0) + probabilities.mean(axis=0) * 1e-3
        raise ModelExecutionError

    def _staged_classify(self, batch: np.ndarray) -> ClassificationResult:
        staged = self.metadata.staged_inference
        if staged is None:
            raise ModelExecutionError
        all_indices = np.arange(len(batch), dtype=np.int64)
        cached_full_views: dict[str, np.ndarray] = {}
        missing_final_values: np.ndarray | None = None
        if staged.early_approval_threshold >= 1.0:
            final_values = self._run_views(batch, all_indices, list(staged.final_views))
            cached_full_views.update(
                {name: final_values[index] for index, name in enumerate(staged.final_views)}
            )
            first_logits = cached_full_views[staged.first_view]
            final_logits = final_values.mean(axis=0)
            ambiguous_indices = all_indices
            missing_final_views = [name for name in staged.final_views if name != staged.first_view]
        else:
            first_logits = self._run_view(batch, all_indices, staged.first_view)
            cached_full_views[staged.first_view] = first_logits
            first_probabilities = _softmax_rows(first_logits, self.metadata.temperature)
            early = first_probabilities.max(axis=1) >= staged.early_approval_threshold
            final_logits = first_logits.copy()
            ambiguous_indices = np.flatnonzero(~early)
            missing_final_views = [name for name in staged.final_views if name != staged.first_view]
            if len(ambiguous_indices):
                final_sum = first_logits[ambiguous_indices].copy()
                if missing_final_views:
                    missing_final_values = self._run_views(
                        batch, ambiguous_indices, missing_final_views
                    )
                    final_sum += missing_final_values.sum(axis=0)
                final_logits[ambiguous_indices] = final_sum / len(staged.final_views)

        final_probabilities = _softmax_rows(final_logits, self.metadata.temperature)
        if staged.approval_metric == "inverse_entropy":
            approval_scores = np.sum(
                final_probabilities * np.log(final_probabilities.clip(1e-12)), axis=1
            )
        else:
            approval_scores = final_probabilities.max(axis=1)
        approval_threshold = (
            self.metadata.approval_threshold
            if staged.approval_threshold is None
            else staged.approval_threshold
        )
        unknown_indices = np.flatnonzero(approval_scores < approval_threshold)
        ranking_logits = final_logits.copy()
        if len(unknown_indices) and staged.ranking_aggregation != "mean_logits":
            cached_views = {
                name: values[unknown_indices] for name, values in cached_full_views.items()
            }
            if missing_final_values is not None and not cached_full_views.keys() >= set(
                staged.final_views
            ):
                positions = np.full(len(batch), -1, dtype=np.int64)
                positions[ambiguous_indices] = np.arange(len(ambiguous_indices))
                unknown_positions = positions[unknown_indices]
                for view_index, name in enumerate(missing_final_views):
                    cached_views[name] = missing_final_values[view_index, unknown_positions]
            missing_ranking_views = [name for name in staged.top3_views if name not in cached_views]
            if missing_ranking_views:
                extra_values = self._run_views(
                    batch,
                    unknown_indices,
                    missing_ranking_views,
                )
                cached_views.update(
                    {name: extra_values[index] for index, name in enumerate(missing_ranking_views)}
                )
            ranking_views = np.stack([cached_views[name] for name in staged.top3_views])
            ranking_logits[unknown_indices] = self._aggregate_ranking_views(
                ranking_views,
                staged.ranking_aggregation,
            )
        missing_top3_views = [name for name in staged.top3_views if name not in staged.final_views]
        if (
            len(unknown_indices)
            and staged.ranking_aggregation == "mean_logits"
            and missing_top3_views
        ):
            ranking_sum = final_logits[unknown_indices] * len(staged.final_views)
            ranking_sum += self._run_views(batch, unknown_indices, missing_top3_views).sum(axis=0)
            ranking_logits[unknown_indices] = ranking_sum / len(staged.top3_views)
        top3_safety_scores = None
        if staged.top3_safety_metric is not None:
            top3_safety_scores = np.zeros(len(batch), dtype=np.float32)
            if len(unknown_indices):
                ranking_probabilities = _softmax_rows(
                    ranking_logits[unknown_indices], self.metadata.temperature
                )
                top3_safety_scores[unknown_indices] = np.sum(
                    ranking_probabilities * np.log(ranking_probabilities.clip(1e-12)), axis=1
                )
        return ClassificationResult(
            logits=final_logits,
            ranking_logits=ranking_logits,
            approval_scores=approval_scores.astype(np.float32),
            top3_safety_scores=top3_safety_scores,
        )

    def _neighbor_mask_classify(
        self,
        batch: np.ndarray,
        detections: list[Detection],
        *,
        image_width: int,
        image_height: int,
    ) -> ClassificationResult:
        policy = self.metadata.neighbor_mask_inference
        if policy is None:
            raise ModelExecutionError
        view_batches = []
        for view in policy.views:
            masks = np.stack(
                [
                    classifier_neighbor_ownership_mask(
                        detections,
                        index,
                        image_width=image_width,
                        image_height=image_height,
                        output_size=batch.shape[-1],
                        margin_ratio=self.metadata.crop_margin_ratio,
                        distance_bias=view.distance_bias,
                        shared_scale=view.shared_scale,
                    )
                    for index in range(len(detections))
                ]
            )
            view_batches.append(apply_classifier_background_masks(batch, masks))
        combined = np.concatenate(view_batches, axis=0).astype(np.float32, copy=False)
        (raw_logits,) = self.runner.run(
            [self.metadata.logits_output], self.metadata.input_name, combined
        )
        values = np.asarray(raw_logits, dtype=np.float32).reshape(
            len(policy.views), len(detections), -1
        )
        if policy.logit_quantum is not None:
            values = (
                np.round((values + policy.logit_phase) / policy.logit_quantum)
                * policy.logit_quantum
                - policy.logit_phase
            )
        if policy.tie_break_bias_span:
            values = (
                values
                + np.linspace(
                    0.0,
                    -policy.tie_break_bias_span,
                    values.shape[2],
                    dtype=np.float32,
                )[None, None, :]
            )
        weights = np.asarray([view.weight for view in policy.views], dtype=np.float32)
        logits = np.sum(values * weights[:, None, None], axis=0)
        orders = np.argsort(-values, axis=2, kind="stable")
        ranks = np.empty_like(orders)
        np.put_along_axis(
            ranks,
            orders,
            np.arange(values.shape[2], dtype=orders.dtype)[None, None, :],
            axis=2,
        )
        ranking_logits = np.sum((1.0 / (ranks + 1.0)) * weights[:, None, None], axis=0)
        view_probabilities = np.stack(
            [_softmax_rows(view, self.metadata.temperature) for view in values]
        )
        ranking_logits += np.sum(view_probabilities * weights[:, None, None], axis=0) * 1e-3
        ranking_probabilities = _softmax_rows(ranking_logits, self.metadata.temperature)
        top3_safety_scores = np.sum(
            ranking_probabilities * np.log(ranking_probabilities.clip(1e-12)), axis=1
        )
        if policy.ranking_tie_break_bias_span:
            ranking_logits += np.linspace(
                0.0,
                -policy.ranking_tie_break_bias_span,
                ranking_logits.shape[1],
                dtype=np.float32,
            )[None, :]
        probabilities = _softmax_rows(logits, self.metadata.temperature)
        if policy.approval_metric == "l2_normalized_logit_margin":
            ordered_logits = np.sort(logits, axis=1)
            approval_scores = (ordered_logits[:, -1] - ordered_logits[:, -2]) / np.linalg.norm(
                logits, axis=1
            ).clip(min=1e-12)
        else:
            ordered_probabilities = np.sort(probabilities, axis=1)
            approval_scores = ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
        return ClassificationResult(
            logits=logits,
            ranking_logits=ranking_logits,
            approval_scores=approval_scores.astype(np.float32),
            top3_safety_scores=top3_safety_scores.astype(np.float32),
        )

    def classify(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray | ClassificationResult:
        if isinstance(image, Image.Image):
            pil_image = image
            pixel_width, pixel_height = image.size
            image_width, image_height = image_original_size(image)
            scale_x = pixel_width / image_width
            scale_y = pixel_height / image_height
        else:
            pil_image = Image.fromarray(image, mode="RGB")
            image_height, image_width = image.shape[:2]
            scale_x = scale_y = 1.0
        crops: list[np.ndarray] = []
        for detection in detections:
            x1, y1, x2, y2 = classifier_crop_box(
                detection,
                image_width,
                image_height,
                margin_ratio=self.metadata.crop_margin_ratio,
                crop_mode=getattr(self.metadata, "crop_mode", "box_resize"),
            )
            crop = pil_image.crop(
                (
                    int(np.floor(x1 * scale_x)),
                    int(np.floor(y1 * scale_y)),
                    int(np.ceil(x2 * scale_x)),
                    int(np.ceil(y2 * scale_y)),
                )
            )
            if crop.width == 0 or crop.height == 0:
                raise ModelExecutionError
            crops.append(
                _prepare_rgb(
                    crop,
                    self.metadata.input_size,
                    self.metadata.mean,
                    self.metadata.std,
                    reducing_gap=self.metadata.resize_reducing_gap,
                )
            )
        batch = np.stack(crops).astype(np.float32, copy=False)
        if self.metadata.staged_inference is not None:
            return self._staged_classify(batch)
        if self.metadata.neighbor_mask_inference is not None:
            return self._neighbor_mask_classify(
                batch,
                detections,
                image_width=image_width,
                image_height=image_height,
            )
        (logits,) = self.runner.run([self.metadata.logits_output], self.metadata.input_name, batch)
        return np.asarray(logits, dtype=np.float32)


class OnnxCountVerifier:
    def __init__(
        self,
        model_path: Path,
        metadata: CountVerifierMetadata,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(model_path, provider, cuda_dll_dir)
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.runner.run([self.metadata.logits_output], self.metadata.input_name, dummy)

    def verify(self, image: np.ndarray | Image.Image) -> tuple[int, float]:
        tensor = _prepare_rgb(
            image,
            self.metadata.input_size,
            self.metadata.mean,
            self.metadata.std,
            reducing_gap=self.metadata.resize_reducing_gap,
        )[None]
        (logits,) = self.runner.run([self.metadata.logits_output], self.metadata.input_name, tensor)
        values = np.asarray(logits, dtype=np.float32)
        if values.shape != (1, len(self.metadata.count_labels)):
            raise ModelExecutionError
        scaled = values[0].astype(np.float64) / self.metadata.temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        return self.metadata.count_labels[index], float(probabilities[index])


class CountVerifiedDetector:
    def __init__(self, detector: OnnxDetector, verifier: OnnxCountVerifier):
        self.detector = detector
        self.verifier = verifier
        self.version = detector.version

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        result = self.detector.detect(image)
        verified_count, confidence = self.verifier.verify(image)
        return DetectionResult(
            detections=result.detections,
            capacity_saturated=result.capacity_saturated,
            verified_count=verified_count,
            count_confidence=confidence,
            uncertain_candidate_count=result.uncertain_candidate_count,
            uncertain_candidate_scores=result.uncertain_candidate_scores,
        )


def build_onnx_adapters(
    model_package: ModelPackage,
    provider_mode: Literal["auto", "cuda", "cpu"],
    *,
    cuda_dll_dir: Path | None = None,
):
    provider = select_provider(provider_mode)

    def create(selected_provider: Literal["cuda", "cpu"]):
        classifier = OnnxClassifier(
            model_package.classifier_path,
            model_package.metadata.classifier,
            selected_provider,
            cuda_dll_dir,
        )
        if getattr(model_package.metadata.detector, "ensemble", None) is None:
            detector = OnnxDetector(
                model_package.detector_path,
                model_package.metadata.detector,
                selected_provider,
                cuda_dll_dir,
            )
        else:
            from .bread_zero_error import BreadZeroErrorDetector

            detector = BreadZeroErrorDetector(
                model_package.detector_paths,
                model_package.metadata.detector,
                classifier,
                selected_provider,
                cuda_dll_dir,
            )
        detector.warmup()
        classifier.warmup()
        count_metadata = getattr(model_package.metadata, "count_verifier", None)
        count_path = getattr(model_package, "count_verifier_path", None)
        if count_metadata is not None:
            if count_path is None:
                raise ProviderInitializationError
            count_verifier = OnnxCountVerifier(
                count_path,
                count_metadata,
                selected_provider,
                cuda_dll_dir,
            )
            count_verifier.warmup()
            detector = CountVerifiedDetector(detector, count_verifier)
        return detector, classifier, selected_provider

    try:
        return create(provider)
    except (ProviderInitializationError, ModelExecutionError):
        if provider_mode != "auto" or provider != "cuda":
            raise
        return create("cpu")
