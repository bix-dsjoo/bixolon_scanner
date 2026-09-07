from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ..contracts.errors import ModelExecutionError, ProviderInitializationError
from ..contracts.image import ORIGINAL_SIZE_INFO_KEY
from ..contracts.model_package import (
    ClassifierMetadata,
    CountVerifierMetadata,
    DetectorMetadata,
    ModelPackage,
)
from ..contracts.runtime_package_v2 import DetectorCrowdingPolicyMetadata
from ..pipeline.ports import ClassificationResult, Detection, DetectionResult
from .geometry import box_containment, box_iou, nms
from .imaging import image_original_size
from .onnx_session import ExecutionProvider, OrtRunner, select_provider
from .preprocessing import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)


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


sigmoid = _sigmoid
_nms = nms
_box_iou = box_iou
_box_containment = box_containment
_prepare_rgb = prepare_rgb


def _minimum_normalized_center_distance(detections: list[Detection]) -> float:
    minimum_distance = math.inf
    for index, left in enumerate(detections):
        left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
        left_center_x = (left.x1 + left.x2) * 0.5
        left_center_y = (left.y1 + left.y2) * 0.5
        for right in detections[index + 1 :]:
            right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
            right_center_x = (right.x1 + right.x2) * 0.5
            right_center_y = (right.y1 + right.y2) * 0.5
            distance = math.hypot(
                left_center_x - right_center_x,
                left_center_y - right_center_y,
            )
            scale = max(math.sqrt((left_area + right_area) * 0.5), 1e-9)
            minimum_distance = min(minimum_distance, distance / scale)
    return minimum_distance


def _selected_area_fraction(
    detections: list[Detection], *, image_width: int, image_height: int
) -> float:
    image_area = float(image_width * image_height)
    return sum(
        (detection.x2 - detection.x1) * (detection.y2 - detection.y1) / image_area
        for detection in detections
    )


def detector_crowding_requires_recapture(
    candidates: list[Detection],
    *,
    selected_detections: list[Detection],
    image_width: int,
    image_height: int,
    nms_iou_threshold: float,
    policy: DetectorCrowdingPolicyMetadata,
) -> bool:
    """Detect detector-query geometry associated with severe object occlusion."""
    if not candidates:
        return False

    large_proposals = _nms(
        [
            candidate
            for candidate in candidates
            if candidate.score >= policy.large_proposal_score_threshold
        ],
        nms_iou_threshold,
    )
    image_area = float(image_width * image_height)
    largest_proposal = max(
        large_proposals,
        key=lambda candidate: (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1),
        default=None,
    )
    if largest_proposal is not None:
        maximum_area_ratio = (
            (largest_proposal.x2 - largest_proposal.x1)
            * (largest_proposal.y2 - largest_proposal.y1)
            / image_area
        )
        if maximum_area_ratio >= policy.large_proposal_minimum_area_ratio:
            corroboration = policy.large_proposal_corroboration
            if corroboration is None:
                return True
            contained_query_count = sum(
                _box_containment(largest_proposal, candidate) >= policy.query_cluster_iou_threshold
                for candidate in candidates
            )
            clustered_query_count = sum(
                _box_iou(largest_proposal, candidate) >= policy.query_cluster_iou_threshold
                for candidate in candidates
            )
            selected_center_count = sum(
                largest_proposal.x1 <= (detection.x1 + detection.x2) * 0.5 <= largest_proposal.x2
                and largest_proposal.y1
                <= (detection.y1 + detection.y2) * 0.5
                <= largest_proposal.y2
                for detection in selected_detections
            )
            if (
                contained_query_count - clustered_query_count
                >= corroboration.query_containment_surplus_minimum
                or len(selected_detections) <= corroboration.selected_count_maximum
                and selected_center_count >= corroboration.selected_center_minimum
            ):
                return True

    selected = _nms(candidates, nms_iou_threshold)
    minimum_normalized_center_distance = _minimum_normalized_center_distance(selected)
    if minimum_normalized_center_distance > policy.proximity_maximum_normalized_center_distance:
        return False

    anchors = _nms(candidates, policy.query_cluster_iou_threshold)
    duplicate_count = sum(
        max(
            sum(
                _box_iou(anchor, candidate) >= policy.query_cluster_iou_threshold
                for candidate in candidates
            )
            - 1,
            0,
        )
        for anchor in anchors
    )
    duplicate_fraction = duplicate_count / len(candidates)
    return duplicate_fraction >= policy.query_duplicate_minimum_fraction


class OnnxDetector:
    def __init__(
        self,
        model_path: Path,
        metadata: DetectorMetadata,
        provider: ExecutionProvider,
        cuda_dll_dir: Path | None = None,
        *,
        crowding_policy: DetectorCrowdingPolicyMetadata | None = None,
        enable_cuda_graph: bool = False,
        detector_class_count: int | None = None,
        class_agnostic: bool = False,
        cpu_intra_op_threads: int = 0,
        openvino_cache_dir: Path | None = None,
    ):
        if enable_cuda_graph and detector_class_count is None:
            raise ValueError("CUDA graph detector requires the packaged class count")
        self.metadata = metadata
        self.crowding_policy = crowding_policy
        self.class_agnostic = class_agnostic
        output_shapes = (
            {
                metadata.logits_output: (1, metadata.max_queries, detector_class_count),
                metadata.boxes_output: (1, metadata.max_queries, 4),
            }
            if enable_cuda_graph
            else None
        )
        self.runner = OrtRunner(
            model_path,
            provider,
            cuda_dll_dir,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_output_shapes=output_shapes,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            dummy,
        )

    def _postprocess_detection_outputs(
        self,
        logits: np.ndarray,
        boxes: np.ndarray,
        *,
        original_width: int,
        original_height: int,
    ) -> tuple[DetectionResult, list[Detection], int, int]:
        logits = np.asarray(logits)
        boxes = np.asarray(boxes)
        if (
            logits.ndim not in {1, 2}
            or (logits.ndim == 2 and logits.shape[1] == 0)
            or boxes.shape != (len(logits), 4)
            or not np.isfinite(logits).all()
            or not np.isfinite(boxes).all()
        ):
            raise ModelExecutionError
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
                    # Preserve the detector's generic class prediction even when containment
                    # suppression is class agnostic.  The NMS policy flag controls only whether
                    # class equality participates in suppression; downstream diagnostics and
                    # model-level fusion still need the detector output that was actually run.
                    class_id = (
                        None
                        if getattr(self, "class_agnostic", False) or logits.ndim != 2
                        else int(np.argmax(logits[index]))
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
        crowding_policy = getattr(self, "crowding_policy", None)
        crowding_candidates = (
            convert(np.flatnonzero(scores >= crowding_policy.candidate_score_threshold))
            if crowding_policy is not None
            else []
        )
        return (
            DetectionResult(
                detections,
                raw_saturated,
                uncertain_candidate_count=uncertain_candidate_count,
                uncertain_candidate_scores=tuple(sorted(uncertain_candidate_scores, reverse=True)),
            ),
            crowding_candidates,
            original_width,
            original_height,
        )

    def _prepare_detection_tensor(
        self, image: np.ndarray | Image.Image
    ) -> tuple[np.ndarray, int, int]:
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
        )
        return tensor, original_width, original_height

    def _detect_prepared_tensor(
        self,
        tensor: np.ndarray,
        *,
        original_width: int,
        original_height: int,
    ) -> tuple[DetectionResult, list[Detection], int, int]:
        logits, boxes = self.runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            tensor[None],
        )
        return self._postprocess_detection_outputs(
            np.asarray(logits)[0],
            np.asarray(boxes)[0],
            original_width=original_width,
            original_height=original_height,
        )

    def _detect_once(
        self, image: np.ndarray | Image.Image
    ) -> tuple[DetectionResult, list[Detection], int, int]:
        tensor, original_width, original_height = self._prepare_detection_tensor(image)
        return self._detect_prepared_tensor(
            tensor,
            original_width=original_width,
            original_height=original_height,
        )

    @staticmethod
    def _rotate_for_recovery(
        image: np.ndarray | Image.Image, degrees: Literal[90, 180, 270]
    ) -> np.ndarray | Image.Image:
        if isinstance(image, Image.Image):
            method = {
                90: Image.Transpose.ROTATE_90,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_270,
            }[degrees]
            rotated = image.transpose(method)
            original_width, original_height = image_original_size(image)
            rotated.info[ORIGINAL_SIZE_INFO_KEY] = (
                (original_height, original_width)
                if degrees in {90, 270}
                else (original_width, original_height)
            )
            return rotated
        return np.ascontiguousarray(np.rot90(image, k=degrees // 90))

    @staticmethod
    def _restore_recovery_coordinates(
        detections: list[Detection],
        *,
        degrees: Literal[90, 180, 270],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        if degrees == 90:
            return [
                Detection(
                    image_width - item.y2,
                    item.x1,
                    image_width - item.y1,
                    item.x2,
                    item.score,
                    item.class_id,
                )
                for item in detections
            ]
        if degrees == 180:
            return [
                Detection(
                    image_width - item.x2,
                    image_height - item.y2,
                    image_width - item.x1,
                    image_height - item.y1,
                    item.score,
                    item.class_id,
                )
                for item in detections
            ]
        return [
            Detection(
                item.y1,
                image_height - item.x2,
                item.y2,
                image_height - item.x1,
                item.score,
                item.class_id,
            )
            for item in detections
        ]

    @staticmethod
    def _fully_matches_recovery(
        primary: list[Detection], recovery: list[Detection], threshold: float
    ) -> bool:
        adjacency = [
            [index for index, other in enumerate(recovery) if _box_iou(box, other) >= threshold]
            for box in primary
        ]
        matched_primary = [-1] * len(recovery)

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

        return all(augment(index, set()) for index in range(len(primary)))

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        base_tensor, original_width, original_height = self._prepare_detection_tensor(image)
        result, crowding_candidates, _, _ = self._detect_prepared_tensor(
            base_tensor,
            original_width=original_width,
            original_height=original_height,
        )
        crowding_policy = getattr(self, "crowding_policy", None)
        if crowding_policy is None:
            return result
        if original_width / original_height < crowding_policy.minimum_image_aspect_ratio:
            return result

        uncertain_candidate_count = result.uncertain_candidate_count
        if detector_crowding_requires_recapture(
            crowding_candidates,
            selected_detections=result.detections,
            image_width=original_width,
            image_height=original_height,
            nms_iou_threshold=self.metadata.nms_iou_threshold,
            policy=crowding_policy,
        ):
            uncertain_candidate_count += 1

        if (
            uncertain_candidate_count == 0
            and len(result.detections) >= crowding_policy.rotation_recovery_minimum_selected_count
            and (
                crowding_policy.rotation_recovery_maximum_selected_count is None
                or len(result.detections)
                <= crowding_policy.rotation_recovery_maximum_selected_count
            )
            and _selected_area_fraction(
                result.detections,
                image_width=original_width,
                image_height=original_height,
            )
            >= crowding_policy.rotation_recovery_minimum_selected_area_fraction
            and _minimum_normalized_center_distance(result.detections)
            >= crowding_policy.rotation_recovery_minimum_normalized_center_distance
        ):
            required_count = (
                len(result.detections) + crowding_policy.rotation_recovery_minimum_count_gain
            )

            def recovers_additional_object(
                degrees: Literal[90, 180, 270], recovered: DetectionResult
            ) -> bool:
                mapped_recovery = self._restore_recovery_coordinates(
                    recovered.detections,
                    degrees=degrees,
                    image_width=original_width,
                    image_height=original_height,
                )
                return len(mapped_recovery) >= required_count and self._fully_matches_recovery(
                    result.detections,
                    mapped_recovery,
                    crowding_policy.rotation_recovery_agreement_iou_threshold,
                )

            for degrees in crowding_policy.rotation_recovery_degrees:
                if degrees == 180:
                    recovered, _, _, _ = self._detect_prepared_tensor(
                        np.ascontiguousarray(base_tensor[:, ::-1, ::-1]),
                        original_width=original_width,
                        original_height=original_height,
                    )
                else:
                    rotated = self._rotate_for_recovery(image, degrees)
                    try:
                        recovered, _, _, _ = self._detect_once(rotated)
                    finally:
                        if isinstance(rotated, Image.Image):
                            rotated.close()
                if recovers_additional_object(degrees, recovered):
                    uncertain_candidate_count += 1
                    break

        return DetectionResult(
            result.detections,
            result.capacity_saturated,
            uncertain_candidate_count=uncertain_candidate_count,
            uncertain_candidate_scores=result.uncertain_candidate_scores,
        )


class OnnxClassifier:
    def __init__(
        self,
        model_path: Path,
        metadata: ClassifierMetadata,
        provider: ExecutionProvider,
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
        provider: ExecutionProvider,
        cuda_dll_dir: Path | None = None,
        *,
        cpu_intra_op_threads: int = 0,
        openvino_cache_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(
            model_path,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.runner.run([self.metadata.logits_output], self.metadata.input_name, dummy)

    def close(self) -> None:
        self.runner.close()

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
    def __init__(
        self,
        detector: OnnxDetector,
        verifier: OnnxCountVerifier,
        *,
        parallel_verification: bool = False,
    ):
        self.detector = detector
        self.verifier = verifier
        self.version = detector.version
        self._verification_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="bixolon-presence")
            if parallel_verification
            else None
        )

    def warmup(self) -> None:
        warmup = getattr(self.detector, "warmup", None)
        if callable(warmup):
            warmup()
        self.verifier.warmup()

    def close(self) -> None:
        self._shutdown_verification_executor()
        close = getattr(self.verifier, "close", None)
        if callable(close):
            close()
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()

    def replace_verifier(
        self,
        verifier: OnnxCountVerifier,
        *,
        parallel_verification: bool,
    ) -> None:
        self._shutdown_verification_executor()
        previous = self.verifier
        self.verifier = verifier
        self._verification_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="bixolon-presence")
            if parallel_verification
            else None
        )
        close = getattr(previous, "close", None)
        if callable(close):
            close()

    def _shutdown_verification_executor(self) -> None:
        if self._verification_executor is not None:
            self._verification_executor.shutdown(wait=True, cancel_futures=True)
            self._verification_executor = None

    def detach_parallel_verification(self) -> bool:
        """Transfer speculative-verifier ownership to another detector adapter."""
        enabled = self._verification_executor is not None
        self._shutdown_verification_executor()
        return enabled

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        verification: Future[tuple[int, float]] | None = None
        if self._verification_executor is not None:
            verification = self._verification_executor.submit(self.verifier.verify, image)
        try:
            result = self.detector.detect(image)
        except Exception:
            if verification is not None:
                try:
                    verification.result()
                except Exception:
                    pass
            raise
        if result.capacity_saturated or result.uncertain_candidate_count or not result.detections:
            if verification is not None:
                try:
                    verification.result()
                except Exception:
                    # The speculative result is not part of an existing hard-recapture
                    # decision. A normal detection still propagates verifier failures.
                    pass
            comparison_mode = self.verifier.metadata.comparison_mode
            verified_count = (
                int(bool(result.detections))
                if comparison_mode == "object_presence"
                else len(result.detections)
            )
            confidence = 1.0
        elif verification is not None:
            verified_count, confidence = verification.result()
        else:
            verified_count, confidence = self.verifier.verify(image)
        return DetectionResult(
            detections=result.detections,
            capacity_saturated=result.capacity_saturated,
            verified_count=verified_count,
            count_confidence=confidence,
            uncertain_candidate_count=result.uncertain_candidate_count,
            uncertain_candidate_scores=result.uncertain_candidate_scores,
            refinement_executed=result.refinement_executed,
            detector_class_ids=result.detector_class_ids,
            detector_class_support_counts=result.detector_class_support_counts,
        )


def build_onnx_adapters(
    model_package: ModelPackage,
    provider_mode: Literal["auto", "cuda", "cpu"],
    *,
    cuda_dll_dir: Path | None = None,
):
    provider = select_provider(provider_mode)

    def create(selected_provider: ExecutionProvider):
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
