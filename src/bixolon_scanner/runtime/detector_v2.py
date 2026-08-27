from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts.runtime_package_v2 import RuntimePackageV2
from ..pipeline.ports import Detection, DetectionResult
from .bread_zero_error import (
    consensus_agreement_count,
    consensus_is_ambiguous,
    containment_select,
    detector_output_to_prediction,
    filter_prediction_by_area,
    fuse_prediction_rows,
    is_ambiguous,
)
from .imaging import image_original_size, restore_original_resolution
from .onnx import (
    CountVerifiedDetector,
    OnnxCountVerifier,
    OnnxDetector,
    OrtRunner,
    box_iou,
    prepare_rgb,
    sigmoid,
)
from .onnx_session import ExecutionProvider


def _area(box: Detection) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _intersection(left: Detection, right: Detection) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def hierarchical_containment_nms(
    detections: list[Detection],
    *,
    iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
) -> list[Detection]:
    if group_minimum < 2:
        raise ValueError("group_minimum must be at least two")
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    for index, candidate in enumerate(ordered):
        candidate_area = _area(candidate)
        if candidate_area <= 0.0:
            continue
        stronger_inside = [
            other
            for other in ordered[:index]
            if _area(other) > 0.0
            and _intersection(candidate, other) / _area(other) >= containment_threshold
        ]
        suppressed = False
        for current in kept:
            current_area = _area(current)
            intersection = _intersection(current, candidate)
            candidate_inside = intersection / candidate_area >= containment_threshold
            current_inside = (
                current_area > 0.0 and intersection / current_area >= containment_threshold
            )
            same_class = current.class_id is not None and current.class_id == candidate.class_id
            if (
                box_iou(current, candidate) > iou_threshold
                or candidate_inside
                or current_inside
                and (same_class or len(stronger_inside) >= group_minimum)
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept


class CrossScaleOnnxDetector:
    def __init__(
        self,
        package: RuntimePackageV2,
        provider: ExecutionProvider,
        cuda_dll_dir: Path | None = None,
        *,
        cpu_intra_op_threads: int = 0,
        openvino_cache_dir: Path | None = None,
    ):
        if package.metadata.detector_refinement is None:
            raise ValueError("cross-scale detector requires refinement metadata")
        self.package = package
        self.primary_metadata = package.metadata.detector
        self.refinement_metadata = package.metadata.detector_refinement
        self.primary = OrtRunner(
            package.detector_path,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
        self.refinement = OrtRunner(
            package.root / self.refinement_metadata.filename,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
        self.version = self.primary_metadata.version

    def warmup(self) -> None:
        for runner, size in (
            (self.primary, self.primary_metadata.input_size),
            (self.refinement, self.refinement_metadata.input_size),
        ):
            runner.run(
                [self.primary_metadata.logits_output, self.primary_metadata.boxes_output],
                self.primary_metadata.input_name,
                np.zeros((1, 3, size[0], size[1]), dtype=np.float32),
            )

    def _run(
        self,
        image: np.ndarray | Image.Image,
        runner: OrtRunner,
        *,
        input_size: tuple[int, int],
        score_threshold: float,
        containment_threshold: float,
        group_minimum: int,
    ) -> tuple[list[Detection], bool]:
        if isinstance(image, Image.Image):
            width, height = image_original_size(image)
        else:
            height, width = image.shape[:2]
        tensor = prepare_rgb(
            image,
            input_size,
            self.primary_metadata.mean,
            self.primary_metadata.std,
            reducing_gap=self.primary_metadata.resize_reducing_gap,
        )[None]
        logits, boxes = runner.run(
            [self.primary_metadata.logits_output, self.primary_metadata.boxes_output],
            self.primary_metadata.input_name,
            tensor,
        )
        logits = np.asarray(logits)[0]
        boxes = np.asarray(boxes)[0]
        probabilities = sigmoid(logits)
        scores = probabilities.max(axis=-1)
        indices = np.flatnonzero(scores >= score_threshold)
        detections = []
        for index in indices:
            center_x, center_y, box_width, box_height = [float(value) for value in boxes[index]]
            x1 = max(0.0, (center_x - box_width / 2.0) * width)
            y1 = max(0.0, (center_y - box_height / 2.0) * height)
            x2 = min(float(width), (center_x + box_width / 2.0) * width)
            y2 = min(float(height), (center_y + box_height / 2.0) * height)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(x1, y1, x2, y2, float(scores[index]), int(np.argmax(logits[index])))
            )
        selected = hierarchical_containment_nms(
            detections,
            iou_threshold=self.primary_metadata.nms_iou_threshold,
            containment_threshold=containment_threshold,
            group_minimum=group_minimum,
        )
        return selected, len(indices) >= self.primary_metadata.max_queries

    @staticmethod
    def _fully_agree(primary: list[Detection], recovery: list[Detection], threshold: float) -> bool:
        if len(primary) != len(recovery):
            return False
        adjacency = [
            [index for index, other in enumerate(recovery) if box_iou(box, other) >= threshold]
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
        primary, primary_saturated = self._run(
            image,
            self.primary,
            input_size=self.primary_metadata.input_size,
            score_threshold=self.primary_metadata.score_threshold,
            containment_threshold=float(self.primary_metadata.nms_containment_threshold),
            group_minimum=2,
        )
        refinement = self.refinement_metadata
        recovery, recovery_saturated = self._run(
            image,
            self.refinement,
            input_size=refinement.input_size,
            score_threshold=refinement.score_threshold,
            containment_threshold=refinement.containment_threshold,
            group_minimum=refinement.group_minimum,
        )
        disagreement = not self._fully_agree(primary, recovery, refinement.agreement_iou_threshold)
        return DetectionResult(
            detections=primary,
            capacity_saturated=primary_saturated or recovery_saturated,
            uncertain_candidate_count=int(disagreement),
        )


class FixedEnsembleOnnxDetector:
    """Store-independent detector ensemble with a selective image recapture gate."""

    def __init__(
        self,
        package: RuntimePackageV2,
        provider: ExecutionProvider,
        cuda_dll_dir: Path | None = None,
        *,
        cpu_detector_workers: int = 1,
        cpu_intra_op_threads: int = 0,
        openvino_cache_dir: Path | None = None,
    ):
        metadata = package.metadata.detector
        if metadata.ensemble is None:
            raise ValueError("fixed ensemble detector requires ensemble metadata")
        self.package = package
        self.metadata = metadata
        self.ensemble = metadata.ensemble
        output_shapes = {
            metadata.logits_output: (
                1,
                metadata.max_queries,
                package.metadata.detector_class_count,
            ),
            metadata.boxes_output: (1, metadata.max_queries, 4),
        }
        self.runners = [
            OrtRunner(
                package.root / member.filename,
                provider,
                cuda_dll_dir,
                enable_cuda_graph=self.ensemble.cuda_graph_execution,
                cuda_graph_output_shapes=output_shapes,
                cpu_intra_op_threads=cpu_intra_op_threads,
                openvino_cache_dir=openvino_cache_dir,
            )
            for member in self.ensemble.members
        ]
        if not 1 <= cpu_detector_workers <= len(self.runners):
            raise ValueError("CPU detector worker count exceeds the detector ensemble")
        executor_workers = (
            1
            if self.ensemble.selective_cascade is not None
            else cpu_detector_workers
            if provider == "cpu"
            else len(self.runners)
            if self.ensemble.parallel_execution
            else 1
        )
        self.executor = (
            ThreadPoolExecutor(max_workers=executor_workers) if executor_workers > 1 else None
        )
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self._run_models(dummy)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def _run_model(self, runner: OrtRunner, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, boxes = runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            tensor,
        )
        return np.asarray(logits)[0], np.asarray(boxes)[0]

    def _run_models(self, tensor: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.executor is None:
            return [self._run_model(runner, tensor) for runner in self.runners]
        return list(self.executor.map(lambda runner: self._run_model(runner, tensor), self.runners))

    def _select_outputs(
        self,
        outputs: list[tuple[np.ndarray, np.ndarray]],
        *,
        width: int,
        height: int,
    ) -> tuple[dict, int, bool, bool]:
        rows = [
            detector_output_to_prediction(
                logits,
                boxes,
                image_width=width,
                image_height=height,
            )
            for logits, boxes in outputs
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
        image_area = float(width * height)
        maximum_area = self.ensemble.maximum_box_area_ratio
        rows = [
            filter_prediction_by_area(
                row,
                image_area=image_area,
                maximum_area_ratio=maximum_area,
            )
            for row in rows
        ]
        raw = filter_prediction_by_area(
            raw,
            image_area=image_area,
            maximum_area_ratio=maximum_area,
        )
        base = self.ensemble.base_selection
        selected = containment_select(
            raw,
            score_threshold=base.score_threshold,
            iou_threshold=base.nms_iou_threshold,
            containment_threshold=base.containment_threshold,
            group_minimum=base.group_minimum,
        )
        rows_by_filename = {member.filename: row for member, row in zip(members, rows, strict=True)}
        agreement_count = consensus_agreement_count(
            selected,
            rows_by_filename,
            self.ensemble.policy_consensus,
        )
        uncertain = consensus_is_ambiguous(
            selected,
            rows_by_filename,
            self.ensemble.policy_consensus,
        ) or is_ambiguous(raw, selected, self.ensemble.ambiguity_union)
        saturated = any(
            sum(score >= member.score_threshold for score in row["scores"])
            >= self.metadata.max_queries
            for member, row in zip(members, rows, strict=True)
        )
        return selected, agreement_count, uncertain, saturated

    def _predict(
        self,
        image: np.ndarray | Image.Image,
        *,
        width: int,
        height: int,
    ) -> tuple[dict, int, bool, bool]:
        tensor = prepare_rgb(
            image,
            self.metadata.input_size,
            self.metadata.mean,
            self.metadata.std,
            reducing_gap=self.metadata.resize_reducing_gap,
        )[None]
        cascade = self.ensemble.selective_cascade
        if cascade is None:
            return self._select_outputs(
                self._run_models(tensor),
                width=width,
                height=height,
            )
        member_filenames = [member.filename for member in self.ensemble.members]
        primary_index = member_filenames.index(cascade.primary_member_filename)
        primary = self._run_model(self.runners[primary_index], tensor)
        primary_outputs = [primary] * len(self.runners)
        primary_result = self._select_outputs(
            primary_outputs,
            width=width,
            height=height,
        )
        if not self._cascade_requires_secondary(
            primary_result[0], cascade, uncertain=primary_result[2]
        ):
            return primary_result
        outputs = list(primary_outputs)
        for index, runner in enumerate(self.runners):
            if index != primary_index:
                outputs[index] = self._run_model(runner, tensor)
        return self._select_outputs(outputs, width=width, height=height)

    @staticmethod
    def _maximum_aspect_ratio_extremity(selected: dict) -> float:
        boxes = np.asarray(selected["boxes_xyxy"], dtype=np.float64)
        if not len(boxes):
            return 1.0
        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        valid = (widths > 0.0) & (heights > 0.0)
        if not np.any(valid):
            return 1.0
        ratios = widths[valid] / heights[valid]
        return float(np.max(np.maximum(ratios, 1.0 / ratios)))

    @staticmethod
    def _cascade_requires_secondary(selected: dict, cascade, *, uncertain: bool = False) -> bool:
        scores = np.asarray(selected["scores"], dtype=np.float64)
        if len(scores) not in cascade.secondary_trigger_selected_counts:
            return False
        maximum = cascade.secondary_trigger_minimum_score_maximum
        minimum = cascade.secondary_trigger_minimum_score_minimum
        if maximum is None and minimum is None and not cascade.secondary_trigger_on_uncertain:
            return True
        if cascade.secondary_trigger_on_uncertain and uncertain:
            return True
        minimum_score = float(np.min(scores))
        return (maximum is not None and minimum_score <= maximum) or (
            minimum is not None and minimum_score >= minimum
        )

    @classmethod
    def _selective_uncertainty(
        cls,
        selected: dict,
        agreement_count: int,
        uncertain: bool,
        policy,
    ) -> bool:
        if policy.mode != "selective":
            return uncertain
        aspect_ratio = cls._maximum_aspect_ratio_extremity(selected)
        if (
            policy.low_agreement_count_maximum is not None
            and agreement_count <= policy.low_agreement_count_maximum
            and aspect_ratio >= policy.low_agreement_aspect_ratio_minimum
        ):
            return True
        if not uncertain:
            return False
        selected_count = len(selected["scores"])
        return aspect_ratio >= policy.high_aspect_ratio_minimum or (
            policy.dense_selected_count_minimum
            <= selected_count
            <= policy.dense_selected_count_maximum
            and agreement_count >= policy.dense_agreement_count_minimum
            and aspect_ratio >= policy.dense_aspect_ratio_minimum
        )

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        if isinstance(image, Image.Image):
            width, height = image_original_size(image)
        else:
            height, width = image.shape[:2]
        selected, agreement_count, uncertain, saturated = self._predict(
            image,
            width=width,
            height=height,
        )
        refinement = self.ensemble.draft_refinement
        complex_scene = (
            refinement is not None
            and agreement_count <= refinement.maximum_agreeing_policy_count
            and len(selected["scores"]) >= refinement.minimum_selected_count
            and self._maximum_aspect_ratio_extremity(selected)
            >= refinement.minimum_selected_box_aspect_ratio_extremity
        )
        refinement_executed = False
        if (
            isinstance(image, Image.Image)
            and image.size != image_original_size(image)
            and complex_scene
        ):
            restored = restore_original_resolution(image)
            try:
                if restored is not image:
                    refinement_executed = True
                    selected, agreement_count, uncertain, refined_saturated = self._predict(
                        restored,
                        width=width,
                        height=height,
                    )
                    saturated = saturated or refined_saturated
            finally:
                if restored is not image:
                    restored.close()
        ambiguity = self.package.metadata.detector_ambiguity
        uncertain = self._selective_uncertainty(
            selected,
            agreement_count,
            uncertain,
            ambiguity,
        )
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                selected["boxes_xyxy"],
                selected["scores"],
                selected["class_ids"],
                strict=True,
            )
        ]
        return DetectionResult(
            detections=detections,
            capacity_saturated=saturated,
            uncertain_candidate_count=int(uncertain),
            refinement_executed=refinement_executed,
        )


def build_detector_v2(
    package: RuntimePackageV2,
    provider: ExecutionProvider,
    cuda_dll_dir: Path | None = None,
    *,
    cpu_detector_workers: int = 1,
    cpu_intra_op_threads: int = 0,
    openvino_cache_dir: Path | None = None,
    count_verifier_provider: ExecutionProvider | None = None,
    parallel_count_verifier: bool = False,
) -> OnnxDetector | CrossScaleOnnxDetector | FixedEnsembleOnnxDetector | CountVerifiedDetector:
    if package.metadata.detector.ensemble is not None:
        detector = FixedEnsembleOnnxDetector(
            package,
            provider,
            cuda_dll_dir,
            cpu_detector_workers=cpu_detector_workers,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
    elif package.metadata.detector_refinement is None:
        detector = OnnxDetector(
            package.detector_path,
            package.metadata.detector,
            provider,
            cuda_dll_dir,
            crowding_policy=package.metadata.detector_crowding,
            enable_cuda_graph=provider == "cuda",
            detector_class_count=package.metadata.detector_class_count,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
    else:
        detector = CrossScaleOnnxDetector(
            package,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
    if package.metadata.count_verifier is None:
        return detector
    if package.count_verifier_path is None:
        raise ValueError("count verifier metadata requires a packaged model")
    try:
        verifier = OnnxCountVerifier(
            package.count_verifier_path,
            package.metadata.count_verifier,
            count_verifier_provider or provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
    except Exception:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
        raise
    return CountVerifiedDetector(
        detector,
        verifier,
        parallel_verification=parallel_count_verifier,
    )


def replace_count_verifier_v2(
    detector: CountVerifiedDetector,
    package: RuntimePackageV2,
    provider: ExecutionProvider,
    cuda_dll_dir: Path | None = None,
    *,
    cpu_intra_op_threads: int = 0,
    openvino_cache_dir: Path | None = None,
    parallel_verification: bool = False,
) -> None:
    """Warm a replacement verifier before swapping it into a live detector."""

    if package.metadata.count_verifier is None or package.count_verifier_path is None:
        raise ValueError("count verifier metadata requires a packaged model")
    verifier = OnnxCountVerifier(
        package.count_verifier_path,
        package.metadata.count_verifier,
        provider,
        cuda_dll_dir,
        cpu_intra_op_threads=cpu_intra_op_threads,
        openvino_cache_dir=openvino_cache_dir,
    )
    try:
        verifier.warmup()
    except Exception:
        verifier.close()
        raise
    detector.replace_verifier(
        verifier,
        parallel_verification=parallel_verification,
    )
