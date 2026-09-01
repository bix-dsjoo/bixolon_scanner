from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image

from ..pipeline.ports import Detection, DetectionResult
from .detector_v2 import FixedEnsembleOnnxDetector
from .imaging import image_original_size
from .onnx import CountVerifiedDetector, OnnxCountVerifier
from .proposal_selection import (
    ClassAssistedSelectionPolicy,
    box_iou,
    containment,
    count_and_class_assisted_select,
    select_single_object_recovery,
)


class ClassifierAssistedEnsembleDetector:
    """Coordinate D-FINE proposals, exact-count evidence, and catalog classes."""

    def __init__(
        self,
        detector: FixedEnsembleOnnxDetector,
        verifier: OnnxCountVerifier,
        classifier: Any,
        *,
        parallel_verification: bool = False,
    ) -> None:
        self.detector = detector
        self.verifier = verifier
        self.classifier = classifier
        self.policy = detector.ensemble.class_verified_selector
        self.version = detector.version
        self._verification_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="bixolon-count")
            if parallel_verification
            else None
        )

    def warmup(self) -> None:
        self.detector.warmup()
        self.verifier.warmup()

    def close(self) -> None:
        if self._verification_executor is not None:
            self._verification_executor.shutdown(wait=True, cancel_futures=True)
            self._verification_executor = None
        self.verifier.close()
        self.detector.close()

    @staticmethod
    def _size(image: np.ndarray | Image.Image) -> tuple[int, int]:
        if isinstance(image, Image.Image):
            return image_original_size(image)
        height, width = image.shape[:2]
        return width, height

    def _selection_policy(self) -> ClassAssistedSelectionPolicy:
        return ClassAssistedSelectionPolicy(
            base_match_iou=self.policy.base_match_iou,
            candidate_minimum_support=self.policy.candidate_minimum_support,
            group_relation_iou=self.policy.group_relation_iou,
            group_area_ratio=self.policy.group_area_ratio,
            group_margin_ratio=self.policy.group_margin_ratio,
            group_novel_margin=self.policy.group_novel_margin,
            group_minimum_score=self.policy.group_minimum_score,
            independent_maximum_iou=self.policy.independent_maximum_iou,
            independent_margin=self.policy.independent_margin,
            independent_minimum_score=self.policy.independent_minimum_score,
            proposal_minimum_score=self.policy.candidate_minimum_score,
            refinement_maximum_support=(
                self.policy.candidate_minimum_support
                if self.policy.refinement_candidate_maximum_support is None
                else self.policy.refinement_candidate_maximum_support
            ),
            single_object_expansion_minimum_containment=getattr(
                self.policy, "single_object_expansion_minimum_containment", None
            ),
            single_object_expansion_minimum_area_ratio=getattr(
                self.policy, "single_object_expansion_minimum_area_ratio", None
            ),
            single_object_expansion_minimum_approval_score=getattr(
                self.policy, "single_object_expansion_minimum_approval_score", None
            ),
            single_object_expansion_minimum_approval_margin=getattr(
                self.policy, "single_object_expansion_minimum_approval_margin", None
            ),
        )

    @staticmethod
    def _localization_is_ambiguous(selected: dict[str, Any]) -> bool:
        boxes = selected["boxes_xyxy"]
        return any(
            left_index != right_index and containment(left, right) >= 0.75
            for left_index, left in enumerate(boxes)
            for right_index, right in enumerate(boxes)
        )

    def _single_object_expansion_is_ambiguous(
        self, selected: dict[str, Any], raw: dict[str, Any]
    ) -> bool:
        minimum_containment = getattr(
            self.policy, "single_object_expansion_minimum_containment", None
        )
        minimum_area_ratio = getattr(
            self.policy, "single_object_expansion_minimum_area_ratio", None
        )
        if (
            minimum_containment is None
            or minimum_area_ratio is None
            or len(selected["boxes_xyxy"]) != 1
        ):
            return False
        base = selected["boxes_xyxy"][0]
        base_area = max(0.0, base[2] - base[0]) * max(0.0, base[3] - base[1])
        return any(
            float(score) >= self.policy.candidate_minimum_score
            and int(support) >= self.policy.candidate_minimum_support
            and containment(candidate, base) >= minimum_containment
            and max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
            >= base_area * minimum_area_ratio
            for candidate, score, support in zip(
                raw["boxes_xyxy"],
                raw["scores"],
                raw["support_counts"],
                strict=True,
            )
        )

    def _classify_proposals(
        self,
        image: np.ndarray | Image.Image,
        base: dict[str, Any],
        raw: dict[str, Any],
    ) -> list[dict[str, Any]]:
        primary = getattr(self.classifier, "primary", self.classifier)
        base_boxes = base["boxes_xyxy"]
        indices = [
            index
            for index, (score, support) in enumerate(
                zip(raw["scores"], raw["support_counts"], strict=True)
            )
            if float(score) >= self.policy.candidate_minimum_score
            and int(support) >= self.policy.candidate_minimum_support
        ]
        tensors = []
        proposal_detections = []
        for index in indices:
            candidate_box = raw["boxes_xyxy"][index]
            context_boxes = [
                box
                for box in base_boxes
                if containment(box, candidate_box) < self.policy.candidate_duplicate_iou
            ]
            context_boxes.append(candidate_box)
            context = [Detection(*box, 1.0) for box in context_boxes]
            tensors.append(
                primary.embedder.prepare_selected_detection_tensors(
                    image,
                    context,
                    np.asarray([len(context) - 1], dtype=np.int64),
                )
            )
            proposal_detections.append(
                Detection(
                    *candidate_box,
                    float(raw["scores"][index]),
                    None,
                )
            )
        if not tensors:
            return []
        prepared = np.concatenate(tensors, axis=0)
        embedding_parts = []
        for start in range(0, len(prepared), self.policy.classifier_batch_size):
            embedding_parts.append(
                primary.embedder.embed_prepared_tensors_raw(
                    prepared[start : start + self.policy.classifier_batch_size]
                )
            )
        embeddings = np.concatenate(embedding_parts, axis=0)
        result = primary.classify_embeddings(embeddings, proposal_detections)
        entries = []
        for row, index in enumerate(indices):
            order = np.argsort(-result.ranking_logits[row], kind="stable")
            entries.append(
                {
                    "proposal_index": index,
                    "box": list(raw["boxes_xyxy"][index]),
                    "detector_score": float(raw["scores"][index]),
                    "support_count": int(raw["support_counts"][index]),
                    "predicted_class": int(order[0]),
                    "approval_score": float(result.approval_scores[row]),
                }
            )
        return entries

    @staticmethod
    def _detector_classes(
        selected: dict[str, Any], raw: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        classes = []
        supports = []
        for box in selected["boxes_xyxy"]:
            overlaps = [box_iou(box, candidate) for candidate in raw["boxes_xyxy"]]
            index = max(range(len(overlaps)), key=overlaps.__getitem__)
            classes.append(int(raw["detector_class_ids"][index]))
            supports.append(int(raw["class_support_counts"][index]))
        return tuple(classes), tuple(supports)

    def refine_unknown_detections(
        self,
        image: np.ndarray | Image.Image,
        current: DetectionResult,
    ) -> DetectionResult | None:
        """Run the full fallback ensemble after an uncertain or risky primary result."""

        if not getattr(self.policy, "low_resolution_unknown_ensemble_fallback", False):
            return None
        minimum_dimension = getattr(
            self.policy,
            "low_resolution_ensemble_fallback_minimum_dimension",
            None,
        )
        width, height = self._size(image)
        if minimum_dimension is None or min(width, height) < minimum_dimension:
            return None
        selected, raw, _agreement, _uncertain, saturated = self.detector.predict_candidates(
            image,
            apply_low_resolution_override=False,
        )
        predicted_count, _count_confidence = self.verifier.verify(image)
        if predicted_count <= 0:
            return None
        if predicted_count != len(selected["boxes_xyxy"]) or self._localization_is_ambiguous(
            selected
        ):
            entries = self._classify_proposals(image, selected, raw)
            if entries:
                selected, _diagnostics = count_and_class_assisted_select(
                    selected,
                    raw,
                    entries,
                    predicted_count,
                    self._selection_policy(),
                )
        detector_classes, detector_supports = self._detector_classes(selected, raw)
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                selected["boxes_xyxy"],
                selected["scores"],
                selected["class_ids"],
                strict=True,
            )
        ]
        if len(detections) != predicted_count:
            return None
        current_boxes = [(item.x1, item.y1, item.x2, item.y2) for item in current.detections]
        refined_boxes = [(item.x1, item.y1, item.x2, item.y2) for item in detections]
        if current_boxes == refined_boxes:
            return None
        return DetectionResult(
            detections=detections,
            capacity_saturated=current.capacity_saturated or saturated,
            verified_count=len(detections),
            count_confidence=1.0,
            uncertain_candidate_count=0,
            detector_class_ids=detector_classes,
            detector_class_support_counts=detector_supports,
        )

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        width, height = self._size(image)
        verification: Future[tuple[int, float]] | None = None
        if self._verification_executor is not None:
            verification = self._verification_executor.submit(self.verifier.verify, image)
        low_resolution_limit = getattr(self.policy, "low_resolution_maximum_dimension", None)
        if low_resolution_limit is not None and max(width, height) <= low_resolution_limit:
            try:
                selected, raw, _agreement, _uncertain, saturated = self.detector.predict_candidates(
                    image
                )
            except Exception:
                if verification is not None:
                    try:
                        verification.result()
                    except Exception:
                        pass
                raise
            predicted_count, count_confidence = (
                verification.result() if verification is not None else self.verifier.verify(image)
            )
            if (
                predicted_count == 0
                and self.policy.zero_count_minimum_confidence is not None
                and count_confidence >= self.policy.zero_count_minimum_confidence
            ):
                return DetectionResult(
                    detections=[],
                    verified_count=0,
                    count_confidence=count_confidence,
                    detector_class_ids=(),
                )
            ensemble_fallback_dimension = getattr(
                self.policy,
                "low_resolution_ensemble_fallback_minimum_dimension",
                None,
            )
            ensemble_fallback_confidence = getattr(
                self.policy,
                "low_resolution_ensemble_fallback_minimum_count_confidence",
                None,
            )
            if (
                ensemble_fallback_dimension is not None
                and ensemble_fallback_confidence is not None
                and min(width, height) >= ensemble_fallback_dimension
                and predicted_count != len(selected["boxes_xyxy"])
                and count_confidence >= ensemble_fallback_confidence
            ):
                selected, raw, _agreement, _uncertain, fallback_saturated = (
                    self.detector.predict_candidates(
                        image,
                        apply_low_resolution_override=False,
                    )
                )
                saturated = saturated or fallback_saturated
                if predicted_count != len(
                    selected["boxes_xyxy"]
                ) or self._localization_is_ambiguous(selected):
                    entries = self._classify_proposals(image, selected, raw)
                    if entries:
                        selected, _diagnostics = count_and_class_assisted_select(
                            selected,
                            raw,
                            entries,
                            predicted_count,
                            self._selection_policy(),
                        )
            fallback_confidence = getattr(
                self.policy,
                "low_resolution_positive_count_minimum_confidence",
                None,
            )
            recovery_maximum_count = getattr(
                self.policy,
                "low_resolution_recovery_maximum_count",
                None,
            )
            if (
                not selected["boxes_xyxy"]
                and predicted_count > 0
                and recovery_maximum_count is not None
                and predicted_count <= recovery_maximum_count
                and fallback_confidence is not None
                and count_confidence >= fallback_confidence
            ):
                entries = self._classify_proposals(image, selected, raw)
                if entries:
                    selected = select_single_object_recovery(
                        entries,
                        minimum_score=self.policy.candidate_minimum_score,
                        minimum_support=self.policy.candidate_minimum_support,
                        maximum_aspect_ratio=self.policy.low_resolution_recovery_maximum_aspect_ratio,
                        minimum_approval_score=self.policy.low_resolution_recovery_minimum_approval_score,
                    )
                    if selected is None:
                        selected = {"boxes_xyxy": [], "scores": [], "class_ids": []}
            if self._single_object_expansion_is_ambiguous(selected, raw):
                entries = self._classify_proposals(image, selected, raw)
                if entries:
                    selected, _diagnostics = count_and_class_assisted_select(
                        selected,
                        raw,
                        entries,
                        len(selected["boxes_xyxy"]),
                        self._selection_policy(),
                    )
            detector_classes, detector_supports = self._detector_classes(selected, raw)
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
                verified_count=len(detections),
                count_confidence=1.0,
                uncertain_candidate_count=0,
                detector_class_ids=detector_classes,
                detector_class_support_counts=detector_supports,
            )
        try:
            selected, raw, _agreement, _uncertain, saturated = self.detector.predict_candidates(
                image
            )
        except Exception:
            if verification is not None:
                try:
                    verification.result()
                except Exception:
                    pass
            raise
        predicted_count, count_confidence = (
            verification.result() if verification is not None else self.verifier.verify(image)
        )
        if (
            predicted_count == 0
            and self.policy.zero_count_minimum_confidence is not None
            and count_confidence >= self.policy.zero_count_minimum_confidence
        ):
            return DetectionResult(
                detections=[],
                verified_count=0,
                count_confidence=count_confidence,
                detector_class_ids=(),
            )

        count_assistance = (
            self.policy.count_assistance_minimum_dimension is not None
            and min(width, height) >= self.policy.count_assistance_minimum_dimension
        )
        count_mismatch = predicted_count != len(selected["boxes_xyxy"])
        if count_assistance and (
            self._localization_is_ambiguous(selected)
            or self.policy.count_assistance_on_count_mismatch
            and count_mismatch
        ):
            entries = self._classify_proposals(image, selected, raw)
            if entries:
                selected, _diagnostics = count_and_class_assisted_select(
                    selected,
                    raw,
                    entries,
                    predicted_count,
                    self._selection_policy(),
                )
        detector_classes, detector_supports = self._detector_classes(selected, raw)
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
            verified_count=len(detections),
            count_confidence=1.0,
            uncertain_candidate_count=0,
            detector_class_ids=detector_classes,
            detector_class_support_counts=detector_supports,
        )


def attach_classifier_assisted_detector(detector, classifier):
    if not isinstance(detector, CountVerifiedDetector) or not isinstance(
        detector.detector, FixedEnsembleOnnxDetector
    ):
        return detector
    policy = detector.detector.ensemble.class_verified_selector
    assisted = any(
        value is not None
        for value in (
            policy.count_assistance_minimum_dimension,
            policy.zero_count_minimum_confidence,
            policy.wide_pair_target_aspect_ratio,
        )
    )
    if not assisted:
        return detector
    parallel_verification = detector.detach_parallel_verification()
    return ClassifierAssistedEnsembleDetector(
        detector.detector,
        detector.verifier,
        classifier,
        parallel_verification=parallel_verification,
    )


__all__ = ["ClassifierAssistedEnsembleDetector", "attach_classifier_assisted_detector"]
