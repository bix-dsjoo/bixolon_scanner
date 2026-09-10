from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np
from PIL import Image

from ..contracts import (
    ModelVersions,
    ScanResponse,
    Status,
)
from ..contracts.errors import ModelExecutionError
from ..contracts.model_package import (
    ClassifierMetadata,
    CountVerifierMetadata,
    DetectorClassVerifiedSelectorMetadata,
    QualityMetadata,
)
from ..contracts.runtime_package_v2 import DetectorPrimaryClassifierRoutingMetadata
from .classification import (
    ClassifierBatch,
    merge_selected_classifier_batch,
    normalize_classification,
    softmax,
)
from .ports import Classifier, Detection, Detector
from .quality import border_detection_indices, contained_detection_pairs, quality_reasons
from .segmentation import build_scan_items, summarize_reason_codes

LOGGER = logging.getLogger(__name__)


_softmax = softmax


class DecisionPipeline:
    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        classifier_metadata: ClassifierMetadata,
        quality_metadata: QualityMetadata,
        count_verifier_metadata: CountVerifierMetadata | None = None,
        *,
        worker_version: str = "1.0.0",
        embedder_version: str | None = None,
        detector_policy_version: str | None = None,
        classifier_policy_version: str | None = None,
        catalog_version: str | None = None,
        assisted_policy: DetectorClassVerifiedSelectorMetadata | None = None,
        detector_primary_classifier_routing: (
            DetectorPrimaryClassifierRoutingMetadata | None
        ) = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.classifier_metadata = classifier_metadata
        self.quality_metadata = quality_metadata
        self.count_verifier_metadata = count_verifier_metadata
        self.worker_version = worker_version
        self.embedder_version = embedder_version
        self.detector_policy_version = detector_policy_version
        self.classifier_policy_version = classifier_policy_version
        self.catalog_version = catalog_version
        self.assisted_policy = assisted_policy
        self.detector_primary_classifier_routing = detector_primary_classifier_routing
        if detector_primary_classifier_routing is not None:
            if not callable(getattr(classifier, "classify_selected", None)):
                raise ValueError("detector-primary routing requires selected ROI classification")
            class_count = len(classifier_metadata.labels)
            if any(
                index >= class_count
                for index in detector_primary_classifier_routing.direct_approval_class_indices
            ):
                raise ValueError("detector-primary class index exceeds classifier label count")

    def _detector_primary_batch(
        self,
        ordered: list[Detection],
        detector_classes: list[int | None],
        *,
        uses_explicit_ranking_scores: bool,
    ) -> ClassifierBatch:
        detection_count = len(ordered)
        class_count = len(self.classifier_metadata.labels)
        probabilities = np.zeros((detection_count, class_count), dtype=np.float32)
        ranking_probabilities = np.zeros_like(probabilities)
        approval_scores = np.zeros(detection_count, dtype=np.float32)
        for index, (detection, detector_class) in enumerate(
            zip(ordered, detector_classes, strict=True)
        ):
            if detector_class is None or not 0 <= detector_class < class_count:
                continue
            probabilities[index, detector_class] = 1.0
            ranking_probabilities[index, detector_class] = 1.0
            approval_scores[index] = float(detection.score)
        return ClassifierBatch(
            probabilities=probabilities,
            ranking_probabilities=ranking_probabilities,
            decision_indices=np.argsort(
                -ranking_probabilities,
                axis=1,
                kind="stable",
            ),
            approval_scores=approval_scores,
            approved=np.ones(detection_count, dtype=bool),
            top3_unsafe=np.zeros(detection_count, dtype=bool),
            segment_recapture_reasons=(None,) * detection_count,
            unknown_reasons=(None,) * detection_count,
            uses_explicit_ranking_scores=uses_explicit_ranking_scores,
        )

    def _classify_primary(
        self,
        image: np.ndarray | Image.Image,
        ordered: list[Detection],
        detector_classes: list[int | None],
    ) -> ClassifierBatch:
        routing = self.detector_primary_classifier_routing
        if routing is None:
            return normalize_classification(
                self.classifier.classify(image, ordered),
                detection_count=len(ordered),
                metadata=self.classifier_metadata,
            )

        direct_classes = set(routing.direct_approval_class_indices)
        class_counts = {
            class_index: detector_classes.count(class_index)
            for class_index in set(detector_classes)
            if class_index is not None
        }
        classifier_indices = np.asarray(
            [
                index
                for index, (detection, detector_class) in enumerate(
                    zip(ordered, detector_classes, strict=True)
                )
                if (
                    detector_class not in direct_classes
                    or detection.score < routing.minimum_detector_score
                    or (
                        routing.require_unique_class_per_image
                        and class_counts.get(detector_class, 0) != 1
                    )
                )
            ],
            dtype=np.int64,
        )
        if not len(classifier_indices):
            batch = self._detector_primary_batch(
                ordered,
                detector_classes,
                uses_explicit_ranking_scores=True,
            )
            record_direct_batch = getattr(
                self.classifier,
                "record_detector_primary_batch",
                None,
            )
            if callable(record_direct_batch):
                record_direct_batch(batch)
            return batch

        classify_selected = getattr(self.classifier, "classify_selected")
        selected = normalize_classification(
            classify_selected(image, ordered, classifier_indices),
            detection_count=len(classifier_indices),
            metadata=self.classifier_metadata,
        )
        base = self._detector_primary_batch(
            ordered,
            detector_classes,
            uses_explicit_ranking_scores=selected.uses_explicit_ranking_scores,
        )
        return merge_selected_classifier_batch(base, selected, classifier_indices)

    def _refine_wide_pair(
        self,
        ordered: list[Detection],
        batch,
        detector_classes: list[int | None],
        detector_supports: list[int],
    ) -> tuple[list[Detection], bool]:
        policy = self.assisted_policy
        if (
            policy is None
            or policy.wide_pair_detector_class_index is None
            or policy.wide_pair_classifier_class_index is None
            or policy.wide_pair_target_aspect_ratio is None
        ):
            return ordered, False
        refined = list(ordered)
        changed = False
        for index, detection in enumerate(ordered):
            if (
                batch.approved[index]
                and int(batch.decision_indices[index, 0]) == policy.wide_pair_classifier_class_index
                and detector_classes[index] == policy.wide_pair_detector_class_index
                and detector_supports[index] >= policy.candidate_minimum_support
            ):
                target_height = (detection.x2 - detection.x1) / policy.wide_pair_target_aspect_ratio
                refined[index] = Detection(
                    detection.x1,
                    detection.y2 - target_height,
                    detection.x2,
                    detection.y2,
                    detection.score,
                    detection.class_id,
                )
                changed = True
        return refined, changed

    def _reconcile_single_views(
        self,
        image: np.ndarray | Image.Image,
        ordered: list[Detection],
        batch,
        detector_classes: list[int | None],
        detector_supports: list[int],
        *,
        use_classifier_fallback: bool = False,
    ):
        policy = self.assisted_policy
        if policy is None or not (
            policy.unknown_consensus_promotion or policy.single_view_top3_fusion
        ):
            return batch
        method_name = (
            "classify_fallback_single_views" if use_classifier_fallback else "classify_single_views"
        )
        classify_single_views = getattr(self.classifier, method_name, None)
        if not callable(classify_single_views):
            raise ValueError("assisted classifier does not provide single-view verification")
        unapproved_indices = np.flatnonzero(~batch.approved)
        if not len(unapproved_indices):
            return batch
        selected_detections = [ordered[int(index)] for index in unapproved_indices]
        single = normalize_classification(
            classify_single_views(image, selected_detections),
            detection_count=len(selected_detections),
            metadata=self.classifier_metadata,
        )
        ranking_probabilities = batch.ranking_probabilities.copy()
        if policy.single_view_top3_fusion:
            ranking_probabilities[unapproved_indices] = np.maximum(
                ranking_probabilities[unapproved_indices],
                single.ranking_probabilities,
            )
        decision_indices = np.argsort(-ranking_probabilities, axis=1, kind="stable")
        approved = batch.approved.copy()
        if policy.unknown_consensus_promotion:
            for single_index, index_value in enumerate(unapproved_indices):
                index = int(index_value)
                detector_class = detector_classes[index]
                if detector_class is None:
                    continue
                single_top3 = single.decision_indices[single_index, :3]
                if (
                    detector_supports[index] >= policy.candidate_minimum_support
                    and (
                        policy.unknown_promotion_minimum_single_view_approval is None
                        or single.approval_scores[single_index]
                        >= policy.unknown_promotion_minimum_single_view_approval
                    )
                    and int(batch.decision_indices[index, 0]) == detector_class
                    and int(single_top3[0]) != detector_class
                    and detector_class in single_top3
                ):
                    approved[index] = True
                    order = decision_indices[index].tolist()
                    order.remove(detector_class)
                    decision_indices[index] = np.asarray(
                        [detector_class, *order], dtype=decision_indices.dtype
                    )
        return replace(
            batch,
            ranking_probabilities=ranking_probabilities,
            decision_indices=decision_indices,
            approved=approved,
        )

    def _reconcile_detector_classifier_consensus(
        self,
        ordered: list[Detection],
        batch,
        detector_classes: list[int | None],
    ):
        policy = self.quality_metadata.detector_classifier_consensus
        if policy is None or not ordered:
            return batch
        original_approved = batch.approved.copy()
        approved = batch.approved.copy()
        decision_indices = batch.decision_indices.copy()
        ranking_probabilities = batch.ranking_probabilities.copy()
        recapture_reasons = list(batch.segment_recapture_reasons or (None,) * len(ordered))
        class_count = len(self.classifier_metadata.labels)

        def move_detector_class_first(index: int, detector_class: int) -> None:
            order = decision_indices[index].tolist()
            order.remove(detector_class)
            decision_indices[index] = np.asarray(
                [detector_class, *order],
                dtype=decision_indices.dtype,
            )
            ranking_probabilities[index, detector_class] = float(ranking_probabilities[index].max())

        for index, (detection, detector_class) in enumerate(
            zip(ordered, detector_classes, strict=True)
        ):
            if (
                detector_class is None
                or not 0 <= detector_class < class_count
                or detection.score < policy.minimum_detector_score
            ):
                continue
            classifier_top1 = int(batch.decision_indices[index, 0])
            classifier_probability_top1 = int(np.argmax(batch.probabilities[index]))
            disagreement = classifier_top1 != detector_class
            if (
                original_approved[index]
                and disagreement
                and batch.approval_scores[index]
                <= policy.approved_disagreement_maximum_classifier_score
            ):
                approved[index] = False
            elif (
                not original_approved[index]
                and policy.promote_recapture_on_top1_consensus
                and recapture_reasons[index] is not None
                and classifier_probability_top1 == detector_class
            ):
                approved[index] = True
                recapture_reasons[index] = None
                move_detector_class_first(index, detector_class)
                continue
            elif (
                not original_approved[index]
                and policy.promote_safe_unknown_on_top3_consensus
                and not batch.top3_unsafe[index]
                and recapture_reasons[index] is None
                and detector_class in batch.decision_indices[index, :3]
            ):
                approved[index] = True
                move_detector_class_first(index, detector_class)
                continue
            if not approved[index] and policy.inject_detector_class_into_unknown_top3:
                move_detector_class_first(index, detector_class)
        return replace(
            batch,
            ranking_probabilities=ranking_probabilities,
            decision_indices=decision_indices,
            approved=approved,
            segment_recapture_reasons=tuple(recapture_reasons),
        )

    def _apply_roi_integrity(self, batch: ClassifierBatch) -> ClassifierBatch:
        threshold = self.quality_metadata.multi_object_recapture_threshold
        if threshold is None:
            return batch
        if batch.multi_object_probabilities is None:
            raise ModelExecutionError
        rejected = batch.multi_object_probabilities >= threshold
        approved = batch.approved & ~rejected
        reasons = list(batch.segment_recapture_reasons or (None,) * len(approved))
        for index in np.flatnonzero(rejected):
            reasons[int(index)] = "CLASSIFIER_MULTIPLE_OBJECTS_IN_ROI"
        return replace(batch, approved=approved, segment_recapture_reasons=tuple(reasons))

    def _classify_with_resolution_fallback(
        self,
        image: np.ndarray | Image.Image,
        ordered: list[Detection],
        detector_classes: list[int | None],
        detector_supports: list[int],
    ) -> tuple[ClassifierBatch, bool]:
        batch = self._classify_primary(image, ordered, detector_classes)
        batch = self._apply_roi_integrity(batch)
        fallback_policy = getattr(
            self.classifier,
            "resolution_fallback_metadata",
            None,
        )
        if fallback_policy is None or not len(ordered):
            return batch, False

        recapture_reasons = batch.segment_recapture_reasons or (None,) * len(ordered)
        safe_unknown = (
            ~batch.approved
            & ~batch.top3_unsafe
            & np.asarray([reason is None for reason in recapture_reasons], dtype=bool)
        )
        fallback_indices = np.zeros(len(ordered), dtype=bool)
        maximum_approval_score_decreases = np.full(len(ordered), np.inf, dtype=np.float32)
        if fallback_policy.fallback_on_unknown:
            fallback_indices |= safe_unknown
        unsafe = ~batch.approved & ~safe_unknown
        if fallback_policy.fallback_on_unsafe:
            fallback_indices |= unsafe

        detector_ids = np.asarray(
            [-1 if value is None else value for value in detector_classes],
            dtype=np.int64,
        )
        supported_disagreement = (
            batch.approved
            & (detector_ids >= 0)
            & (
                np.asarray(detector_supports, dtype=np.int64)
                >= fallback_policy.minimum_detector_support
            )
            & (batch.decision_indices[:, 0] != detector_ids)
        )
        for rule in fallback_policy.approval_disagreement_rules:
            count_matches = len(ordered) >= rule.minimum_detection_count and (
                rule.maximum_detection_count is None or len(ordered) <= rule.maximum_detection_count
            )
            approval_candidates = (
                supported_disagreement.copy()
                if rule.require_detector_disagreement
                else batch.approved.copy()
            )
            minimum_aspect_ratio = getattr(rule, "minimum_box_aspect_ratio", None)
            if minimum_aspect_ratio is not None:
                widths = np.asarray(
                    [max(detection.x2 - detection.x1, 1e-6) for detection in ordered],
                    dtype=np.float32,
                )
                heights = np.asarray(
                    [max(detection.y2 - detection.y1, 1e-6) for detection in ordered],
                    dtype=np.float32,
                )
                approval_candidates &= (
                    np.maximum(widths / heights, heights / widths) >= minimum_aspect_ratio
                )
            if count_matches:
                rule_indices = approval_candidates & (
                    batch.approval_scores <= rule.maximum_approval_score
                )
                fallback_indices |= rule_indices
                maximum_decrease = getattr(rule, "maximum_approval_score_decrease", None)
                if maximum_decrease is not None:
                    maximum_approval_score_decreases[rule_indices] = np.minimum(
                        maximum_approval_score_decreases[rule_indices],
                        maximum_decrease,
                    )
        integrity_threshold = self.quality_metadata.multi_object_recapture_threshold
        if integrity_threshold is not None:
            fallback_indices &= batch.multi_object_probabilities < integrity_threshold
        selected_indices = np.flatnonzero(fallback_indices)
        if not len(selected_indices):
            return batch, False

        def apply_fallback_safety(
            fallback_batch: ClassifierBatch,
            base_indices: np.ndarray,
        ) -> ClassifierBatch:
            score_decreases = batch.approval_scores[base_indices] - fallback_batch.approval_scores
            demote_for_score_decrease = fallback_batch.approved & (
                score_decreases > maximum_approval_score_decreases[base_indices]
            )
            below_minimum_fallback_score = np.zeros(
                len(fallback_batch.approved),
                dtype=bool,
            )
            minimum_fallback_score = getattr(
                fallback_policy,
                "minimum_fallback_approval_score",
                None,
            )
            if minimum_fallback_score is not None:
                below_minimum_fallback_score = (
                    fallback_batch.approval_scores < minimum_fallback_score
                )
            demote_for_minimum_score = fallback_batch.approved & below_minimum_fallback_score
            demote_approvals = demote_for_score_decrease | demote_for_minimum_score
            if not np.any(demote_approvals) and not np.any(below_minimum_fallback_score):
                return fallback_batch
            approved = fallback_batch.approved.copy()
            approved[demote_approvals] = False
            unknown_reasons = list(
                fallback_batch.unknown_reasons or (None,) * len(fallback_batch.approved)
            )
            for index in np.flatnonzero(below_minimum_fallback_score):
                unknown_reasons[index] = "BELOW_APPROVAL_THRESHOLD"
            for index in np.flatnonzero(demote_approvals):
                if unknown_reasons[index] is None:
                    unknown_reasons[index] = "BELOW_APPROVAL_THRESHOLD"
            return replace(
                fallback_batch,
                approved=approved,
                unknown_reasons=tuple(unknown_reasons),
            )

        classify_fallback_selected = getattr(
            self.classifier,
            "classify_fallback_selected",
            None,
        )
        if getattr(fallback_policy, "selective_roi_only", False) and callable(
            classify_fallback_selected
        ):
            classification = classify_fallback_selected(image, ordered, selected_indices)
            selected_batch = normalize_classification(
                classification,
                detection_count=len(selected_indices),
                metadata=self.classifier_metadata,
            )
            selected_batch = apply_fallback_safety(selected_batch, selected_indices)
            return (
                merge_selected_classifier_batch(
                    batch,
                    selected_batch,
                    selected_indices,
                    fuse_unapproved_top3=getattr(
                        fallback_policy,
                        "fuse_unapproved_top3",
                        False,
                    ),
                ),
                True,
            )

        classify_fallback = getattr(self.classifier, "classify_fallback", None)
        if not callable(classify_fallback):
            raise ValueError("classifier resolution fallback is not available")
        classification = classify_fallback(image, ordered)
        fallback_batch = normalize_classification(
            classification,
            detection_count=len(ordered),
            metadata=self.classifier_metadata,
        )
        all_indices = np.arange(len(ordered), dtype=np.int64)
        fallback_batch = apply_fallback_safety(fallback_batch, all_indices)
        return (
            merge_selected_classifier_batch(
                batch,
                fallback_batch,
                all_indices,
                fuse_unapproved_top3=getattr(
                    fallback_policy,
                    "fuse_unapproved_top3",
                    False,
                ),
            ),
            True,
        )

    @property
    def versions(self) -> ModelVersions:
        return ModelVersions(detector=self.detector.version, classifier=self.classifier.version)

    def scan(self, image: np.ndarray | Image.Image, request_id: str) -> ScanResponse:
        started = time.perf_counter()
        detector_started = time.perf_counter()
        detection_result = self.detector.detect(image)
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
        reasons = quality_reasons(image, detection_result, self.quality_metadata)
        if detection_result.uncertain_candidate_count:
            reasons.append("DETECTOR_UNCERTAIN_OBJECT")
        if self.count_verifier_metadata is not None:
            if detection_result.verified_count is None or detection_result.count_confidence is None:
                raise ValueError("count verifier result is missing")
            if (
                detection_result.count_confidence
                < self.count_verifier_metadata.confidence_threshold
            ):
                reasons.append("DETECTOR_COUNT_UNCERTAIN")
            elif detection_result.verified_count != (
                int(bool(detection_result.detections))
                if self.count_verifier_metadata.comparison_mode == "object_presence"
                else len(detection_result.detections)
            ):
                reasons.append("DETECTOR_COUNT_MISMATCH")
            reasons = list(dict.fromkeys(reasons))
        if reasons:
            response = ScanResponse(
                request_id=request_id,
                status=Status.IMAGE_RECAPTURE,
                reason_codes=["IMAGE_RECAPTURE_REQUIRED"],
                segmentations=[],
                processing_time_ms=(time.perf_counter() - started) * 1000.0,
                worker_version=self.worker_version,
                detector_version=self.detector.version,
                classifier_version=None,
                embedder_version=None,
                detector_policy_version=self.detector_policy_version,
                classifier_policy_version=None,
                catalog_version=None,
            )
            self._log(
                response,
                detector_ms=detector_ms,
                classifier_ms=0.0,
                diagnostic_reason_codes=reasons,
            )
            return response

        ordered_rows = sorted(
            enumerate(detection_result.detections),
            key=lambda row: (row[1].y1, row[1].x1),
        )
        ordered = [row[1] for row in ordered_rows]
        detector_classes = [
            (
                detection_result.detector_class_ids[row[0]]
                if detection_result.detector_class_ids
                else row[1].class_id
            )
            for row in ordered_rows
        ]
        detector_supports = [
            detection_result.detector_class_support_counts[row[0]]
            if detection_result.detector_class_support_counts
            else 0
            for row in ordered_rows
        ]
        classifier_started = time.perf_counter()
        batch, used_classifier_fallback = self._classify_with_resolution_fallback(
            image,
            ordered,
            detector_classes,
            detector_supports,
        )
        ordered, refined = self._refine_wide_pair(
            ordered,
            batch,
            detector_classes,
            detector_supports,
        )
        if refined:
            batch, used_classifier_fallback = self._classify_with_resolution_fallback(
                image,
                ordered,
                detector_classes,
                detector_supports,
            )
        contextual_approved = batch.approved.copy()
        contextual_top1 = batch.decision_indices[:, 0].copy()
        batch = self._reconcile_single_views(
            image,
            ordered,
            batch,
            detector_classes,
            detector_supports,
            use_classifier_fallback=used_classifier_fallback,
        )
        batch = self._reconcile_detector_classifier_consensus(
            ordered,
            batch,
            detector_classes,
        )
        refinement_ms = 0.0
        refine_unknown = getattr(self.detector, "refine_unknown_detections", None)
        recapture_reasons = batch.segment_recapture_reasons
        refinement_eligible = ~batch.approved & ~batch.top3_unsafe
        if recapture_reasons is not None:
            refinement_eligible &= np.asarray(
                [reason is None for reason in recapture_reasons],
                dtype=bool,
            )
        policy = self.assisted_policy
        if policy is not None and getattr(
            policy,
            "low_resolution_risky_approval_ensemble_fallback",
            False,
        ):
            detector_ids = np.asarray(
                [-1 if value is None else value for value in detector_classes],
                dtype=np.int64,
            )
            detector_support_array = np.asarray(detector_supports, dtype=np.int64)
            supported_disagreement = (
                contextual_approved
                & (detector_ids >= 0)
                & (detector_support_array >= policy.candidate_minimum_support)
                & (contextual_top1 != detector_ids)
            )
            risky_shape_or_score = np.zeros(len(ordered), dtype=bool)
            maximum_score = getattr(
                policy,
                "low_resolution_risky_approval_maximum_score",
                None,
            )
            if maximum_score is not None:
                risky_shape_or_score |= batch.approval_scores <= maximum_score
            minimum_aspect_ratio = getattr(
                policy,
                "low_resolution_risky_approval_minimum_aspect_ratio",
                None,
            )
            if minimum_aspect_ratio is not None:
                aspect_ratios = np.asarray(
                    [
                        max(
                            (detection.x2 - detection.x1) / max(detection.y2 - detection.y1, 1e-9),
                            (detection.y2 - detection.y1) / max(detection.x2 - detection.x1, 1e-9),
                        )
                        for detection in ordered
                    ],
                    dtype=np.float32,
                )
                risky_shape_or_score |= aspect_ratios >= minimum_aspect_ratio
            supported_disagreement &= risky_shape_or_score
            promoted = ~contextual_approved & batch.approved
            refinement_eligible |= supported_disagreement | promoted
        if callable(refine_unknown) and np.any(refinement_eligible):
            refinement_started = time.perf_counter()
            refined_detection_result = refine_unknown(image, detection_result)
            refinement_ms = (time.perf_counter() - refinement_started) * 1000.0
            detector_ms += refinement_ms
            if refined_detection_result is not None:
                detection_result = refined_detection_result
                ordered_rows = sorted(
                    enumerate(detection_result.detections),
                    key=lambda row: (row[1].y1, row[1].x1),
                )
                ordered = [row[1] for row in ordered_rows]
                detector_classes = [
                    (
                        detection_result.detector_class_ids[row[0]]
                        if detection_result.detector_class_ids
                        else row[1].class_id
                    )
                    for row in ordered_rows
                ]
                detector_supports = [
                    detection_result.detector_class_support_counts[row[0]] for row in ordered_rows
                ]
                batch, used_classifier_fallback = self._classify_with_resolution_fallback(
                    image,
                    ordered,
                    detector_classes,
                    detector_supports,
                )
                ordered, refined = self._refine_wide_pair(
                    ordered,
                    batch,
                    detector_classes,
                    detector_supports,
                )
                if refined:
                    batch, used_classifier_fallback = self._classify_with_resolution_fallback(
                        image,
                        ordered,
                        detector_classes,
                        detector_supports,
                    )
                batch = self._reconcile_single_views(
                    image,
                    ordered,
                    batch,
                    detector_classes,
                    detector_supports,
                    use_classifier_fallback=used_classifier_fallback,
                )
                batch = self._reconcile_detector_classifier_consensus(
                    ordered,
                    batch,
                    detector_classes,
                )
        # Later agreement paths cannot promote a structurally invalid ROI back to APPROVED.
        batch = self._apply_roi_integrity(batch)
        classifier_ms = (time.perf_counter() - classifier_started) * 1000.0 - refinement_ms
        decision_indices = batch.decision_indices
        duplicate_review_indices = {
            lower_index
            for lower_index, higher_index in contained_detection_pairs(
                ordered, self.quality_metadata.duplicate_review_containment_threshold
            )
            if decision_indices[lower_index, 0] == decision_indices[higher_index, 0]
        }

        border_indices: set[int] = set()
        if self.quality_metadata.border_policy == "classifier_confidence":
            border_indices = border_detection_indices(image, ordered, self.quality_metadata)

        items = build_scan_items(
            ordered,
            batch,
            self.classifier_metadata,
            border_indices=border_indices,
            duplicate_review_indices=duplicate_review_indices,
        )
        response = ScanResponse(
            request_id=request_id,
            status=Status.SEGMENTATION,
            reason_codes=summarize_reason_codes(items),
            segmentations=items,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            worker_version=self.worker_version,
            detector_version=self.detector.version,
            classifier_version=self.classifier.version,
            embedder_version=self.embedder_version,
            detector_policy_version=self.detector_policy_version,
            classifier_policy_version=self.classifier_policy_version,
            catalog_version=self.catalog_version,
        )
        self._log(response, detector_ms=detector_ms, classifier_ms=classifier_ms)
        return response

    def close(self) -> None:
        """Release optional runtime resources owned by pipeline adapters."""

        closed: set[int] = set()
        for adapter in (self.classifier, self.detector):
            if id(adapter) in closed:
                continue
            closed.add(id(adapter))
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _log(
        response: ScanResponse,
        *,
        detector_ms: float,
        classifier_ms: float,
        diagnostic_reason_codes: list[str] | None = None,
    ) -> None:
        LOGGER.info(
            "scan_complete",
            extra={
                "request_id": response.request_id,
                "status": response.status.value,
                "reason_codes": response.reason_codes,
                "diagnostic_reason_codes": diagnostic_reason_codes or [],
                "segmentation_count": len(response.segmentations),
                "detector_ms": round(detector_ms, 3),
                "classifier_ms": round(classifier_ms, 3),
                "processing_time_ms": round(response.processing_time_ms, 3),
                "worker_version": response.worker_version,
                "detector_version": response.detector_version,
                "classifier_version": response.classifier_version,
            },
        )
