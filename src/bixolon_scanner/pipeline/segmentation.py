from __future__ import annotations

import numpy as np

from ..contracts import BoundingBox, Candidate, ItemStatus, Prediction, ScanItem
from ..contracts.model_package import ClassifierMetadata
from .classification import ClassifierBatch
from .ports import Detection


def _top3_candidates(
    metadata: ClassifierMetadata,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
) -> list[Candidate]:
    return [
        Candidate(
            class_id=metadata.labels[int(candidate_index)].class_id,
            class_name=metadata.labels[int(candidate_index)].class_name,
            confidence=float(candidate_scores[int(candidate_index)]),
        )
        for candidate_index in candidate_indices[:3]
    ]


def build_scan_items(
    detections: list[Detection],
    batch: ClassifierBatch,
    metadata: ClassifierMetadata,
    *,
    border_indices: set[int],
    duplicate_review_indices: set[int],
    detector_recapture_threshold: float | None = None,
) -> list[ScanItem]:
    """Apply the per-segmentation decision priority without changing batch order."""

    recapture_labels = {index for index, label in enumerate(metadata.labels) if label.recapture}
    items: list[ScanItem] = []
    for index, (detection, indices, scores, candidate_indices, candidate_scores) in enumerate(
        zip(
            detections,
            batch.decision_indices,
            batch.probabilities,
            batch.decision_indices,
            batch.ranking_probabilities,
        )
    ):
        ordinal = index + 1
        top1_index = int(indices[0])
        top1_score = float(scores[top1_index])
        label = metadata.labels[top1_index]
        bbox = BoundingBox(
            x=max(0, int(round(detection.x1))),
            y=max(0, int(round(detection.y1))),
            width=max(1, int(round(detection.x2 - detection.x1))),
            height=max(1, int(round(detection.y2 - detection.y1))),
        )
        if top1_index in recapture_labels:
            item = _recapture_item(ordinal, bbox, top1_score)
        elif index in border_indices and not batch.approved[index]:
            item = _recapture_item(ordinal, bbox, top1_score)
        elif batch.segment_recapture_reasons is not None and batch.segment_recapture_reasons[index]:
            item = _recapture_item(ordinal, bbox, 0.0)
        elif index in duplicate_review_indices and batch.approved[index]:
            item = ScanItem(
                segmentation_id=f"segmentation_{ordinal:03d}",
                bbox=bbox,
                status=ItemStatus.UNKNOWN,
                reason_codes=["DETECTOR_CONTAINED_DUPLICATE"],
                prediction=None,
                top3=_top3_candidates(metadata, candidate_indices, candidate_scores),
                confidence=float(batch.approval_scores[index]),
            )
        elif (
            detector_recapture_threshold is not None
            and detection.score < detector_recapture_threshold
        ):
            item = _recapture_item(ordinal, bbox, 0.0)
        elif batch.approved[index]:
            item = ScanItem(
                segmentation_id=f"segmentation_{ordinal:03d}",
                bbox=bbox,
                status=ItemStatus.APPROVED,
                reason_codes=[],
                prediction=Prediction(class_id=label.class_id, class_name=label.class_name),
                top3=[],
                confidence=float(batch.approval_scores[index]),
            )
        elif batch.top3_unsafe[index]:
            confidence = (
                0.0
                if batch.uses_explicit_ranking_scores
                else float(candidate_scores[int(candidate_indices[0])])
            )
            item = _recapture_item(ordinal, bbox, confidence)
        else:
            reason = (
                "BELOW_APPROVAL_THRESHOLD"
                if batch.unknown_reasons is None or batch.unknown_reasons[index] is None
                else batch.unknown_reasons[index]
            )
            item = ScanItem(
                segmentation_id=f"segmentation_{ordinal:03d}",
                bbox=bbox,
                status=ItemStatus.UNKNOWN,
                reason_codes=[reason],
                prediction=None,
                top3=_top3_candidates(metadata, candidate_indices, candidate_scores),
                confidence=float(batch.approval_scores[index]),
            )
        items.append(item)
    return items


def _recapture_item(ordinal: int, bbox: BoundingBox, confidence: float) -> ScanItem:
    return ScanItem(
        segmentation_id=f"segmentation_{ordinal:03d}",
        bbox=bbox,
        status=ItemStatus.SEGMENT_RECAPTURE,
        reason_codes=["SEGMENT_RECAPTURE_REQUIRED"],
        prediction=None,
        top3=[],
        confidence=confidence,
    )


def summarize_reason_codes(items: list[ScanItem]) -> list[str]:
    reason_codes: list[str] = []
    if any(
        item.status is ItemStatus.UNKNOWN and item.reason_codes != ["DETECTOR_CONTAINED_DUPLICATE"]
        for item in items
    ):
        reason_codes.append("SEGMENT_BELOW_APPROVAL_THRESHOLD")
    if any("DETECTOR_CONTAINED_DUPLICATE" in item.reason_codes for item in items):
        reason_codes.append("SEGMENT_DUPLICATE_REVIEW_REQUIRED")
    if any(item.status is ItemStatus.SEGMENT_RECAPTURE for item in items):
        reason_codes.append("SEGMENT_RECAPTURE_REQUIRED")
    return reason_codes


__all__ = ["build_scan_items", "summarize_reason_codes"]
