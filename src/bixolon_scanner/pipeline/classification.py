from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.model_package import ClassifierMetadata
from .ports import ClassificationResult


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class ClassifierBatch:
    probabilities: np.ndarray
    ranking_probabilities: np.ndarray
    decision_indices: np.ndarray
    approval_scores: np.ndarray
    approved: np.ndarray
    top3_unsafe: np.ndarray
    segment_recapture_reasons: tuple[str | None, ...] | None
    unknown_reasons: tuple[str | None, ...] | None
    uses_explicit_ranking_scores: bool


def normalize_classification(
    classification: np.ndarray | ClassificationResult,
    *,
    detection_count: int,
    metadata: ClassifierMetadata,
) -> ClassifierBatch:
    """Validate adapter output and derive policy-neutral batch decision inputs."""

    if isinstance(classification, ClassificationResult):
        logits = classification.logits
        ranking_logits = classification.ranking_logits
        approval_scores = classification.approval_scores
        top3_safety_scores = classification.top3_safety_scores
        ranking_scores = classification.ranking_scores
        segment_recapture_reasons = classification.segment_recapture_reasons
        unknown_reasons = classification.unknown_reasons
        approval_blocked = classification.approval_blocked
    else:
        logits = classification
        ranking_logits = classification
        approval_scores = None
        top3_safety_scores = None
        ranking_scores = None
        segment_recapture_reasons = None
        unknown_reasons = None
        approval_blocked = None

    expected_shape = (detection_count, len(metadata.labels))
    if logits.shape != expected_shape:
        raise ValueError("classifier output shape does not match package labels")
    if ranking_logits.shape != expected_shape:
        raise ValueError("classifier ranking output shape does not match logits")

    probabilities = softmax(logits, metadata.temperature)
    ranking_probabilities = (
        softmax(ranking_logits, metadata.temperature)
        if ranking_scores is None
        else np.asarray(ranking_scores, dtype=np.float32)
    )
    if ranking_probabilities.shape != expected_shape or np.any(
        (ranking_probabilities < 0.0) | (ranking_probabilities > 1.0)
    ):
        raise ValueError("classifier ranking scores must match labels and be in [0, 1]")
    decision_indices = np.argsort(-ranking_probabilities, axis=1, kind="stable")

    if approval_scores is None:
        normalized_approval_scores = probabilities.max(axis=1)
    else:
        normalized_approval_scores = np.asarray(approval_scores, dtype=np.float32)
    if normalized_approval_scores.shape != (detection_count,):
        raise ValueError("classifier approval scores do not match detections")
    if top3_safety_scores is not None:
        top3_safety_scores = np.asarray(top3_safety_scores, dtype=np.float32)
        if top3_safety_scores.shape != (detection_count,):
            raise ValueError("classifier Top-3 safety scores do not match detections")
    if segment_recapture_reasons is not None and len(segment_recapture_reasons) != detection_count:
        raise ValueError("classifier recapture reasons do not match detections")
    if segment_recapture_reasons is not None:
        segment_recapture_reasons = tuple(segment_recapture_reasons)
    if unknown_reasons is not None and len(unknown_reasons) != detection_count:
        raise ValueError("classifier unknown reasons do not match detections")
    if unknown_reasons is not None:
        unknown_reasons = tuple(unknown_reasons)
    if approval_blocked is not None:
        approval_blocked = np.asarray(approval_blocked, dtype=bool)
        if approval_blocked.shape != (detection_count,):
            raise ValueError("classifier approval blocks do not match detections")

    configured_thresholds = metadata.approval_thresholds
    if configured_thresholds is None:
        staged_policy = metadata.staged_inference
        default_threshold = (
            metadata.approval_threshold
            if staged_policy is None or staged_policy.approval_threshold is None
            else staged_policy.approval_threshold
        )
        approval_thresholds = np.full(detection_count, default_threshold, dtype=np.float32)
    else:
        approval_thresholds = np.asarray(
            [
                metadata.approval_threshold
                if configured_thresholds[int(indices[0])] is None
                else configured_thresholds[int(indices[0])]
                for indices in decision_indices
            ],
            dtype=np.float32,
        )
    approved = normalized_approval_scores >= approval_thresholds
    if approval_blocked is not None:
        approved &= ~approval_blocked

    mask_policy = metadata.neighbor_mask_inference
    staged_policy = metadata.staged_inference
    top3_safety_threshold = (
        mask_policy.top3_safety_threshold
        if mask_policy is not None
        else (staged_policy.top3_safety_threshold if staged_policy is not None else None)
    )
    top3_unsafe = (
        np.zeros(detection_count, dtype=bool)
        if top3_safety_scores is None or top3_safety_threshold is None
        else top3_safety_scores < top3_safety_threshold
    )
    return ClassifierBatch(
        probabilities=probabilities,
        ranking_probabilities=ranking_probabilities,
        decision_indices=decision_indices,
        approval_scores=normalized_approval_scores,
        approved=approved,
        top3_unsafe=top3_unsafe,
        segment_recapture_reasons=segment_recapture_reasons,
        unknown_reasons=unknown_reasons,
        uses_explicit_ranking_scores=ranking_scores is not None,
    )


__all__ = ["ClassifierBatch", "normalize_classification", "softmax"]
