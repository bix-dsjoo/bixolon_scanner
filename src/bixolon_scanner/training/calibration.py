from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    approved_count: int
    approved_precision: float
    coverage: float
    false_approval_rate_upper: float
    risk_control_satisfied: bool


def binomial_rate_upper_bound(errors: int, count: int, confidence_level: float = 0.95) -> float:
    if count <= 0:
        return 1.0
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    from scipy.stats import beta

    if errors >= count:
        return 1.0
    return float(beta.ppf(confidence_level, errors + 1, count - errors))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    def nll(log_temperature: float) -> float:
        probabilities = softmax(logits, float(np.exp(log_temperature)))
        selected = probabilities[np.arange(len(targets)), targets]
        return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())

    result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def select_approval_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    max_false_approval_rate: float = 0.005,
    confidence_level: float | None = 0.95,
) -> ThresholdResult:
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    candidates = sorted({float(value) for value in confidence}, reverse=True)
    best: ThresholdResult | None = None
    for threshold in candidates:
        approved = confidence >= threshold
        count = int(approved.sum())
        if count == 0:
            continue
        correct = int((predictions[approved] == targets[approved]).sum())
        errors = count - correct
        precision = correct / count
        false_rate = 1.0 - precision
        false_rate_upper = (
            binomial_rate_upper_bound(errors, count, confidence_level)
            if confidence_level is not None
            else false_rate
        )
        if false_rate_upper <= max_false_approval_rate:
            candidate = ThresholdResult(
                threshold,
                count,
                precision,
                count / len(targets),
                false_rate_upper,
                True,
            )
            if best is None or candidate.coverage > best.coverage:
                best = candidate
    if best is None:
        return ThresholdResult(1.0, 0, 1.0, 0.0, 1.0, False)
    return best


def topk_accuracy(probabilities: np.ndarray, targets: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-np.asarray(probabilities), axis=1)[:, :k]
    return float(np.any(top == np.asarray(targets)[:, None], axis=1).mean())
