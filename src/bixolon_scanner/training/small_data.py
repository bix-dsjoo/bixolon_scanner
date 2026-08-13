from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticHead:
    weights: np.ndarray
    bias: np.ndarray
    iterations: int
    classes: np.ndarray


@dataclass(frozen=True)
class PrototypeHead:
    weights: np.ndarray
    bias: np.ndarray
    counts: np.ndarray
    classes: np.ndarray


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def apply_layer_norm(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=-1, keepdims=True)
    variance = values.var(axis=-1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + float(epsilon))
    return normalized * np.asarray(weight, dtype=np.float32) + np.asarray(bias, dtype=np.float32)


def fit_cosine_prototype_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> PrototypeHead:
    """Build nearest-class-mean cosine weights without fitting an optimizer."""
    values = l2_normalize(np.asarray(features, dtype=np.float32))
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("prototype features must have shape [samples, channels]")
    if len(values) != len(targets):
        raise ValueError("prototype feature and label counts differ")
    expected = np.arange(num_classes, dtype=np.int64)
    observed = np.unique(targets)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"prototype construction requires classes {expected.tolist()}, got {observed.tolist()}"
        )
    counts = np.bincount(targets, minlength=num_classes).astype(np.int64)
    prototypes = np.stack([values[targets == class_index].mean(axis=0) for class_index in expected])
    return PrototypeHead(
        weights=l2_normalize(prototypes),
        bias=np.zeros(num_classes, dtype=np.float32),
        counts=counts,
        classes=expected,
    )


def brightness_c2_frofa(
    patch_features: np.ndarray,
    *,
    magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply per-channel brightness c²FroFA in the mapped feature range."""
    if not 0.0 <= magnitude <= 1.0:
        raise ValueError("FroFA brightness magnitude must be between 0 and 1")
    values = np.asarray(patch_features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("patch_features must have shape [batch, patches, channels]")
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    span = np.maximum(maximum - minimum, 1e-6)
    mapped = (values - minimum) / span
    delta = rng.uniform(-magnitude, magnitude, size=(len(values), 1, values.shape[-1])).astype(
        np.float32
    )
    augmented = np.clip(mapped + delta, 0.0, 1.0)
    return augmented * span + minimum


def build_frofa_training_set(
    patch_features: np.ndarray,
    labels: np.ndarray,
    *,
    layer_norm_weight: np.ndarray,
    layer_norm_bias: np.ndarray,
    layer_norm_epsilon: float,
    magnitude: float,
    views: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if views < 0:
        raise ValueError("FroFA views must be non-negative")
    patches = np.asarray(patch_features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if len(patches) != len(targets):
        raise ValueError("patch feature and label counts differ")
    rng = np.random.default_rng(seed)

    def pool(values: np.ndarray) -> np.ndarray:
        pooled = apply_layer_norm(
            values.mean(axis=1),
            layer_norm_weight,
            layer_norm_bias,
            epsilon=layer_norm_epsilon,
        )
        return l2_normalize(pooled)

    feature_parts = [pool(patches)]
    label_parts = [targets]
    for _ in range(views):
        feature_parts.append(pool(brightness_c2_frofa(patches, magnitude=magnitude, rng=rng)))
        label_parts.append(targets)
    return np.concatenate(feature_parts), np.concatenate(label_parts)


def fit_logistic_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    regularization_c: float,
    max_iterations: int,
    seed: int,
) -> LogisticHead:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - training extra contract
        raise RuntimeError("install the 'training' extra for logistic fitting") from exc
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    expected = np.arange(num_classes, dtype=np.int64)
    observed = np.unique(targets)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"logistic training requires classes {expected.tolist()}, got {observed.tolist()}"
        )
    if regularization_c <= 0:
        raise ValueError("logistic regularization C must be positive")
    classifier = LogisticRegression(
        C=float(regularization_c),
        solver="lbfgs",
        max_iter=int(max_iterations),
        random_state=int(seed),
    ).fit(values, targets)
    iterations = int(np.max(classifier.n_iter_))
    if iterations >= max_iterations:
        raise RuntimeError("logistic optimizer did not converge")
    weights = np.asarray(classifier.coef_, dtype=np.float32)
    bias = np.asarray(classifier.intercept_, dtype=np.float32)
    if num_classes == 2 and len(weights) == 1:
        weights = np.concatenate((-weights / 2.0, weights / 2.0), axis=0)
        bias = np.concatenate((-bias / 2.0, bias / 2.0), axis=0)
    return LogisticHead(
        weights=weights,
        bias=bias,
        iterations=iterations,
        classes=np.asarray(classifier.classes_, dtype=np.int64),
    )


def fit_linear_svm_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    regularization_c: float,
    max_iterations: int,
    seed: int,
) -> LogisticHead:
    try:
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - training extra contract
        raise RuntimeError("install the 'training' extra for linear SVM fitting") from exc
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    expected = np.arange(num_classes, dtype=np.int64)
    observed = np.unique(targets)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"linear SVM training requires classes {expected.tolist()}, got {observed.tolist()}"
        )
    if regularization_c <= 0:
        raise ValueError("linear SVM regularization C must be positive")
    classifier = LinearSVC(
        C=float(regularization_c),
        dual="auto",
        max_iter=int(max_iterations),
        random_state=int(seed),
    ).fit(values, targets)
    iterations = int(np.max(classifier.n_iter_))
    if iterations >= max_iterations:
        raise RuntimeError("linear SVM optimizer did not converge")
    weights = np.asarray(classifier.coef_, dtype=np.float32)
    bias = np.asarray(classifier.intercept_, dtype=np.float32)
    if num_classes == 2 and len(weights) == 1:
        weights = np.concatenate((-weights / 2.0, weights / 2.0), axis=0)
        bias = np.concatenate((-bias / 2.0, bias / 2.0), axis=0)
    return LogisticHead(
        weights=weights,
        bias=bias,
        iterations=iterations,
        classes=np.asarray(classifier.classes_, dtype=np.int64),
    )
