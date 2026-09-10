"""Bounded exact-input verifier reuse within one scan, never across requests."""

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_cache: ContextVar[dict | None] = ContextVar("verifier_embeddings", default=None)


@contextmanager
def reuse_verifier_embeddings():
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def exact_embedding(owner, batch: np.ndarray, compute) -> np.ndarray:
    cache = _cache.get()
    # Keep at most 16 small verifier batches; large batches use ordinary inference.
    if cache is None or batch.nbytes > 1_048_576:
        return compute(batch)
    key = (owner, batch.dtype.str, batch.shape, batch.tobytes())
    if key in cache:
        return cache[key].copy()
    result = compute(batch)
    if len(cache) < 16:
        cache[key] = result.copy()
    return result
