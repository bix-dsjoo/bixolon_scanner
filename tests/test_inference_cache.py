from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from bixolon_scanner.runtime.inference_cache import exact_embedding, reuse_verifier_embeddings


def test_exact_reuse_is_request_local_and_outputs_cannot_poison_it():
    calls = []

    def compute(batch):
        calls.append(batch.copy())
        return batch + 1

    owner = object()
    batch = np.ones((1, 3), dtype=np.float32)
    with reuse_verifier_embeddings():
        result = exact_embedding(owner, batch, compute)
        result[:] = -999
        result = exact_embedding(owner, batch.copy(), compute)
        np.testing.assert_array_equal(result, batch + 1)
        result[:] = -888
        np.testing.assert_array_equal(exact_embedding(owner, batch, compute), batch + 1)
        assert len(calls) == 1
        exact_embedding(object(), batch, compute)
        exact_embedding(owner, batch + 0.00001, compute)
        exact_embedding(owner, batch.reshape(3, 1), compute)
        exact_embedding(owner, batch.astype(np.float64), compute)
        assert len(calls) == 5
    with reuse_verifier_embeddings():
        exact_embedding(owner, batch, compute)
    exact_embedding(owner, batch, compute)
    exact_embedding(owner, batch, compute)
    assert len(calls) == 8


def test_failure_is_not_cached_and_nested_scope_restores_outer():
    owner, batch = object(), np.ones((1, 3))
    attempts = []

    def compute(batch):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("failed")
        return batch * len(attempts)

    with reuse_verifier_embeddings():
        with pytest.raises(RuntimeError):
            exact_embedding(owner, batch, compute)
        np.testing.assert_array_equal(exact_embedding(owner, batch, compute), batch * 2)
        with reuse_verifier_embeddings():
            np.testing.assert_array_equal(exact_embedding(owner, batch, compute), batch * 3)
        np.testing.assert_array_equal(exact_embedding(owner, batch, compute), batch * 2)


def test_parallel_requests_and_memory_bounds():
    owner, batch = object(), np.zeros((1, 3))

    def request(value):
        with reuse_verifier_embeddings():
            return exact_embedding(owner, batch, lambda _: batch + value)

    with ThreadPoolExecutor(2) as executor:
        outputs = list(executor.map(request, [1, 2]))
    for value, output in enumerate(outputs, start=1):
        np.testing.assert_array_equal(output, batch + value)
    calls = []

    def compute(batch):
        calls.append(1)
        return batch

    with reuse_verifier_embeddings():
        for value in range(20):
            exact_embedding(owner, batch + value, compute)
        exact_embedding(owner, batch + 19, compute)
        large = np.zeros(1_048_577, dtype=np.uint8)
        exact_embedding(owner, large, compute)
        exact_embedding(owner, large, compute)
    assert len(calls) == 23
