from threading import Event

import numpy as np
import pytest

from bixolon_scanner.runtime.inference_cache import exact_embedding, reuse_verifier_embeddings
from bixolon_scanner.runtime.parallel_inference import verification_pair


def test_parallel_evidence_overlaps_and_inherits_request_cache():
    started = Event()
    owner, tensor, calls = object(), np.ones((1, 3)), []

    def independent():
        started.set()
        return exact_embedding(owner, tensor, lambda value: calls.append(1) or value + 1)

    def rotation():
        assert started.wait(2)
        return tensor + 2

    with reuse_verifier_embeddings():
        a, b = verification_pair(rotation, independent, parallel=True)
        np.testing.assert_array_equal(a, tensor + 2)
        np.testing.assert_array_equal(b, tensor + 1)
        exact_embedding(owner, tensor, lambda _: pytest.fail("cache was not inherited"))
    assert len(calls) == 1


def test_error_waits_for_other_evidence_before_resource_recovery():
    release, finished = Event(), Event()

    def independent():
        assert release.wait(2)
        finished.set()
        return 1

    def rotation():
        release.set()
        raise RuntimeError("GPU failed")

    with pytest.raises(RuntimeError, match="GPU failed"):
        verification_pair(rotation, independent, parallel=True)
    assert finished.is_set()


def test_sequential_mode_and_independent_failure():
    order = []
    assert verification_pair(
        lambda: order.append("rotation") or 1,
        lambda: order.append("independent") or 2,
        parallel=False,
    ) == (1, 2)
    assert order == ["rotation", "independent"]

    def fail():
        raise RuntimeError("CPU failed")

    with pytest.raises(RuntimeError, match="CPU failed"):
        verification_pair(lambda: 1, fail, parallel=True)
