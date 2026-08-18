import numpy as np
import pytest

from bixolon_scanner.experiments.bread.classifier_logit_stabilization import (
    stabilize_logits,
)


def test_stabilize_logits_applies_global_class_bias_without_quantization() -> None:
    values = np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32)

    actual = stabilize_logits(
        values,
        logit_quantum=None,
        logit_phase=0.0,
        tie_break_bias_span=0.02,
    )

    np.testing.assert_allclose(actual, [[2.0, 1.99, 1.98]])


def test_stabilize_logits_rejects_phase_without_quantization() -> None:
    with pytest.raises(ValueError, match="phase requires"):
        stabilize_logits(
            np.zeros((1, 3), dtype=np.float32),
            logit_quantum=None,
            logit_phase=0.01,
            tie_break_bias_span=0.0,
        )
