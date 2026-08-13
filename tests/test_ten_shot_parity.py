from __future__ import annotations

import numpy as np

from bixolon_scanner.training.ten_shot_parity import strict_classifier_parity


def test_strict_parity_requires_ordered_top3_and_state_equality():
    pytorch = np.asarray([[9.0, 8.0, 7.0, 1.0], [2.0, 5.0, 3.0, 1.0]])
    report = strict_classifier_parity(
        pytorch_logits=pytorch,
        cpu_logits=pytorch + 1e-5,
        cuda_logits=pytorch - 1e-5,
        temperature=1.0,
        approval_threshold=0.5,
        pytorch_onnx_tolerance=1e-3,
        cross_provider_tolerance=1e-3,
    )
    assert report["passes"] is True
    changed = pytorch.copy()
    changed[0, 1], changed[0, 2] = changed[0, 2], changed[0, 1]
    report = strict_classifier_parity(
        pytorch_logits=pytorch,
        cpu_logits=pytorch,
        cuda_logits=changed,
        temperature=1.0,
        approval_threshold=0.5,
        pytorch_onnx_tolerance=2.0,
        cross_provider_tolerance=2.0,
    )
    assert report["passes"] is False
    assert report["checks"]["top3_set_and_order_equal"] is False
