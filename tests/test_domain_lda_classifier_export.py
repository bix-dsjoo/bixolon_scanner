from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.training.domain_lda_classifier_export import (
    transfer_margin_thresholds_by_quantile,
)


def test_transfer_margin_thresholds_preserves_rejection_count() -> None:
    oof = np.asarray([[5.0, 0.0], [4.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    final = np.asarray([[50.0, 0.0], [40.0, 0.0], [30.0, 0.0], [0.0, 50.0]])
    thresholds, diagnostics = transfer_margin_thresholds_by_quantile(oof, final, [4.0, None])
    assert thresholds[0] == pytest.approx(np.nextafter(np.float32(40.0), np.float32(np.inf)))
    assert thresholds[1] is None
    assert diagnostics[0]["oof_rejection_count"] == 2


def test_transfer_margin_thresholds_rejects_wrong_class_dimension() -> None:
    with pytest.raises(ValueError, match="same class dimension"):
        transfer_margin_thresholds_by_quantile(np.ones((2, 2)), np.ones((2, 3)), [None, None])
