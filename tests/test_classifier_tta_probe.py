from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.archive.bread_1_0_0.classifier_tta_probe import fusion_candidates


def test_fusion_candidates_can_select_complementary_views():
    logits = {
        0.8: np.asarray([[3.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        1.0: np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32),
    }
    targets = np.asarray([0, 1], dtype=np.int64)
    config = {"inference": {}}

    candidates = fusion_candidates(logits, targets, config)

    assert candidates[0]["top1_accuracy"] == 1.0
