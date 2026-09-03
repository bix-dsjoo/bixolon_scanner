from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.bix_classifier_cascade import (
    bix_source_view_fold,
    route_predictions,
    validate_source_directory,
)


def test_bix_source_view_fold_covers_ten_views() -> None:
    views = [
        ("normal", "vertical"),
        ("flipped", "vertical"),
        ("normal", "ground_60_dir_02"),
        ("flipped", "ground_60_dir_02"),
        ("normal", "ground_30_dir_01"),
        ("flipped", "ground_30_dir_01"),
        ("normal", "ground_60_dir_01"),
        ("normal", "ground_30_dir_02"),
        ("flipped", "ground_30_dir_02"),
        ("flipped", "ground_60_dir_01"),
    ]
    assert [bix_source_view_fold(side=side, view=view) for side, view in views].count(0) == 4
    assert [bix_source_view_fold(side=side, view=view) for side, view in views].count(1) == 3
    assert [bix_source_view_fold(side=side, view=view) for side, view in views].count(2) == 3


def test_route_predictions_uses_224_and_vit_only_for_uncertain_primary() -> None:
    logits_192 = np.asarray([[3.0, 0.0], [1.0, 0.9], [1.0, 0.9]])
    logits_224 = np.asarray([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    logits_vit = np.asarray([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    result = route_predictions(
        logits_192,
        logits_224,
        logits_vit,
        primary_margin=0.2,
        fallback_margin=0.05,
        verifier_margin=0.05,
    )
    assert result["fallback"].tolist() == [False, True, True]
    assert result["verifier"].tolist() == [False, True, False]
    assert result["approved"].tolist() == [True, True, False]
    assert result["predictions"].tolist() == [0, 0, 1]


def test_source_directory_lock_rejects_another_image_root(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    other = tmp_path / "other"
    config = {"dataset": {"allowed_root": str(allowed)}}
    validate_source_directory(config, allowed)
    with pytest.raises(ValueError, match="image access is locked"):
        validate_source_directory(config, other)
