from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.single2_augmentation_ablation import (
    _variant_cache,
    source_view_fold,
)


def test_source_view_fold_keeps_every_derived_view_with_its_source() -> None:
    assert source_view_fold({"view": "ground_0_dir_01", "side": "unpaired"}) == 0
    assert source_view_fold({"view": "vertical", "side": "normal"}) == 0
    assert source_view_fold({"view": "ground_30_dir_01", "side": "flipped"}) == 1
    assert source_view_fold({"view": "ground_60_dir_01", "side": "normal"}) == 1
    assert source_view_fold({"view": "ground_30_dir_02", "side": "normal"}) == 2
    assert source_view_fold({"view": "ground_60_dir_01", "side": "flipped"}) == 2


def test_balanced_small_selects_first_four_views_per_bank() -> None:
    features = {
        "clean": {
            "features": np.zeros((2, 3)),
            "labels": np.asarray([0, 1]),
            "folds": np.asarray([0, 1]),
            "views": np.asarray([0, 0]),
        },
        "appearance": {
            "features": np.zeros((8, 3)),
            "labels": np.zeros(8, dtype=np.int64),
            "folds": np.zeros(8, dtype=np.int64),
            "views": np.arange(8),
        },
        "probe": {
            "features": np.zeros((1, 3)),
            "labels": np.asarray([0]),
            "folds": np.asarray([2]),
            "views": np.asarray([0]),
        },
    }
    result = _variant_cache(features, ["clean", "appearance:first4"])
    assert result["train_features"].shape == (6, 3)
    assert result["train_labels"].tolist() == [0, 1, 0, 0, 0, 0]


def test_repeat_selector_changes_sampling_weight_without_new_images() -> None:
    bank = {
        "features": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "labels": np.asarray([0, 1]),
        "folds": np.asarray([0, 1]),
        "views": np.asarray([0, 0]),
    }
    result = _variant_cache({"clean": bank, "probe": bank}, ["clean:repeat4"])
    assert result["train_features"].shape == (8, 2)
    assert result["train_labels"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
