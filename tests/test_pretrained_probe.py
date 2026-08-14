import numpy as np
from PIL import Image

from bixolon_scanner.experiments.archive.bread_1_0_0.pretrained_probe import (
    _evaluation_samples,
    _metrics,
    _training_views,
)


def test_pretrained_probe_training_views_are_derived_from_one_original():
    views = _training_views(Image.new("RGB", (32, 24), (120, 100, 80)))

    assert len(views) == 6
    assert all(view.size == (32, 24) for view in views)


def test_pretrained_probe_reports_top1_and_top3_by_dataset():
    metrics = _metrics(
        np.asarray([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0]], dtype=np.float32),
        np.asarray([0, 2]),
        ["multi", "scan"],
    )

    assert metrics["ALL"]["top1_accuracy"] == 0.5
    assert metrics["ALL"]["top3_accuracy"] == 1.0
    assert metrics["SCAN"]["top1_error_count"] == 1


def test_pretrained_probe_defaults_to_multi_object_scenes_only():
    assert _evaluation_samples.__defaults__ == (("multi_object_scenes",),)
