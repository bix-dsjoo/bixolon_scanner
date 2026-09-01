from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from bixolon_scanner.training.cache_detector import build_cache
from bixolon_scanner.training.ssdlite_objectness_detector import CachedObjectnessDataset


def test_detector_cache_applies_exif_orientation_before_resizing(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source = Image.new("RGB", (6, 4), (120, 80, 40))
    exif = Image.Exif()
    exif[274] = 6
    source.save(dataset_root / "oriented.jpg", exif=exif)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "record_type": "detection",
                "image_id": 1,
                "image_path": "oriented.jpg",
                "width": 4,
                "height": 6,
                "annotations": [{"bbox_xywh": [1, 2, 2, 3]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"

    build_cache(
        argparse.Namespace(
            manifest=manifest,
            dataset_root=dataset_root,
            output_dir=cache,
            image_size=12,
        )
    )

    metadata = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    assert metadata["source_shapes"] == {"1": [4, 6]}
    dataset = CachedObjectnessDataset(
        manifest,
        dataset_root,
        cache,
        training=False,
    )
    image, target = dataset[0]
    assert image.shape == (3, 12, 12)
    np.testing.assert_allclose(
        target["boxes"].numpy(),
        np.asarray([[3.0, 4.0, 9.0, 10.0]], dtype=np.float32),
    )


def test_objectness_dataset_accepts_operational_bbox_shape_metadata(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(dataset_root / "images" / "sample.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_id": 7,
                "image_path": "images/sample.png",
                "annotations": [{"bbox": [2, 1, 10, 5]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    build_cache(
        argparse.Namespace(
            manifest=manifest,
            dataset_root=dataset_root,
            output_dir=cache,
            image_size=20,
        )
    )

    dataset = CachedObjectnessDataset(
        manifest,
        dataset_root,
        cache,
        training=False,
    )
    _image, target = dataset[0]
    np.testing.assert_allclose(
        target["boxes"].numpy(),
        np.asarray([[2.0, 2.0, 12.0, 12.0]], dtype=np.float32),
    )
