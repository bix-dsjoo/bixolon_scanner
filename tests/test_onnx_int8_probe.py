import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bixolon_scanner.experiments.bread.onnx_int8_probe import (
    CalibrationReader,
    load_calibration_tensors,
)


def test_calibration_reader_rewinds() -> None:
    tensor = np.zeros((1, 3, 32, 32), dtype=np.float32)
    reader = CalibrationReader("pixel_values", [tensor])

    assert reader.get_next() == {"pixel_values": tensor}
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() == {"pixel_values": tensor}


def test_calibration_images_are_checksum_locked_and_preprocessed(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir()
    Image.new("RGB", (40, 20), (128, 64, 32)).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_path": "images/sample.png", "image_sha256": digest}) + "\n",
        encoding="utf-8",
    )

    tensors = load_calibration_tensors(
        manifest,
        tmp_path,
        input_size=32,
        maximum_samples=1,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    assert len(tensors) == 1
    assert tensors[0].shape == (1, 3, 32, 32)
    assert tensors[0].dtype == np.float32
