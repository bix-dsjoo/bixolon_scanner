from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from bixolon_scanner.training.bix_bakery_multi import prepare_bix_bakery_multi_object


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_bix_bakery_multi_object_writes_verified_manifest(tmp_path: Path) -> None:
    root = tmp_path / "bix_bakery_dataset"
    image_root = root / "multi_object"
    image_root.mkdir(parents=True)
    images = []
    for index in range(20):
        path = image_root / f"scene_{index:02d}.jpg"
        Image.new("RGB", (40, 30), (240, 240, 240)).save(path)
        images.append(
            {
                "image": path.name,
                "sha256": _sha256(path),
                "objects": [
                    {
                        "category_id": index + 1,
                        "bbox_xyxy": [5, 4, 25, 24],
                    }
                ],
            }
        )
    config = tmp_path / "annotations.json"
    config.write_text(
        json.dumps({"schema_version": "1.0", "images": images}),
        encoding="utf-8",
    )

    metadata = prepare_bix_bakery_multi_object(root, config, tmp_path / "prepared")

    assert metadata["image_count"] == 20
    assert metadata["annotation_count"] == 20
    assert metadata["external_training_images_accessed"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["image_path"] == "multi_object/scene_00.jpg"
    assert rows[-1]["annotations"][0]["category_id"] == 20
    assert len(list((tmp_path / "prepared" / "qa").glob("*.jpg"))) == 20
