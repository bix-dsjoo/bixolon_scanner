import json
from pathlib import Path

from PIL import Image

from bixolon_scanner.experiments.bread.development_identity_manifest import (
    build_development_identity_manifest,
)


def test_development_identity_manifest_records_portable_image_identity(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images = dataset_root / "images"
    annotations = dataset_root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    image_path = images / "sample.png"
    Image.new("RGB", (16, 12), (80, 120, 160)).save(image_path)
    annotation_path = annotations / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 7,
                        "file_name": "../images/sample.png",
                        "width": 16,
                        "height": 12,
                    }
                ],
                "annotations": [],
                "categories": [],
            }
        ),
        encoding="utf-8",
    )

    rows = build_development_identity_manifest(
        dataset_root=dataset_root,
        annotation_path=annotation_path,
        dataset_version="development-v1",
        evaluation_set="rejected_test_as_development",
    )

    assert rows[0]["image_id"] == 7
    assert rows[0]["image_path"] == "../images/sample.png"
    assert len(str(rows[0]["image_sha256"])) == 64
    assert isinstance(rows[0]["perceptual_hash"], int)
