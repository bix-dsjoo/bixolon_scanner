from __future__ import annotations

import json

from PIL import Image

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.scanner_v2_development_identity import build_identity_lineage
from bixolon_scanner.evaluation.scanner_v2_private_preflight import difference_hash


def test_development_identity_lineage_recomputes_missing_hash_and_deduplicates(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), (100, 120, 140)).save(image)
    image_sha256 = sha256_file(image)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"image_path": "image.jpg", "image_sha256": image_sha256}) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "image_path": "image.jpg",
                "image_sha256": image_sha256,
                "perceptual_hash": difference_hash(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "lineage.jsonl"

    report = build_identity_lineage([first, second], tmp_path, output)

    assert report["unique_image_count"] == 1
    assert json.loads(output.read_text()) == {
        "image_sha256": image_sha256,
        "perceptual_hash": f"{difference_hash(image):016x}",
    }
