from __future__ import annotations

import json
from pathlib import Path

from bixolon_scanner.evaluation.scanner_017_safety_report import (
    _build_detector415_source_ground_truth,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_detector415_source_ground_truth_corrects_only_stale_categories(
    tmp_path: Path,
) -> None:
    multi_annotations = {
        "images": [
            {"id": 1, "file_name": "../multi_object_scenes/a.jpg", "width": 20, "height": 10}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 16,
                "bbox": [1, 2, 3, 4],
                "area": 12,
                "iscrowd": 0,
            }
        ],
    }
    operational_annotations = {
        "images": [{"id": 1, "file_name": "../images/b.jpg", "width": 30, "height": 15}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 8,
                "bbox": [2, 3, 4, 5],
                "area": 20,
                "iscrowd": 0,
            }
        ],
    }
    _write_json(
        tmp_path / "datasets/bread_dataset/annotations/multi_object_instances.json",
        multi_annotations,
    )
    _write_json(
        tmp_path
        / "datasets/bread_dataset/operational_collections/2026-08-18/annotations/instances.json",
        operational_annotations,
    )

    rows = [
        {
            "annotations": [
                {
                    "annotation_id": 1,
                    "area": 12,
                    "bbox_xywh": [1, 2, 3, 4],
                    "category_id": 17,
                    "iscrowd": 0,
                }
            ],
            "evaluation_set": "multi_object_scenes",
            "height": 10,
            "image_id": 1,
            "image_path": "multi_object_scenes/a.jpg",
            "source_image_id": 1,
            "width": 20,
        },
        {
            "annotations": [
                {
                    "annotation_id": 10_000_000_001,
                    "area": 20,
                    "bbox_xywh": [2, 3, 4, 5],
                    "category_id": 8,
                    "iscrowd": 0,
                }
            ],
            "evaluation_set": "operational_collections/2026-08-18",
            "height": 15,
            "image_id": 1_000_001,
            "image_path": "operational_collections/2026-08-18/images/b.jpg",
            "source_image_id": 1,
            "width": 30,
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(
        ("\r\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\r\n").encode()
    )
    corrected = tmp_path / "corrected.jsonl"

    audit = _build_detector415_source_ground_truth(tmp_path, manifest, corrected)

    assert audit["composition"] == {
        "multi_object_scenes": 1,
        "operational_collections/2026-08-18": 1,
    }
    assert audit["source_annotation_count"] == 2
    assert audit["structural_mismatch_count"] == 0
    assert audit["category_correction_count"] == 1
    assert audit["category_corrections"][0]["registry_class_id"] == "bread_17"
    assert audit["category_corrections"][0]["source_class_id"] == "bread_16"
    assert (
        audit["hashes"]["registry_worktree_raw_sha256"]
        != audit["hashes"]["registry_lf_normalized_sha256"]
    )
    assert b"\r\n" not in corrected.read_bytes()
    corrected_rows = [json.loads(line) for line in corrected.read_text().splitlines()]
    assert corrected_rows[0]["annotations"][0]["category_id"] == 16
    assert corrected_rows[1]["annotations"][0]["category_id"] == 8
