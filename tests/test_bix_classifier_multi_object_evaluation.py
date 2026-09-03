from __future__ import annotations

import json

from bixolon_scanner.experiments.bread.bix_classifier_multi_object_evaluation import (
    _load_detector_entries,
    _load_gt_entries,
)


def test_load_gt_entries_preserves_image_group_and_label(tmp_path) -> None:
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "../multi_object_scenes/easy/easy_001.jpg",
                    }
                ],
                "annotations": [
                    {"id": 2, "image_id": 1, "category_id": 3, "bbox": [5, 6, 7, 8]},
                    {"id": 1, "image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4]},
                ],
                "categories": [
                    {"id": 2, "name": "Croffle"},
                    {"id": 3, "name": "Waffle"},
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = _load_gt_entries(annotations)

    assert [row["annotation_id"] for row in rows] == [1, 2]
    assert rows[0]["target"] == 1
    assert rows[0]["bbox_xyxy"] == (1.0, 2.0, 4.0, 6.0)
    assert rows[1]["boxes"] == rows[0]["boxes"]
    assert rows[0]["image_path"] == "multi_object_scenes/easy/easy_001.jpg"


def test_load_detector_entries_uses_fixed_trace_boxes_and_matched_targets(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "image_id": 1,
                "status": "SEGMENTATION",
                "decision": {
                    "segmentations": [
                        {
                            "segmentation_id": "segmentation_001",
                            "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                        }
                    ]
                },
                "matched_classifier_diagnostics": [
                    {"detection_index": 0, "target_class_id": "bread_06"}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_detector_entries(
        trace,
        {1: "multi_object_scenes/hard/hard_001.jpg"},
    )

    assert len(rows) == 1
    assert rows[0]["target"] == 5
    assert rows[0]["bbox_xyxy"] == (10.0, 20.0, 40.0, 60.0)
    assert rows[0]["difficulty"] == "hard"
