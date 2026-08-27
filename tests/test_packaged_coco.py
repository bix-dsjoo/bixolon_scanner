from __future__ import annotations

import json

from bixolon_scanner.evaluation.packaged_coco import evaluate_packaged_coco


def test_packaged_coco_scores_approved_and_empty_recapture(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    trace = tmp_path / "trace.jsonl"
    manifest_rows = [
        {
            "image_id": 1,
            "annotations": [{"bbox": [10, 10, 20, 20], "category_id": 2}],
        },
        {"image_id": 2, "annotations": []},
    ]
    trace_rows = [
        {
            "image_id": 1,
            "response": {
                "status": "SEGMENTATION",
                "segmentations": [
                    {
                        "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                        "status": "APPROVED",
                        "prediction": {"class_id": "bread_02"},
                        "top3": [],
                    }
                ],
            },
        },
        {"image_id": 2, "response": {"status": "IMAGE_RECAPTURE", "segmentations": []}},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
    trace.write_text("".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8")

    report = evaluate_packaged_coco(manifest, trace)

    assert report["counts"]["ground_truth_count"] == 1
    assert report["counts"]["matched_count"] == 1
    assert report["counts"]["false_negative_count"] == 0
    assert report["counts"]["false_positive_count"] == 0
    assert report["counts"]["approved_wrong_count"] == 0
    assert report["counts"]["empty_image_recapture_count"] == 1
