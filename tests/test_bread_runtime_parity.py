from copy import deepcopy

from bixolon_scanner.evaluation.bread_runtime_parity import compare_runtime_traces


def _trace() -> list[dict]:
    return [
        {
            "image_id": 1,
            "status": "SEGMENTATION",
            "reason_codes": [],
            "segmentations": [
                {
                    "bbox": {"x": 10, "y": 20, "width": 100, "height": 80},
                    "status": "APPROVED",
                    "reason_codes": [],
                    "prediction": {"class_id": "bread_01", "class_name": "bread_01"},
                    "top3": [],
                    "confidence": 0.99,
                }
            ],
            "worker_version": "1.1.0",
            "detector_version": "1.1.0",
            "classifier_version": "1.1.0",
        }
    ]


def test_runtime_parity_accepts_exact_final_decisions() -> None:
    report = compare_runtime_traces(_trace(), deepcopy(_trace()))

    assert report["passes"] is True
    assert report["final_status_class_rank_parity_exact"] is True
    assert report["minimum_observed_bbox_iou"] == 1.0


def test_runtime_parity_rejects_class_or_bbox_changes() -> None:
    changed = deepcopy(_trace())
    changed[0]["segmentations"][0]["prediction"]["class_id"] = "bread_02"
    changed[0]["segmentations"][0]["bbox"]["x"] = 12

    report = compare_runtime_traces(_trace(), changed)

    assert report["passes"] is False
    assert report["decision_mismatch_image_ids"] == [1]
    assert report["bbox_mismatch_image_ids"] == [1]
