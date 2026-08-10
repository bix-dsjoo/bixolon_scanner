import json

from bixolon_scanner.inference import Detection
from bixolon_scanner.training.evaluate_difficulty import (
    _empty_counts,
    _finalize_counts,
    _load_records,
    _match_detections,
)


def test_match_detections_reports_matches_and_missed_ground_truth():
    detections = [
        Detection(10.0, 20.0, 40.0, 60.0, 0.99),
        Detection(200.0, 200.0, 220.0, 220.0, 0.9),
    ]
    annotations = [
        {"bbox": [10.0, 20.0, 30.0, 40.0]},
        {"bbox": [100.0, 100.0, 20.0, 20.0]},
    ]

    matches, missed = _match_detections(detections, annotations, 0.5)

    assert matches[0][0] == 0
    assert matches[0][1] == 1.0
    assert missed == {1}


def test_finalize_counts_uses_requested_error_denominators():
    counts = _empty_counts()
    counts.update(
        {
            "images": 10,
            "ground_truth_boxes": 20,
            "matched_boxes": 18,
            "missed_boxes": 2,
            "predicted_boxes": 21,
            "false_positive_boxes": 3,
            "approved_boxes": 12,
            "approved_correct": 10,
            "approved_wrong": 2,
            "approved_wrong_matched": 2,
            "unknown_matched_boxes": 5,
            "unknown_top3_correct": 4,
            "unknown_top3_missing": 1,
            "classified_matched_boxes": 17,
            "top1_correct": 12,
            "end_to_end_latency_ms_total": 650.0,
        }
    )
    counts["response_status"]["RECAPTURE"] = 2

    result = _finalize_counts(counts)

    assert result["rates"]["recapture_image_rate"] == 0.2
    assert result["rates"]["detector_box_failure_rate"] == 0.1
    assert result["rates"]["approved_wrong_rate"] == 2 / 12
    assert result["rates"]["approved_accuracy"] == 10 / 12
    assert result["rates"]["unknown_top3_missing_rate"] == 0.2
    assert result["rates"]["unknown_top3_accuracy"] == 0.8
    assert result["rates"]["classifier_top1_accuracy_excluding_recapture"] == 12 / 17
    assert result["all_ground_truth_box_outcomes"] == {
        "denominator": 20,
        "counts": {
            "recognized_approved_correct": 10,
            "top3_candidate": 4,
            "candidate_out": 1,
            "approved_misclassification": 2,
            "recapture_matched": 1,
            "segmentation_missed": 2,
        },
        "rates": {
            "recognized_approved_correct": 0.5,
            "top3_candidate": 0.2,
            "candidate_out": 0.05,
            "approved_misclassification": 0.1,
            "recapture_matched": 0.05,
            "segmentation_missed": 0.1,
        },
        "false_positive_boxes": 3,
        "false_positive_boxes_per_ground_truth": 0.15,
    }
    assert result["end_to_end_latency_ms"] == {"sample_count": 10, "mean": 65.0}


def test_load_records_supports_root_coco_with_difficulty_directories(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "E").mkdir()
    (tmp_path / "E" / "E_001.jpg").write_bytes(b"not decoded by record loader")
    coco = {
        "images": [{"id": 1, "file_name": "E/E_001.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}],
    }
    (tmp_path / "annotations" / "instances.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )

    records = _load_records(tmp_path)

    assert len(records) == 1
    assert records[0]["difficulty"] == "E"
    assert records[0]["group"] == "E"
    assert records[0]["image_path"] == tmp_path / "E" / "E_001.jpg"
