from bixolon_scanner.experiments.bread.proposal_failure_analysis import (
    analyze_error_rows,
)


def test_failure_analysis_separates_score_and_nms_failures():
    records = [
        {
            "image_id": 1,
            "annotations": [
                {"annotation_id": 1, "category_id": 1, "bbox_xywh": [0, 0, 10, 10]},
                {"annotation_id": 2, "category_id": 2, "bbox_xywh": [20, 0, 10, 10]},
            ],
        }
    ]
    ranked = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
            "scores": [0.8, 0.4],
            "class_ids": [1, 2],
        }
    ]
    errors = [
        {
            "image_id": 1,
            "fold": 0,
            "difficulty": "TEST",
            "false_negatives": [
                {"annotation_id": 1, "category_id": 1, "bbox_xywh": [0, 0, 10, 10]},
                {"annotation_id": 2, "category_id": 2, "bbox_xywh": [20, 0, 10, 10]},
            ],
            "false_positives": [],
        }
    ]

    analysis = analyze_error_rows(records, ranked, errors, score_threshold=0.5)

    assert analysis["false_negative_stage_counts"] == {
        "score_rejected": 1,
        "nms_or_assignment": 1,
    }
