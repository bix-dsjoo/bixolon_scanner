from bixolon_scanner.experiments.bread.detector_union_coverage import union_coverage


def test_union_coverage_combines_complementary_scales():
    records = [
        {
            "image_id": 1,
            "annotations": [
                {"annotation_id": 1, "category_id": 1, "bbox_xywh": [0, 0, 10, 10]},
                {"annotation_id": 2, "category_id": 2, "bbox_xywh": [20, 20, 10, 10]},
            ],
        }
    ]
    first = {"1": {"boxes_xyxy": [[0, 0, 10, 10]], "scores": [0.9]}}
    second = {"1": {"boxes_xyxy": [[20, 20, 30, 30]], "scores": [0.8]}}

    report = union_coverage(
        records,
        [first, second],
        score_threshold=0.1,
        match_iou_threshold=0.5,
    )

    assert report["covered_count"] == 2
    assert report["miss_count"] == 0
    assert report["recall_upper_bound"] == 1.0
