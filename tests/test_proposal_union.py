from bixolon_scanner.experiments.bread.proposal_union import union_predictions


def _prediction(image_id, scores):
    return {
        "image_id": image_id,
        "boxes_xyxy": [[index * 20, 0, index * 20 + 10, 10] for index in range(len(scores))],
        "scores": scores,
        "class_ids": list(range(len(scores))),
    }


def test_union_predictions_preserves_source_provenance_after_filtering():
    union = union_predictions(
        [_prediction(1, [0.9, 0.1])],
        [_prediction(1, [0.8])],
        left_score_threshold=0.5,
        left_nms_iou_threshold=0.5,
        right_score_threshold=0.5,
        right_nms_iou_threshold=0.5,
    )

    assert union[0]["scores"] == [0.9, 0.8]
    assert union[0]["source_ids"] == [0, 1]
