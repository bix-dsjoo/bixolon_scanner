from bixolon_scanner.experiments.bread.detector_policy_consensus import (
    policy_agreement_count,
)


def _row(image_id: int, boxes: list[list[float]]) -> dict[str, object]:
    return {
        "image_id": image_id,
        "boxes_xyxy": boxes,
        "scores": [0.9] * len(boxes),
    }


def test_policy_agreement_count_requires_count_and_one_to_one_geometry() -> None:
    primary = _row(1, [[0, 0, 10, 10], [20, 20, 30, 30]])
    policies = [
        primary,
        _row(1, [[1, 1, 11, 11], [20, 20, 30, 30]]),
        _row(1, [[0, 0, 10, 10]]),
        _row(1, [[50, 50, 60, 60], [70, 70, 80, 80]]),
    ]

    assert policy_agreement_count(primary, policies, iou_threshold=0.5) == 2
