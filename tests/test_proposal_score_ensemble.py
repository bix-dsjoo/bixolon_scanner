import pytest

from bixolon_scanner.experiments.bread.proposal_score_ensemble import (
    combine_ranked_predictions,
)


def _ranked(scores):
    return [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "scores": scores,
            "class_ids": [1, 2],
        }
    ]


def test_combine_ranked_predictions_supports_geometric_scores():
    combined = combine_ranked_predictions(
        _ranked([0.8, 0.2]),
        _ranked([0.2, 0.8]),
        left_weight=0.5,
        mode="geometric",
    )

    assert combined[0]["scores"] == pytest.approx([0.4, 0.4])


def test_combine_ranked_predictions_rejects_misaligned_boxes():
    right = _ranked([0.2, 0.8])
    right[0]["boxes_xyxy"][0][0] = 1

    with pytest.raises(ValueError, match="boxes differ"):
        combine_ranked_predictions(
            _ranked([0.8, 0.2]),
            right,
            left_weight=0.5,
            mode="arithmetic",
        )
