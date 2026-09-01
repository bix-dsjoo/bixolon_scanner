import numpy as np

from bixolon_scanner.experiments.bread.dinov3_count_oof import oof_prediction_rows


def test_oof_prediction_rows_preserve_fold_and_ranked_probabilities() -> None:
    rows = oof_prediction_rows(
        [
            {
                "image_id": 7,
                "image_path": "image.jpg",
                "fold": 2,
                "count_label": 4,
            }
        ],
        np.asarray([[0.0, 2.0]], dtype=np.float32),
        np.asarray([3, 4], dtype=np.int64),
        temperature=1.0,
    )

    assert rows[0]["fold"] == 2
    assert rows[0]["expected_count"] == 4
    assert rows[0]["predicted_count"] == 4
    assert rows[0]["probabilities"]["4"] > rows[0]["probabilities"]["3"]
