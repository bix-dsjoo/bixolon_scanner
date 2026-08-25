import numpy as np

from bixolon_scanner.experiments.bread.classifier_source_comparison import (
    prototype_scores,
    ranking_metrics,
    select_source,
)


def test_prototype_source_comparison_uses_cosine_rankings() -> None:
    support = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    targets = np.asarray([0, 1])
    evaluation = np.asarray([[3.0, 0.1], [0.1, 3.0]])

    scores = prototype_scores(support, targets, evaluation, class_count=2)

    assert ranking_metrics(scores, targets)["top1_error_count"] == 0


def test_source_selection_uses_common_top1_then_top3_metrics() -> None:
    selected = select_source(
        [
            {
                "source": "single_objects",
                "metrics": {"top1_error_count": 2, "top3_miss_count": 0},
            },
            {
                "source": "single_objects_3",
                "metrics": {"top1_error_count": 1, "top3_miss_count": 1},
            },
        ]
    )

    assert selected == "single_objects_3"
