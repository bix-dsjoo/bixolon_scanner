from bixolon_scanner.experiments.bread.rfdetr_box_evaluation import select_candidate


def test_rfdetr_box_candidate_minimizes_total_error_then_false_negative():
    candidates = [
        {
            "score_threshold": 0.1,
            "metrics": {
                "false_positive_count": 2,
                "false_negative_count": 1,
                "exact_image_rate": 0.8,
            },
        },
        {
            "score_threshold": 0.2,
            "metrics": {
                "false_positive_count": 1,
                "false_negative_count": 2,
                "exact_image_rate": 0.8,
            },
        },
        {
            "score_threshold": 0.3,
            "metrics": {
                "false_positive_count": 4,
                "false_negative_count": 0,
                "exact_image_rate": 0.9,
            },
        },
    ]

    assert select_candidate(candidates)["score_threshold"] == 0.1
