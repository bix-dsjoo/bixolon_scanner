import numpy as np

from bixolon_scanner.evaluation.incremental_catalog import evaluate_append_only_consensus


def test_three_way_append_only_consensus_preserves_existing_outputs() -> None:
    supports = np.repeat(np.eye(3, dtype=np.float32), 2, axis=0).reshape(3, 2, 3)
    supports = supports.reshape(6, 3)
    evaluation = np.repeat(np.eye(3, dtype=np.float32), 2, axis=0)
    targets = np.repeat(np.arange(3, dtype=np.int64), 2)

    report = evaluate_append_only_consensus(
        (supports, supports, supports),
        (evaluation, evaluation, evaluation),
        targets,
        class_count=3,
        supports_per_class=2,
        alpha=0.01,
    )

    assert report["existing_decision_count"] == 12
    assert report["existing_output_change_count"] == 0
    assert report["existing_correct_to_incorrect_count"] == 0
    assert report["new_sku_selected_count"] == 6
    assert report["existing_output_preserved"] is True
    assert report["existing_performance_regression_free"] is True
