from bixolon_scanner.evaluation.packaged_profile_summary import Outcome


def test_profile_outcome_separates_recapture_and_segmentation_metrics() -> None:
    outcome = Outcome()
    outcome.add(
        {"annotations": [{"bbox": [0, 0, 10, 10], "category_id": 1}]},
        {
            "client_elapsed_ms": 12.0,
            "response": {"status": "IMAGE_RECAPTURE", "processing_time_ms": 10.0},
        },
        0.5,
    )
    outcome.add(
        {"annotations": [{"bbox": [0, 0, 10, 10], "category_id": 2}]},
        {
            "client_elapsed_ms": 22.0,
            "response": {
                "status": "SEGMENTATION",
                "processing_time_ms": 20.0,
                "segmentations": [
                    {
                        "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
                        "status": "UNKNOWN",
                        "top3": [{"class_id": "bread_01"}],
                    }
                ],
            },
        },
        0.5,
    )

    rendered = outcome.render()
    counts = rendered["counts"]
    assert counts["image_count"] == 2
    assert counts["image_recapture_count"] == 1
    assert counts["recaptured_ground_truth_count"] == 1
    assert counts["false_negative_count"] == 0
    assert counts["unknown_candidate_out_count"] == 1
    assert rendered["rates"]["unknown_output_rate"] == 1.0
    assert rendered["performance"]["worker_processing_all"]["mean_ms"] == 15.0
