from bixolon_scanner.experiments.bread.oof_difficulty_report import (
    summarize_difficulty_bucket,
)


def test_summary_reports_image_detector_and_classifier_rates():
    result = summarize_difficulty_bucket(
        image_count=100,
        image_recapture_count=3,
        ground_truth_count=500,
        accepted_detector_metrics={
            "false_positive_count": 0,
            "false_negative_count": 0,
        },
        raw_detector_metrics={
            "false_positive_count": 0,
            "false_negative_count": 2,
        },
        approved_count=390,
        approved_error_count=0,
        unknown_count=50,
        unknown_top3_miss_count=0,
        segment_recapture_count=40,
    )

    assert result["images"]["segmentation"] == 97
    assert result["images"]["image_recapture_rate"] == 0.03
    assert result["detector"]["fn_per_segmentation_image"] == 0.0
    assert result["classifier"]["end_to_end_approved_rate"] == 390 / 500
    assert result["classifier"]["accepted_matched_approved_rate_diagnostic"] == 390 / 480
    assert result["classifier"]["segment_recapture_rate"] == 40 / 480
