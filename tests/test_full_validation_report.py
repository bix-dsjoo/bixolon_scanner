from __future__ import annotations

from bixolon_scanner.experiments.bread.full_validation_report import (
    build_gate_summary,
)


def _summary(
    *,
    total_image_count: int = 400,
    total_ground_truth_object_count: int = 1000,
    total_image_recapture_count: int = 38,
    approved_count: int = 900,
    false_positive_image_count: int = 0,
):
    return build_gate_summary(
        raw_detector_metrics={"false_positive_count": 0, "false_negative_count": 3},
        accepted_detector_metrics={
            "false_positive_count": false_positive_image_count,
            "false_negative_count": 0,
            "false_positive_image_count": false_positive_image_count,
            "false_negative_image_count": 0,
            "exact_image_rate": 1.0,
        },
        classifier_metrics={
            "sample_count": 1000,
            "approved_count": approved_count,
            "approved_misrecognition_count": 1,
            "unknown_top3_candidate_out_count": 1,
            "unknown_rate": 0.05,
            "segment_recapture_rate": 0.05,
        },
        image_quality_metrics={"recapture_recall": 1.0},
        total_image_count=total_image_count,
        annotated_image_count=369,
        total_ground_truth_object_count=total_ground_truth_object_count,
        total_image_recapture_count=total_image_recapture_count,
        false_annotated_image_recapture_count=7,
        minimum_segmentation_image_rate=0.90,
        minimum_approved_rate=0.90,
        target_end_to_end_approved_rate=0.99,
        maximum_segmentation_image_false_negative_rate=0.001,
        maximum_segmentation_image_false_positive_rate=0.001,
        maximum_approved_misrecognition_rate=0.001,
        maximum_unknown_top3_candidate_out_rate=0.001,
    )


def test_pipeline_gate_accepts_end_to_end_target_and_inclusive_point_rate_gates():
    summary = _summary(approved_count=990)

    assert not summary["raw_detector_aspiration_met"]
    assert summary["finite_development_pipeline_goals_met"]
    assert summary["rates"]["segmentation_image_rate"] == 0.905
    assert summary["rates"]["end_to_end_approved_object_rate"] == 0.99
    assert summary["official_gates"]["approved_object_misrecognition_rate"]


def test_pipeline_gate_rejects_low_segmentation_and_approved_rates():
    summary = _summary(
        total_image_count=400,
        total_image_recapture_count=41,
        approved_count=899,
    )

    assert not summary["finite_development_pipeline_goals_met"]
    assert not summary["official_gates"]["segmentation_image_rate"]
    assert not summary["official_gates"]["end_to_end_approved_object_rate"]


def test_recaptured_or_unmatched_gt_remains_in_end_to_end_approved_denominator():
    summary = _summary(
        total_ground_truth_object_count=1100,
        approved_count=990,
    )

    assert summary["rates"]["end_to_end_approved_object_rate"] == 0.9
    assert summary["official_gates"]["end_to_end_approved_object_rate"]
    assert summary["operational_gates_met"]
    assert not summary["final_end_to_end_approved_goal_met"]
    assert not summary["finite_development_pipeline_goals_met"]


def test_pipeline_gate_rejects_image_level_false_positive_rate() -> None:
    summary = _summary(
        total_image_count=1000,
        total_image_recapture_count=0,
        false_positive_image_count=2,
    )

    assert not summary["official_gates"]["segmentation_image_false_positive_rate"]


def test_unknown_and_segment_recapture_rates_are_diagnostic_only() -> None:
    summary = _summary()

    assert "unknown_rate" not in summary["official_gates"]
    assert "segment_recapture_rate" not in summary["official_gates"]
    assert summary["diagnostics"]["unknown_and_segment_recapture_rates_are_not_promotion_gates"]
