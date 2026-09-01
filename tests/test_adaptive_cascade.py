from __future__ import annotations

from bixolon_scanner.command_registry import DIAGNOSTIC_COMMANDS
from bixolon_scanner.experiments.bread.adaptive_cascade import (
    AdaptiveCascadePolicy,
    AdaptiveCascadeRecord,
    CandidateEvidence,
    CandidateOutcome,
    GroundTruthObject,
    RankedClass,
    SceneEvidence,
    Stage2Evidence,
    Stage2Verdict,
    Stage3Evidence,
    decide_image,
    evaluate_decisions,
)


def _policy(**overrides) -> AdaptiveCascadePolicy:
    values = {
        "proposal_minimum_score": 0.05,
        "early_detector_score_minimum": 0.8,
        "early_localization_score_minimum": 0.8,
        "early_quality_score_minimum": 0.8,
        "early_catalog_inlier_minimum": 0.8,
        "early_top1_score_minimum": 0.8,
        "early_top1_margin_minimum": 0.2,
        "early_support_agreement_minimum": 0.8,
        "stage2_confidence_minimum": 0.8,
        "stage3_catalog_inlier_minimum": 0.8,
        "stage3_approval_score_minimum": 0.8,
        "stage3_approval_margin_minimum": 0.2,
        "stage3_support_agreement_minimum": 0.8,
        "count_confidence_minimum": 0.8,
        "unexplained_foreground_maximum": 0.1,
        "require_exact_count": True,
        "require_foreground_evidence": True,
        "maximum_prediction_set_size": 3,
    }
    values.update(overrides)
    return AdaptiveCascadePolicy.model_validate(values)


def _ranking(top1: str = "bread_01") -> list[RankedClass]:
    remainder = [class_id for class_id in ("bread_01", "bread_02", "bread_03") if class_id != top1]
    return [
        RankedClass(class_id=top1, score=0.9),
        RankedClass(class_id=remainder[0], score=0.08),
        RankedClass(class_id=remainder[1], score=0.02),
    ]


def _candidate(candidate_id: str = "p1", **overrides) -> CandidateEvidence:
    values = {
        "candidate_id": candidate_id,
        "detector_score": 0.95,
        "localization_score": 0.95,
        "quality_score": 0.95,
        "catalog_inlier_score": 0.95,
        "support_agreement": 0.95,
        "top3": _ranking(),
        "matched_ground_truth_id": "gt1",
    }
    values.update(overrides)
    return CandidateEvidence.model_validate(values)


def _record(*candidates: CandidateEvidence, count: int = 1) -> AdaptiveCascadeRecord:
    return AdaptiveCascadeRecord(
        image_id="image-1",
        perceptual_group_id="capture-1",
        scene=SceneEvidence(
            predicted_count=count,
            count_confidence=0.95,
            unexplained_foreground_score=0.0,
        ),
        candidates=list(candidates),
        ground_truth=[GroundTruthObject(object_id="gt1", class_id="bread_01")],
    )


def test_easy_candidate_locks_without_running_later_stages() -> None:
    policy = _policy()
    record = _record(_candidate())

    decision = decide_image(record, policy)
    report = evaluate_decisions([record], [decision], policy)

    assert decision.status == "SEGMENTATION"
    assert decision.early_lock_count == 1
    assert decision.stage2_call_count == 0
    assert decision.stage3_call_count == 0
    assert decision.candidates[0].outcome is CandidateOutcome.APPROVED
    assert report["zero_count_safety"]["all_met"] is True
    assert report["metrics"]["early_lock_rate"] == 1.0


def test_identity_ambiguity_routes_only_to_stage3_and_returns_safe_top3() -> None:
    candidate = _candidate(
        quality_score=0.4,
        stage3=Stage3Evidence(
            catalog_inlier_score=0.95,
            support_agreement=0.7,
            verifier_agreement=False,
            prediction_set_size=3,
            top3=_ranking(),
        ),
    )
    record = _record(candidate)

    decision = decide_image(record, _policy())

    assert decision.status == "SEGMENTATION"
    assert decision.stage2_call_count == 0
    assert decision.stage3_call_count == 1
    assert decision.candidates[0].outcome is CandidateOutcome.UNKNOWN
    assert decision.candidates[0].top3[0] == "bread_01"


def test_box_only_ambiguity_uses_stage2_then_reuses_fast_identity() -> None:
    candidate = _candidate(
        detector_score=0.4,
        stage2=Stage2Evidence(verdict=Stage2Verdict.REFINE, confidence=0.95),
    )

    decision = decide_image(_record(candidate), _policy())

    assert decision.status == "SEGMENTATION"
    assert decision.stage2_call_count == 1
    assert decision.stage3_call_count == 0
    assert decision.candidates[0].outcome is CandidateOutcome.APPROVED
    assert decision.candidates[0].reason == "STAGE2_BOX_VERIFIED_FAST_IDENTITY_SAFE"


def test_exact_count_mismatch_hides_early_locks_from_public_result() -> None:
    record = _record(_candidate(), count=2)

    decision = decide_image(record, _policy())

    assert decision.status == "IMAGE_RECAPTURE"
    assert decision.early_lock_count == 1
    assert decision.public_candidate_ids == []
    assert "EXACT_COUNT_MISMATCH" in decision.diagnostic_reasons


def test_stage1_miss_is_reported_even_when_scene_recapture_blocks_public_fn() -> None:
    policy = _policy()
    base = _record(_candidate(), count=2)
    record = base.model_copy(
        update={
            "ground_truth": [
                GroundTruthObject(object_id="gt1", class_id="bread_01"),
                GroundTruthObject(object_id="gt2", class_id="bread_02"),
            ]
        }
    )

    decision = decide_image(record, policy)
    report = evaluate_decisions([record], [decision], policy)

    assert decision.status == "IMAGE_RECAPTURE"
    assert report["metrics"]["stage1_proposal_miss_count"] == 1
    assert report["metrics"]["accepted_image_false_negative_count"] == 0


def test_stage2_can_remove_confirmed_background_without_creating_public_fp() -> None:
    false_candidate = _candidate(
        "p2",
        detector_score=0.4,
        matched_ground_truth_id=None,
        stage2=Stage2Evidence(verdict=Stage2Verdict.BACKGROUND, confidence=0.99),
    )
    policy = _policy()
    record = _record(_candidate(), false_candidate)

    decision = decide_image(record, policy)
    report = evaluate_decisions([record], [decision], policy)

    assert decision.status == "SEGMENTATION"
    assert decision.public_candidate_ids == ["p1"]
    assert decision.candidates[1].outcome is CandidateOutcome.BACKGROUND
    assert report["metrics"]["accepted_object_false_positive_count"] == 0


def test_large_stage3_prediction_set_requires_segment_recapture() -> None:
    candidate = _candidate(
        catalog_inlier_score=0.4,
        stage3=Stage3Evidence(
            catalog_inlier_score=0.95,
            support_agreement=0.95,
            verifier_agreement=True,
            prediction_set_size=4,
            top3=_ranking(),
        ),
    )

    decision = decide_image(_record(candidate), _policy())

    assert decision.status == "SEGMENTATION"
    assert decision.candidates[0].outcome is CandidateOutcome.SEGMENT_RECAPTURE
    assert decision.candidates[0].reason == "STAGE3_PREDICTION_SET_TOO_LARGE"


def test_hard_quality_failure_stops_before_candidate_work() -> None:
    base = _record(_candidate())
    record = base.model_copy(
        update={
            "scene": SceneEvidence(
                hard_quality_failure=True,
                predicted_count=1,
                count_confidence=0.95,
                unexplained_foreground_score=0.0,
            )
        }
    )

    decision = decide_image(record, _policy())

    assert decision.status == "IMAGE_RECAPTURE"
    assert decision.candidates == []
    assert decision.stage2_call_count == 0
    assert decision.stage3_call_count == 0


def test_adaptive_cascade_is_exposed_only_as_a_diagnostic_command() -> None:
    assert DIAGNOSTIC_COMMANDS[("experiment", "bread-adaptive-cascade")] == (
        "bixolon_scanner.experiments.bread.adaptive_cascade",
        "main",
    )
