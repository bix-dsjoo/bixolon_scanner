from bixolon_scanner.experiments.bread.dinov3_detector_oof_report import (
    count_gate_catches_proposal_miss,
)


def test_count_gate_catches_miss_when_exact_count_exceeds_filtered_proposals() -> None:
    assert count_gate_catches_proposal_miss(
        expected_count=5,
        missed_count=1,
        count_prediction=5,
        count_confidence=0.99,
        minimum_confidence=0.9,
    )


def test_count_gate_recaptures_low_confidence_even_if_wrong_count_agrees() -> None:
    assert count_gate_catches_proposal_miss(
        expected_count=5,
        missed_count=1,
        count_prediction=4,
        count_confidence=0.5,
        minimum_confidence=0.9,
    )


def test_count_gate_exposes_dangerous_high_confidence_agreement() -> None:
    assert not count_gate_catches_proposal_miss(
        expected_count=5,
        missed_count=1,
        count_prediction=4,
        count_confidence=0.99,
        minimum_confidence=0.9,
    )
