from bixolon_scanner.experiments.bread.detector_ambiguity_gate import (
    ambiguity_recapture_mask,
)


def _row(scores: list[float]) -> dict:
    return {"scores": scores}


def test_ambiguity_gate_supports_exact_and_at_least_extra_counts() -> None:
    available = [_row([0.9, 0.8, 0.7]), _row([0.9, 0.8, 0.7, 0.6])]
    selected = [_row([0.9, 0.8]), _row([0.9, 0.8])]

    exact = ambiguity_recapture_mask(
        available,
        selected,
        minimum_selected_count=2,
        extra_candidate_count=1,
        extra_count_mode="exact",
        next_score_threshold=0.65,
    )
    at_least = ambiguity_recapture_mask(
        available,
        selected,
        minimum_selected_count=2,
        extra_candidate_count=1,
        extra_count_mode="at_least",
        next_score_threshold=0.65,
    )

    assert exact.tolist() == [True, False]
    assert at_least.tolist() == [True, True]
