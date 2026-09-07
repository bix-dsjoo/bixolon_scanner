from copy import deepcopy

import pytest

from bixolon_scanner.experiments.bread.limited220_diagnostics import summarize_repetitions


def _report(latency):
    return {
        "counts": {"approved_count": 100},
        "dataset": {"image_count": 20},
        "artifacts": {"runtime": "same"},
        "environment": {"provider": "cpu"},
        "performance": {"full_path": {"p95_ms": latency, "sample_count": 20}},
    }


def test_repetitions_select_median_run_and_do_not_inflate_unique_images():
    summary = summarize_repetitions([_report(120), _report(80), _report(100)])
    assert summary["representative_index"] == 2
    assert summary["median_run_p95_ms"] == 100
    assert summary["full_path_measurement_count"] == 60
    assert summary["unique_original_images"] == 20
    assert not summary["independent_validation"]


@pytest.mark.parametrize("field", ["counts", "dataset", "artifacts", "environment"])
def test_changed_decisions_or_identity_cannot_be_averaged_away(field):
    first = _report(100)
    changed = deepcopy(first)
    changed[field] = {"changed": True}
    with pytest.raises(ValueError, match="repetitions changed"):
        summarize_repetitions([first, changed])


def test_no_full_path_samples_preserves_missing_latency():
    report = _report(None)
    report["performance"]["full_path"]["sample_count"] = 0
    result = summarize_repetitions([report, report])
    assert result["median_run_p95_ms"] is None
    assert result["full_path_measurement_count"] == 0
