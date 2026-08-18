import pytest

from bixolon_scanner.experiments.bread.detector_ensemble_latency import latency_summary


def test_latency_summary_reports_tail_percentiles() -> None:
    result = latency_summary([1.0, 2.0, 3.0, 4.0])

    assert result["sample_count"] == 4
    assert result["mean_ms"] == pytest.approx(2.5)
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["p99_ms"] == pytest.approx(3.97)
