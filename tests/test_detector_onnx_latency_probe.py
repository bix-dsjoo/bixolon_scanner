from bixolon_scanner.experiments.archive.bread_1_0_0.detector_onnx_latency_probe import (
    latency_summary,
)


def test_latency_summary_reports_tail_percentiles():
    result = latency_summary([1.0, 2.0, 3.0, 4.0])

    assert result["sample_count"] == 4
    assert result["mean_ms"] == 2.5
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == 3.8499999999999996
    assert result["p99_ms"] == 3.9699999999999998
