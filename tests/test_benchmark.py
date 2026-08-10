from bixolon_scanner.benchmark import _latency_summary


def test_latency_summary_reports_tail_percentiles():
    summary = _latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary["sample_count"] == 4
    assert summary["p50_ms"] == 25.0
    assert summary["p95_ms"] > summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]
