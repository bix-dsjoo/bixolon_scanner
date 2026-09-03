from bixolon_scanner.evaluation.packaged_worker_http import (
    expected_provider_label,
    latency_summary,
    percentile,
)


def test_packaged_http_latency_summary_uses_full_path_samples_only() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile([], 0.95) is None
    assert percentile(values, 0.50) == 25.0
    assert latency_summary(values) == {
        "sample_count": 4,
        "mean_ms": 25.0,
        "p50_ms": 25.0,
        "p95_ms": 38.5,
        "p99_ms": 39.7,
    }


def test_packaged_http_provider_label_describes_split_execution() -> None:
    assert expected_provider_label("openvino", "same") == "openvino"
    assert expected_provider_label("openvino", "openvino_gpu") == "openvino+openvino_gpu"
