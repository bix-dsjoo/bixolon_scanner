from __future__ import annotations

import pytest

from bixolon_scanner.experiments.bread.classifier_rotation_consistency import (
    _validate_trace_reproduction,
)


def test_cpu_trace_reproduction_rejects_an_unexpected_mismatch() -> None:
    with pytest.raises(ValueError, match="differs from trace for 3 objects"):
        _validate_trace_reproduction(
            3,
            provider="cpu",
            allow_alternative_runtime=False,
        )


def test_alternative_runtime_comparison_reports_expected_mismatch() -> None:
    _validate_trace_reproduction(
        3,
        provider="cpu",
        allow_alternative_runtime=True,
    )
