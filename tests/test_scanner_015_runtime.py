from __future__ import annotations

import pytest

from bixolon_scanner.contracts.runtime_package_v2 import DetectorCrowdingPolicyMetadata
from bixolon_scanner.operations.scanner_015_runtime import (
    corroborated_large_proposal_policy,
    object_presence_verifier_metadata,
)


def test_large_proposal_policy_requires_independent_crowding_evidence() -> None:
    policy = corroborated_large_proposal_policy()

    assert isinstance(policy, DetectorCrowdingPolicyMetadata)
    assert policy.large_proposal_minimum_area_ratio == 0.21
    assert policy.large_proposal_corroboration is not None
    assert policy.large_proposal_corroboration.query_containment_surplus_minimum == 1
    assert policy.large_proposal_corroboration.selected_center_minimum == 2
    assert policy.large_proposal_corroboration.selected_count_maximum == 5


def test_object_presence_verifier_uses_selected_uncertainty_threshold() -> None:
    metadata = object_presence_verifier_metadata(
        {
            "model_role": "generic_object_presence_count_verifier",
            "comparison_mode": "object_presence",
            "image_size": 192,
            "temperature": 1.0,
        },
        version="0.1.8",
        confidence_threshold=0.54,
    )

    assert metadata.filename == "count-verifier.onnx"
    assert metadata.version == "0.1.8"
    assert metadata.input_size == (192, 192)
    assert metadata.count_labels == [0, 1]
    assert metadata.comparison_mode == "object_presence"
    assert metadata.confidence_threshold == 0.54


def test_object_presence_verifier_rejects_non_presence_report() -> None:
    with pytest.raises(ValueError, match="model role"):
        object_presence_verifier_metadata(
            {
                "model_role": "exact_count_verifier",
                "comparison_mode": "exact_count",
                "image_size": 192,
            },
            version="0.1.8",
            confidence_threshold=0.54,
        )
