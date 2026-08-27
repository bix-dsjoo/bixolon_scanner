from __future__ import annotations

from bixolon_scanner.contracts.runtime_package_v2 import DetectorCrowdingPolicyMetadata
from bixolon_scanner.operations.scanner_015_runtime import corroborated_large_proposal_policy


def test_large_proposal_policy_requires_independent_crowding_evidence() -> None:
    policy = corroborated_large_proposal_policy()

    assert isinstance(policy, DetectorCrowdingPolicyMetadata)
    assert policy.large_proposal_minimum_area_ratio == 0.21
    assert policy.large_proposal_corroboration is not None
    assert policy.large_proposal_corroboration.query_containment_surplus_minimum == 1
    assert policy.large_proposal_corroboration.selected_center_minimum == 2
    assert policy.large_proposal_corroboration.selected_count_maximum == 5
