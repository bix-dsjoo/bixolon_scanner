"""Diagnose unreachable inference routes without modifying packaged metadata."""

from __future__ import annotations

from ..contracts.runtime_package_v2 import RuntimePackageV2Metadata


def audit_routing(metadata: RuntimePackageV2Metadata) -> dict:
    policy = metadata.classifier_policy
    default = (
        policy.ridge_approval_minimum_pair_probability
        if policy.ridge_approval_metric == "top2_pair_probability"
        else policy.ridge_approval_minimum_margin
    )
    if default is None:
        # Prototype-only catalogs expose 1.0 for accepted candidates.
        default = 1.0
    thresholds = (
        [default]
        if policy.ridge_approval_thresholds is None
        else [default if value is None else value for value in policy.ridge_approval_thresholds]
    )
    minimum = min(thresholds)
    issues = []
    verifier = metadata.classifier_verification
    if verifier is not None and not verifier.verify_all_approved_candidates:
        if verifier.ambiguity_maximum_approval_score <= minimum:
            issues.append(
                {
                    "route": "approved_verifier",
                    "code": "EMPTY_APPROVAL_INTERVAL",
                    "minimum_approval_score": minimum,
                    "maximum_score_exclusive": verifier.ambiguity_maximum_approval_score,
                }
            )
    fallback = metadata.classifier_resolution_fallback
    if fallback is not None:
        for index, rule in enumerate(fallback.approval_disagreement_rules):
            if rule.maximum_approval_score < minimum:
                issues.append(
                    {
                        "route": f"approved_detail_rule_{index}",
                        "code": "EMPTY_APPROVAL_INTERVAL",
                        "minimum_approval_score": minimum,
                        "maximum_score_inclusive": rule.maximum_approval_score,
                    }
                )
            if (
                rule.require_detector_disagreement
                and metadata.detector_class_mode == "class_agnostic"
            ):
                issues.append(
                    {
                        "route": f"approved_detail_rule_{index}",
                        "code": "CLASS_AGNOSTIC_DETECTOR_CANNOT_DISAGREE_ON_SKU",
                    }
                )
    return {"schema_version": "1.0", "issues": issues, "reachable": not issues}
