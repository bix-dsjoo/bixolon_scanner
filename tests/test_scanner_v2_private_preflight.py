from __future__ import annotations

import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.scanner_v2_private_preflight import (
    MINIMUM_ZERO_ERROR_TRIALS,
    PrivateImageRecord,
    PrivateTestPlan,
    validate_development_identity_manifests,
    validate_locked_gate_tool,
    validate_private_trials,
)


def _eligible_record(*, membership: str = "in_catalog", status: str = "APPROVED") -> dict:
    return {
        "image_id": "eligible-1",
        "image_path": "private/eligible.jpg",
        "image_sha256": "a" * 64,
        "perceptual_hash": "0123456789abcdef",
        "store_id": "store-a",
        "capture_session_id": "session-a",
        "expected_image_status": "SEGMENTATION",
        "annotations": [
            {
                "annotation_id": "object-1",
                "bbox_xywh": [1, 2, 30, 40],
                "target_class_id": "bread_01",
                "physical_object_id": "physical-1",
                "catalog_membership": membership,
                "expected_item_status": status,
            }
        ],
    }


def _recapture_record() -> dict:
    return {
        "image_id": "recapture-1",
        "image_path": "private/recapture.jpg",
        "image_sha256": "b" * 64,
        "perceptual_hash": "fedcba9876543210",
        "store_id": "store-a",
        "capture_session_id": "session-b",
        "expected_image_status": "IMAGE_RECAPTURE",
        "annotations": [],
    }


def _plan(trials: list[dict]) -> PrivateTestPlan:
    return PrivateTestPlan.model_validate(
        {
            "dataset_id": "private-v1",
            "immutable_revision": "revision-1",
            "review_status": "locked",
            "release_lock_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "image_count": 2,
            "store_count": 1,
            "trials": trials,
        }
    )


def test_private_preflight_trial_contract_maps_every_endpoint() -> None:
    records = [
        PrivateImageRecord.model_validate(_eligible_record()),
        PrivateImageRecord.model_validate(_recapture_record()),
    ]
    plan = _plan(
        [
            {
                "endpoint": endpoint,
                "group_id": f"group-{endpoint}",
                "image_id": (
                    "recapture-1" if endpoint == "image_recapture_recall" else "eligible-1"
                ),
                "annotation_id": (
                    "object-1" if endpoint in {"approval_safety", "top3_safety"} else None
                ),
            }
            for endpoint in (
                "approval_safety",
                "detector_fn",
                "detector_fp",
                "top3_safety",
                "image_recapture_recall",
                "unnecessary_image_recapture",
            )
        ]
    )

    assert validate_private_trials(plan, records) == {
        endpoint: 1
        for endpoint in (
            "approval_safety",
            "detector_fn",
            "detector_fp",
            "top3_safety",
            "image_recapture_recall",
            "unnecessary_image_recapture",
        )
    }


def test_private_plan_rejects_reused_endpoint_group() -> None:
    trial = {
        "endpoint": "approval_safety",
        "group_id": "same-group",
        "image_id": "eligible-1",
        "annotation_id": "object-1",
    }

    with pytest.raises(ValidationError):
        _plan([trial, trial])


def test_private_manifest_rejects_path_escape_and_unsafe_ood_target() -> None:
    escaped = _eligible_record()
    escaped["image_path"] = "../outside.jpg"
    with pytest.raises(ValidationError):
        PrivateImageRecord.model_validate(escaped)

    unsafe_ood = _eligible_record(membership="ood", status="APPROVED")
    with pytest.raises(ValidationError):
        PrivateImageRecord.model_validate(unsafe_ood)


def test_private_gate_locks_statistical_minimums() -> None:
    assert MINIMUM_ZERO_ERROR_TRIALS["approval_safety"] == 2_995
    assert MINIMUM_ZERO_ERROR_TRIALS["image_recapture_recall"] == 299


def test_private_preflight_requires_exact_locked_development_lineage(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text('{"different":true}\n', encoding="utf-8")
    lock = {
        "private_test_policy": {
            "required_development_identity_manifests": [
                {"sha256": sha256_file(first)},
                {"sha256": sha256_file(second)},
            ]
        }
    }

    validate_development_identity_manifests(lock, [first, second])
    with pytest.raises(ValueError, match="exact locked"):
        validate_development_identity_manifests(lock, [first])


def test_private_gate_tool_must_match_release_lock(tmp_path) -> None:
    tool = tmp_path / "gate.py"
    tool.write_text("locked\n", encoding="utf-8")
    lock = {
        "supply_chain": {
            "private_gate_tools": [{"path": "tools/gate.py", "sha256": sha256_file(tool)}]
        }
    }

    assert validate_locked_gate_tool(lock, tool) == sha256_file(tool)
    tool.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        validate_locked_gate_tool(lock, tool)


def test_private_trials_cannot_split_one_physical_object_into_fake_groups() -> None:
    first = PrivateImageRecord.model_validate(_eligible_record())
    second_payload = _eligible_record()
    second_payload.update(
        {
            "image_id": "eligible-2",
            "image_path": "private/eligible-2.jpg",
            "image_sha256": "e" * 64,
            "perceptual_hash": "1123456789abcdef",
            "capture_session_id": "session-c",
        }
    )
    second_payload["annotations"][0]["annotation_id"] = "object-2"
    second = PrivateImageRecord.model_validate(second_payload)
    plan = _plan(
        [
            {
                "endpoint": "top3_safety",
                "group_id": "invented-a",
                "image_id": "eligible-1",
                "annotation_id": "object-1",
            },
            {
                "endpoint": "top3_safety",
                "group_id": "invented-b",
                "image_id": "eligible-2",
                "annotation_id": "object-2",
            },
        ]
    )

    with pytest.raises(ValueError, match="one-to-one"):
        validate_private_trials(plan, [first, second])
