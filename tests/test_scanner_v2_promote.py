from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.operations.scanner_v2_owner_waiver_promote import (
    promote_catalog_checksum_only,
    validate_owner_waiver,
)
from bixolon_scanner.operations.scanner_v2_promote import (
    EXPECTED_POINT_GATES,
    EXPECTED_STATISTICAL_GATES,
    _promote_catalog,
    _promote_runtime,
)


def test_production_finalizer_requires_every_locked_private_gate() -> None:
    assert len(EXPECTED_POINT_GATES) == 11
    assert EXPECTED_STATISTICAL_GATES == {
        "approval_safety",
        "detector_fn",
        "detector_fp",
        "top3_safety",
        "ood_false_approval",
        "image_recapture_recall",
        "unnecessary_image_recapture",
        "invalid_roi_action",
    }


def test_production_finalizer_only_changes_allowed_runtime_metadata(tmp_path) -> None:
    source = tmp_path / "source-runtime"
    target = tmp_path / "target-runtime"
    source.mkdir()
    metadata = {
        "worker_version": "2.0.0-rc.7",
        "promotion_status": "independent_test_pending",
        "detector_policy_version": "2.0.0-rc.7",
        "detector": {"version": "2.0.0-rc.7"},
        "embedder": {"version": "2.0.0-rc.7"},
        "classifier_policy": {"version": "2.0.0-rc.7"},
        "sources": {"embedder": {"training_pipeline_version": "2.0.0-rc.7"}},
    }
    (source / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "weight.onnx").write_bytes(b"immutable-weight")

    _promote_runtime(source, target, "2.0.0-rc.7")

    promoted = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert promoted["worker_version"] == "2.0.0"
    assert promoted["promotion_status"] == "production"
    assert promoted["detector"]["version"] == "2.0.0"
    assert promoted["embedder"]["version"] == "2.0.0"
    assert promoted["classifier_policy"]["version"] == "2.0.0"
    assert promoted["sources"] == metadata["sources"]
    assert (target / "weight.onnx").read_bytes() == b"immutable-weight"
    assert json.loads((source / "metadata.json").read_text()) == metadata


def test_production_finalizer_resigns_only_catalog_metadata(tmp_path) -> None:
    source = tmp_path / "source-catalog"
    target = tmp_path / "target-catalog"
    source.mkdir()
    metadata = {
        "catalog_version": "2.0.0-rc.7",
        "embedder_version": "2.0.0-rc.7",
        "classifier_policy_version": "2.0.0-rc.7",
    }
    (source / "catalog.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "supports.bin").write_bytes(b"immutable-supports")
    checksums = {
        "catalog.json": sha256_file(source / "catalog.json"),
        "supports.bin": sha256_file(source / "supports.bin"),
    }
    (source / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    (source / "signature.json").write_text("{}", encoding="utf-8")
    key = b"production-key-that-is-at-least-32-bytes"

    _promote_catalog(
        source,
        target,
        release_candidate="2.0.0-rc.7",
        production_key_id="production-key-1",
        production_signing_key=key,
    )

    promoted = json.loads((target / "catalog.json").read_text())
    promoted_checksums = json.loads((target / "checksums.json").read_text())
    signature = json.loads((target / "signature.json").read_text())
    assert set(promoted.values()) == {"2.0.0"}
    assert promoted_checksums["supports.bin"] == checksums["supports.bin"]
    assert promoted_checksums["catalog.json"] == sha256_file(target / "catalog.json")
    assert signature["key_id"] == "production-key-1"
    assert (
        signature["digest"]
        == hmac.new(key, (target / "checksums.json").read_bytes(), hashlib.sha256).hexdigest()
    )


def test_owner_waiver_catalog_promotion_keeps_payload_and_uses_checksums_only(
    tmp_path,
) -> None:
    source = tmp_path / "source-catalog"
    target = tmp_path / "target-catalog"
    source.mkdir()
    metadata = {
        "catalog_version": "2.0.0-rc.8",
        "embedder_version": "2.0.0-rc.8",
        "classifier_policy_version": "2.0.0-rc.8",
    }
    (source / "catalog.json").write_text(json.dumps(metadata), encoding="utf-8")
    (source / "supports.bin").write_bytes(b"immutable-supports")
    checksums = {
        "catalog.json": sha256_file(source / "catalog.json"),
        "supports.bin": sha256_file(source / "supports.bin"),
    }
    (source / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    (source / "signature.json").write_text("{}", encoding="utf-8")

    promote_catalog_checksum_only(source, target, "2.0.0-rc.8")

    promoted = json.loads((target / "catalog.json").read_text())
    promoted_checksums = json.loads((target / "checksums.json").read_text())
    assert promoted["authentication"] == "CHECKSUM-SHA256"
    assert promoted["catalog_version"] == "2.0.0"
    assert promoted["embedder_version"] == "2.0.0"
    assert promoted["classifier_policy_version"] == "2.0.0"
    assert promoted_checksums["supports.bin"] == checksums["supports.bin"]
    assert promoted_checksums["catalog.json"] == sha256_file(target / "catalog.json")
    assert not (target / "signature.json").exists()


def test_owner_waiver_must_name_both_skipped_gates() -> None:
    waiver = {
        "schema_version": "2.0",
        "release": "2.0.0",
        "source_release_candidate": "2.0.0-rc.8",
        "source_release_lock_sha256": "a" * 64,
        "decision": "owner_approved_manual_waiver",
        "approved": True,
        "production_eligible_by_owner_waiver": True,
        "independent_certified": False,
        "catalog_authentication": "CHECKSUM-SHA256",
        "failures": [
            "owner_private_locked_production_test",
            "catalog_hmac_authentication",
        ],
        "waivers": [
            {"gate": "owner_private_locked_production_test"},
            {"gate": "catalog_hmac_authentication"},
        ],
    }

    validate_owner_waiver(
        waiver,
        release_lock_sha256="a" * 64,
        release_candidate="2.0.0-rc.8",
    )

    waiver["waivers"].pop()
    with pytest.raises(ValueError):
        validate_owner_waiver(
            waiver,
            release_lock_sha256="a" * 64,
            release_candidate="2.0.0-rc.8",
        )
