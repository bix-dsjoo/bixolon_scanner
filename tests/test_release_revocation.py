from __future__ import annotations

import json

import pytest

from bixolon_scanner.contracts.artifact import (
    assert_release_not_revoked,
    canonical_sha256,
)


def test_release_without_revocation_remains_available(tmp_path) -> None:
    release_lock = tmp_path / "release-lock.json"
    release_lock.write_text("{}", encoding="utf-8")

    assert_release_not_revoked(release_lock, "a" * 64)


def test_valid_revocation_blocks_release(tmp_path) -> None:
    release_lock = tmp_path / "release-lock.json"
    release_lock.write_text("{}", encoding="utf-8")
    body = {
        "schema_version": "2.0",
        "status": "revoked",
        "release_lock_sha256": "a" * 64,
        "reason": "failed_promotion_gate",
    }
    (tmp_path / "release-revocation.json").write_text(
        json.dumps({**body, "revocation_sha256": canonical_sha256(body)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revoked"):
        assert_release_not_revoked(release_lock, "a" * 64)


def test_invalid_revocation_fails_closed(tmp_path) -> None:
    release_lock = tmp_path / "release-lock.json"
    release_lock.write_text("{}", encoding="utf-8")
    (tmp_path / "release-revocation.json").write_text(
        json.dumps(
            {
                "status": "revoked",
                "release_lock_sha256": "a" * 64,
                "revocation_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid"):
        assert_release_not_revoked(release_lock, "a" * 64)
