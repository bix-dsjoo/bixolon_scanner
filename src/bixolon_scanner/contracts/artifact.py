from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .catalog import sha256_file


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def directory_content_manifest(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"artifact directory is missing: {resolved}")
    files = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file())
    ]
    if not files:
        raise ValueError(f"artifact directory is empty: {resolved}")
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }


def assert_release_not_revoked(release_lock_path: Path, lock_sha256: str) -> None:
    revocation_path = release_lock_path.resolve().with_name("release-revocation.json")
    if not revocation_path.is_file():
        return
    try:
        payload = json.loads(revocation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("release revocation record is invalid") from exc
    expected = payload.get("revocation_sha256")
    body = {key: value for key, value in payload.items() if key != "revocation_sha256"}
    if (
        payload.get("status") != "revoked"
        or payload.get("release_lock_sha256") != lock_sha256
        or expected != canonical_sha256(body)
    ):
        raise ValueError("release revocation record is invalid")
    raise ValueError("release candidate was revoked after a failed promotion gate")
