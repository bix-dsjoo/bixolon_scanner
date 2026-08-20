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
