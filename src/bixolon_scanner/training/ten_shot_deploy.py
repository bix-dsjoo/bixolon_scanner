from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from ..package import load_model_package


def atomic_switch_package(
    *,
    pointer_path: Path,
    candidate_package: Path,
    fallback_package: Path,
    smoke_test: Callable[[Path], None],
) -> dict:
    """Validate both packages, smoke candidate, then atomically replace a pointer file."""
    candidate = candidate_package.resolve()
    fallback = fallback_package.resolve()
    load_model_package(candidate)
    load_model_package(fallback)
    smoke_test(candidate)
    payload = {
        "schema_version": "1.0",
        "active_package": str(candidate),
        "fallback_package": str(fallback),
        "switched_at": datetime.now(UTC).isoformat(),
    }
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer_path.with_name(f".{pointer_path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, pointer_path)
    return payload


def rollback_package(*, pointer_path: Path, smoke_test: Callable[[Path], None]) -> dict:
    current = json.loads(pointer_path.read_text(encoding="utf-8"))
    fallback = Path(current["fallback_package"]).resolve()
    load_model_package(fallback)
    smoke_test(fallback)
    payload = {
        **current,
        "active_package": str(fallback),
        "fallback_package": str(Path(current["active_package"]).resolve()),
        "rolled_back_at": datetime.now(UTC).isoformat(),
    }
    temporary = pointer_path.with_name(f".{pointer_path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, pointer_path)
    return payload
