from __future__ import annotations

from pathlib import Path

from .errors import PackageValidationError


def validate_package_filename(value: str | None) -> str | None:
    """Validate a package-owned, relative file name without touching the filesystem."""

    if value is None:
        return None
    path = Path(value)
    if not value.strip() or path.is_absolute() or path.anchor or ".." in path.parts:
        raise ValueError("package files must use confined relative paths")
    return value


def resolve_package_file(root: Path, filename: str) -> Path:
    """Resolve an existing package file and reject paths escaping ``root``."""

    package_root = root.resolve()
    candidate = (package_root / filename).resolve()
    try:
        candidate.relative_to(package_root)
    except ValueError as exc:
        raise PackageValidationError from exc
    if not candidate.is_file():
        raise PackageValidationError
    return candidate


__all__ = ["resolve_package_file", "validate_package_filename"]
