from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import read_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    body = "".join(f"{value}\n" for value in sorted(values))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClassifierAllowlist:
    records: tuple[dict[str, Any], ...]
    audit: dict[str, Any]

    @property
    def image_sha256(self) -> frozenset[str]:
        return frozenset(str(row["image_sha256"]) for row in self.records)


def _relative_source_path(value: object, *, allowed_directory: str) -> Path:
    text = str(value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"classifier image path must be safe and relative: {text}")
    if path.parts[0] != allowed_directory:
        raise ValueError(f"classifier source must be {allowed_directory} only, got {path.parts[0]}")
    return path


def _validate_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    allowed_directory: str,
    expected_class_count: int,
    expected_shots_per_class: int,
    expected_folds: frozenset[int],
) -> list[dict[str, Any]]:
    expected_count = expected_class_count * expected_shots_per_class
    if len(rows) != expected_count:
        raise ValueError(
            f"classifier allowlist must contain exactly {expected_count} rows, got {len(rows)}"
        )
    required = {"category_id", "fold", "image_path", "image_sha256"}
    missing = [index for index, row in enumerate(rows) if not required <= set(row)]
    if missing:
        raise ValueError(f"classifier allowlist rows are missing required fields: {missing[:3]}")
    for row in rows:
        _relative_source_path(row["image_path"], allowed_directory=allowed_directory)
        digest = str(row["image_sha256"])
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("classifier allowlist requires lowercase SHA-256 image digests")
    paths = [str(row["image_path"]).replace("\\", "/") for row in rows]
    digests = [str(row["image_sha256"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("classifier allowlist contains duplicate image paths")
    if len(digests) != len(set(digests)):
        raise ValueError("classifier allowlist contains duplicate image SHA-256 digests")
    counts = Counter(int(row["category_id"]) for row in rows)
    expected_counts = Counter(
        {
            category_id: expected_shots_per_class
            for category_id in range(1, expected_class_count + 1)
        }
    )
    if counts != expected_counts:
        raise ValueError("classifier allowlist must be class-balanced with contiguous category IDs")
    folds = {int(row["fold"]) for row in rows}
    if folds != expected_folds:
        raise ValueError(
            f"classifier allowlist folds must be {sorted(expected_folds)}, got {sorted(folds)}"
        )
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group = str(row.get("perceptual_group_id", row["image_sha256"]))
        group_folds[group].add(int(row["fold"]))
    leaked = sorted(group for group, assigned in group_folds.items() if len(assigned) != 1)
    if leaked:
        raise ValueError(f"classifier perceptual groups cross folds: {leaked[:3]}")
    return sorted(
        rows,
        key=lambda row: (int(row["category_id"]), str(row["image_path"]).replace("\\", "/")),
    )


def audit_classifier_allowlist(
    dataset_root: Path,
    manifest: Path,
    *,
    expected_manifest_sha256: str | None = None,
    allowed_directory: str = "single_objects",
    expected_class_count: int = 20,
    expected_shots_per_class: int = 10,
    expected_folds: frozenset[int] = frozenset({0, 1, 2}),
) -> ClassifierAllowlist:
    """Verify the exact 1.1.1+ classifier source before any model work starts."""
    manifest = manifest.resolve()
    dataset_root = dataset_root.resolve()
    manifest_sha256 = sha256_file(manifest)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("classifier allowlist manifest checksum mismatch")
    rows = _validate_manifest_rows(
        list(read_manifest(manifest)),
        allowed_directory=allowed_directory,
        expected_class_count=expected_class_count,
        expected_shots_per_class=expected_shots_per_class,
        expected_folds=expected_folds,
    )
    accessed: list[dict[str, Any]] = []
    for row in rows:
        relative = _relative_source_path(row["image_path"], allowed_directory=allowed_directory)
        image_path = (dataset_root / relative).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError("classifier image resolves outside the dataset root") from exc
        if not image_path.is_file():
            raise ValueError(f"classifier allowlist image is missing: {relative.as_posix()}")
        actual_sha256 = sha256_file(image_path)
        expected_sha256 = str(row["image_sha256"])
        if actual_sha256 != expected_sha256:
            raise ValueError(f"classifier image checksum mismatch: {relative.as_posix()}")
        accessed.append(
            {
                "image_path": relative.as_posix(),
                "image_sha256": actual_sha256,
                "category_id": int(row["category_id"]),
                "fold": int(row["fold"]),
            }
        )
    manifest_set_sha256 = sha256_lines(row["image_sha256"] for row in accessed)
    audit = {
        "schema_version": "1.0",
        "policy": "bread_classifier_200_only_1.1.1_plus",
        "allowed_directory": allowed_directory,
        "manifest_sha256": manifest_sha256,
        "record_count": len(rows),
        "class_count": expected_class_count,
        "shots_per_class": expected_shots_per_class,
        "folds": sorted(expected_folds),
        "source_image_set_sha256": manifest_set_sha256,
        "actual_access_image_set_sha256": sha256_lines(row["image_sha256"] for row in accessed),
        "actual_access_matches_allowlist": True,
        "images": accessed,
    }
    return ClassifierAllowlist(tuple(rows), audit)


def verify_equivalent_upstream_manifest(
    allowlist: ClassifierAllowlist,
    upstream_manifest: Path,
    *,
    recorded_manifest_sha256: str,
) -> dict[str, Any]:
    """Allow an older checkpoint only when its recorded source set is exactly the frozen 200."""
    actual_manifest_sha256 = sha256_file(upstream_manifest)
    if actual_manifest_sha256 != recorded_manifest_sha256:
        raise ValueError("upstream checkpoint manifest checksum mismatch")
    rows = list(read_manifest(upstream_manifest))
    upstream_sha256 = [
        str(row["image_sha256"])
        for row in rows
        if row.get("record_type") in {None, "classification"}
    ]
    if len(upstream_sha256) != len(set(upstream_sha256)):
        raise ValueError("upstream checkpoint manifest contains duplicate image SHA-256 digests")
    if set(upstream_sha256) != set(allowlist.image_sha256):
        raise ValueError(
            "upstream checkpoint source set differs from the frozen 200-image allowlist"
        )
    return {
        "manifest_sha256": actual_manifest_sha256,
        "record_count": len(upstream_sha256),
        "source_image_set_sha256": sha256_lines(upstream_sha256),
        "equivalent_to_frozen_allowlist": True,
    }


def write_allowlist_audit(path: Path, allowlist: ClassifierAllowlist) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(allowlist.audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
