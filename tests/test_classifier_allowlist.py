from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bixolon_scanner.training.classifier_allowlist import (
    audit_classifier_allowlist,
    sha256_file,
    verify_equivalent_upstream_manifest,
)


def _write_dataset(root: Path, *, classes: int = 2, shots: int = 3) -> Path:
    rows = []
    for category_id in range(1, classes + 1):
        for shot in range(shots):
            relative = Path("single_objects") / f"bread_{category_id:02d}" / f"{shot}.jpg"
            image = root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{category_id}-{shot}".encode())
            rows.append(
                {
                    "record_type": "classification",
                    "split": "development",
                    "category_id": category_id,
                    "fold": shot,
                    "perceptual_group_id": f"group-{category_id}-{shot}",
                    "image_path": relative.as_posix(),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            )
    manifest = root / "classifier.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return manifest


def _audit(root: Path, manifest: Path):
    return audit_classifier_allowlist(
        root,
        manifest,
        expected_manifest_sha256=sha256_file(manifest),
        expected_class_count=2,
        expected_shots_per_class=3,
    )


def test_allowlist_recomputes_every_source_checksum(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path)

    result = _audit(tmp_path, manifest)

    assert result.audit["record_count"] == 6
    assert result.audit["actual_access_matches_allowlist"] is True
    assert result.audit["source_image_set_sha256"] == result.audit["actual_access_image_set_sha256"]
    assert len(result.audit["images"]) == 6


def test_allowlist_fails_before_training_when_image_bytes_change(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path)
    image = tmp_path / "single_objects" / "bread_01" / "0.jpg"
    image.write_bytes(b"changed")

    with pytest.raises(ValueError, match="image checksum mismatch"):
        _audit(tmp_path, manifest)


def test_allowlist_rejects_source_outside_single_objects(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["image_path"] = "operational_collections/forbidden.jpg"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="single_objects only"):
        _audit(tmp_path, manifest)


def test_upstream_checkpoint_manifest_must_have_exact_same_source_set(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path)
    allowlist = _audit(tmp_path, manifest)
    upstream = tmp_path / "upstream.jsonl"
    upstream.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")

    report = verify_equivalent_upstream_manifest(
        allowlist,
        upstream,
        recorded_manifest_sha256=sha256_file(upstream),
    )

    assert report["equivalent_to_frozen_allowlist"] is True
    rows = upstream.read_text(encoding="utf-8").splitlines()
    upstream.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source set differs"):
        verify_equivalent_upstream_manifest(
            allowlist,
            upstream,
            recorded_manifest_sha256=sha256_file(upstream),
        )
