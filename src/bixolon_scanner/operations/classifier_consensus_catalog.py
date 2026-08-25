from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..contracts.catalog import (
    CatalogMetadata,
    load_store_catalog_package,
    sha256_file,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalize_store_id(catalog_dir: Path, store_id: str) -> None:
    catalog_path = catalog_dir / "catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["store_id"] = store_id
    metadata = CatalogMetadata.model_validate(payload)
    catalog_path.write_bytes(_canonical_json(metadata.model_dump(mode="json", exclude_none=True)))
    checksums_path = catalog_dir / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["catalog.json"] = sha256_file(catalog_path)
    checksums_path.write_bytes(_canonical_json(checksums))


def build_consensus_catalog(
    primary_dir: Path,
    rotation_dir: Path,
    independent_dir: Path,
    output_dir: Path,
) -> dict:
    """Bind three source-identical Catalog views into one checksummed package."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    primary = load_store_catalog_package(primary_dir)
    rotation = load_store_catalog_package(rotation_dir)
    independent = load_store_catalog_package(independent_dir)
    catalogs = (primary, rotation, independent)
    if any(catalog.metadata.authentication != "CHECKSUM-SHA256" for catalog in catalogs):
        raise ValueError("classifier consensus requires checksum-authenticated Catalogs")
    expected = (
        primary.metadata.catalog_version,
        primary.metadata.classifier_policy_version,
        primary.metadata.source_manifest_sha256,
        tuple(label.class_id for label in primary.metadata.labels),
        primary.metadata.append_only_base_class_count,
    )
    for catalog in catalogs[1:]:
        observed = (
            catalog.metadata.catalog_version,
            catalog.metadata.classifier_policy_version,
            catalog.metadata.source_manifest_sha256,
            tuple(label.class_id for label in catalog.metadata.labels),
            catalog.metadata.append_only_base_class_count,
        )
        if observed != expected:
            raise ValueError("classifier consensus Catalog contracts differ")

    shutil.copytree(primary.root, output_dir)
    rotation_name = "rotation-verifier"
    independent_name = "independent-verifier"
    shutil.copytree(rotation.root, output_dir / rotation_name)
    shutil.copytree(independent.root, output_dir / independent_name)
    _normalize_store_id(output_dir / rotation_name, primary.metadata.store_id)
    _normalize_store_id(output_dir / independent_name, primary.metadata.store_id)
    payload = primary.metadata.model_dump(mode="json")
    payload["verification"] = {
        "rotation_catalog_directory": rotation_name,
        "rotation_checksums_sha256": sha256_file(output_dir / rotation_name / "checksums.json"),
        "independent_catalog_directory": independent_name,
        "independent_checksums_sha256": sha256_file(
            output_dir / independent_name / "checksums.json"
        ),
    }
    metadata = CatalogMetadata.model_validate(payload)
    (output_dir / "catalog.json").write_bytes(
        _canonical_json(metadata.model_dump(mode="json", exclude_none=True))
    )
    checksums_path = output_dir / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["catalog.json"] = sha256_file(output_dir / "catalog.json")
    checksums_path.write_bytes(_canonical_json(checksums))
    loaded = load_store_catalog_package(output_dir)
    return {
        "schema_version": "1.0",
        "operation": "build_selective_classifier_consensus_catalog",
        "catalog": output_dir.resolve().as_posix(),
        "catalog_version": loaded.metadata.catalog_version,
        "class_count": len(loaded.metadata.labels),
        "source_manifest_sha256": loaded.metadata.source_manifest_sha256,
        "rotation_checksums_sha256": loaded.metadata.verification.rotation_checksums_sha256,
        "independent_checksums_sha256": (loaded.metadata.verification.independent_checksums_sha256),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Scanner 0.1.3 consensus Catalog")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_consensus_catalog(
        args.primary,
        args.rotation,
        args.independent,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
