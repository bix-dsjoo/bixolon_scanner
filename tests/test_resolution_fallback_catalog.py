import json
from types import SimpleNamespace

import pytest

from bixolon_scanner.contracts.catalog import (
    CatalogLabel,
    CatalogMetadata,
    load_store_catalog_package,
    sha256_file,
)
from bixolon_scanner.contracts.errors import PackageValidationError
from bixolon_scanner.runtime.catalog import load_resolution_fallback_catalog


def write_catalog(root, embedder_id):
    root.mkdir(parents=True)
    (root / "source-manifest.jsonl").write_text("{}\n")
    metadata = CatalogMetadata(
        catalog_version="0.2.1",
        store_id="test",
        embedder_id=embedder_id,
        embedder_version="0.2.1",
        classifier_policy_version="0.2.1",
        embedding_dimension=2,
        support_count_per_class=10,
        support_count=10,
        labels=[
            CatalogLabel(
                class_id="bread",
                class_name="Bread",
                support_offset=0,
                support_count=10,
                compactness=0.9,
            )
        ],
        source_manifest_sha256=sha256_file(root / "source-manifest.jsonl"),
        authentication="CHECKSUM-SHA256",
    )
    (root / "catalog.json").write_text(metadata.model_dump_json())
    (root / "activation.json").write_text('{"state":"active"}')
    for name in ["supports.bin", "prototypes.bin", "statistics.json"]:
        (root / name).write_bytes(b"diagnostic payload")
    checksums = {p.name: sha256_file(p) for p in root.iterdir() if p.is_file()}
    (root / "checksums.json").write_text(json.dumps(checksums))
    return load_store_catalog_package(root)


def setup_pair(tmp_path):
    primary = write_catalog(tmp_path / "catalog", "student")
    detail = write_catalog(primary.root / "detail", "teacher-detail")
    policy = SimpleNamespace(
        catalog_directory="detail",
        catalog_checksums_sha256=sha256_file(detail.root / "checksums.json"),
        embedder=SimpleNamespace(
            embedder_id="teacher-detail", embedding_dimension=2, version="0.2.1"
        ),
    )
    runtime = SimpleNamespace(metadata=SimpleNamespace(classifier_resolution_fallback=policy))
    return runtime, primary, detail


def test_distinct_feature_space_loads_its_own_catalog(tmp_path):
    runtime, primary, detail = setup_pair(tmp_path)
    actual = load_resolution_fallback_catalog(runtime, primary)
    assert actual.root == detail.root
    assert actual.metadata.embedder_id != primary.metadata.embedder_id


@pytest.mark.parametrize(
    "corruption", ["checksum", "payload", "feature_space", "version", "escape"]
)
def test_invalid_detail_catalog_is_startup_error(tmp_path, corruption):
    runtime, primary, detail = setup_pair(tmp_path)
    policy = runtime.metadata.classifier_resolution_fallback
    if corruption == "checksum":
        policy.catalog_checksums_sha256 = "f" * 64
    elif corruption == "payload":
        (detail.root / "supports.bin").write_bytes(b"changed")
    elif corruption == "feature_space":
        policy.embedder.embedder_id = "wrong-space"
    elif corruption == "version":
        policy.embedder.version = "0.2.2"
    else:
        policy.catalog_directory = "../outside"
    with pytest.raises(PackageValidationError):
        load_resolution_fallback_catalog(runtime, primary)


def test_legacy_catalog_remains_shared_without_new_metadata(tmp_path):
    runtime, primary, _ = setup_pair(tmp_path)
    runtime.metadata.classifier_resolution_fallback.catalog_directory = None
    assert load_resolution_fallback_catalog(runtime, primary) is primary
