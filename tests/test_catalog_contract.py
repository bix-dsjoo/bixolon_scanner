from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from bixolon_scanner.contracts.catalog import (
    CatalogActivation,
    CatalogLabel,
    CatalogMetadata,
    load_store_catalog_package,
    sha256_file,
)
from bixolon_scanner.contracts.errors import PackageValidationError
from bixolon_scanner.contracts.runtime_package_v2 import (
    CatalogDecisionPolicy,
    CatalogSupportAugmentationMetadata,
)
from bixolon_scanner.operations.catalog_activation import _adapter_features, fit_ridge_adapter
from bixolon_scanner.runtime.catalog import OnnxCatalogClassifier

SIGNING_KEY = b"test-only-catalog-signing-key"


def _write_catalog(root: Path) -> None:
    root.mkdir()
    source_manifest = root / "source-manifest.jsonl"
    source_manifest.write_text('{"image_id":"image-01"}\n', encoding="utf-8")
    metadata = CatalogMetadata(
        catalog_version="2.0.0-rc.2",
        store_id="test-store",
        embedder_id="dinov3-convnext-tiny-frozen",
        embedder_version="2.0.0-rc.2",
        classifier_policy_version="2.0.0-rc.2",
        embedding_dimension=768,
        support_count_per_class=10,
        support_count=10,
        labels=[
            CatalogLabel(
                class_id="bread_01",
                class_name="Bread",
                support_offset=0,
                support_count=10,
                compactness=0.9,
            )
        ],
        source_manifest_sha256=sha256_file(source_manifest),
    )
    (root / "catalog.json").write_text(
        json.dumps(metadata.model_dump(mode="json")), encoding="utf-8"
    )
    (root / "activation.json").write_text(
        CatalogActivation(state="active").model_dump_json(), encoding="utf-8"
    )
    (root / "supports.bin").write_bytes(b"supports")
    (root / "prototypes.bin").write_bytes(b"prototypes")
    (root / "statistics.json").write_text("{}", encoding="utf-8")
    signed_files = {
        name: sha256_file(root / name)
        for name in (
            "activation.json",
            "catalog.json",
            "prototypes.bin",
            "source-manifest.jsonl",
            "statistics.json",
            "supports.bin",
        )
    }
    checksums = json.dumps(signed_files, sort_keys=True, separators=(",", ":")).encode()
    (root / "checksums.json").write_bytes(checksums)
    signature = {
        "algorithm": "HMAC-SHA256",
        "key_id": "test-key",
        "signed_file": "checksums.json",
        "digest": hmac.new(SIGNING_KEY, checksums, hashlib.sha256).hexdigest(),
    }
    (root / "signature.json").write_text(json.dumps(signature), encoding="utf-8")


def test_catalog_metadata_rejects_ridge_head_without_adapter() -> None:
    with pytest.raises(ValidationError):
        CatalogMetadata(
            catalog_version="2.0.0",
            store_id="store",
            embedder_id="embedder",
            embedder_version="2.0.0",
            classifier_policy_version="2.0.0",
            embedding_dimension=4,
            support_count_per_class=10,
            support_count=10,
            labels=[
                CatalogLabel(
                    class_id="bread",
                    class_name="Bread",
                    support_offset=0,
                    support_count=10,
                    compactness=0.5,
                )
            ],
            source_manifest_sha256="0" * 64,
            decision_head="ridge_adapter",
        )


def test_catalog_metadata_rejects_adapter_path_escape() -> None:
    with pytest.raises(ValidationError):
        CatalogMetadata(
            catalog_version="2.0.0",
            store_id="store",
            embedder_id="embedder",
            embedder_version="2.0.0",
            classifier_policy_version="2.0.0",
            embedding_dimension=4,
            support_count_per_class=10,
            support_count=10,
            labels=[
                CatalogLabel(
                    class_id="bread",
                    class_name="Bread",
                    support_offset=0,
                    support_count=10,
                    compactness=0.5,
                )
            ],
            source_manifest_sha256="0" * 64,
            decision_head="ridge_adapter",
            adapter_filename="../outside.bin",
        )


def test_signed_catalog_loads_and_rejects_wrong_identity_or_tampering(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _write_catalog(root)

    loaded = load_store_catalog_package(
        root,
        signing_key=SIGNING_KEY,
        expected_store_id="test-store",
        expected_key_id="test-key",
    )
    assert loaded.metadata.catalog_version == "2.0.0-rc.2"

    with pytest.raises(PackageValidationError):
        load_store_catalog_package(
            root,
            signing_key=SIGNING_KEY,
            expected_store_id="another-store",
        )
    with pytest.raises(PackageValidationError):
        load_store_catalog_package(
            root,
            signing_key=SIGNING_KEY,
            expected_key_id="another-key",
        )

    (root / "supports.bin").write_bytes(b"tampered")
    with pytest.raises(PackageValidationError):
        load_store_catalog_package(root, signing_key=SIGNING_KEY)


def test_checksum_only_catalog_loads_without_a_key_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _write_catalog(root)
    metadata_path = root / "catalog.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["authentication"] = "CHECKSUM-SHA256"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    checksums_path = root / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["catalog.json"] = sha256_file(metadata_path)
    checksums_path.write_text(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    (root / "signature.json").unlink()

    loaded = load_store_catalog_package(
        root,
        expected_store_id="test-store",
    )
    assert loaded.metadata.authentication == "CHECKSUM-SHA256"

    (root / "supports.bin").write_bytes(b"tampered")
    with pytest.raises(PackageValidationError):
        load_store_catalog_package(root)


def test_catalog_ridge_adapter_is_deterministic_and_fits_support_labels() -> None:
    features = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    first = fit_ridge_adapter(features, labels, alpha=0.01, class_count=2)
    second = fit_ridge_adapter(features, labels, alpha=0.01, class_count=2)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    logits = features @ first[0] + first[1]
    assert np.array_equal(np.argmax(logits, axis=1), labels)


def test_catalog_support_augmentation_is_deterministic() -> None:
    class FakeEmbedder:
        @staticmethod
        def embed_images_raw(images: list[Image.Image]) -> np.ndarray:
            return np.asarray(
                [
                    [
                        np.asarray(image, dtype=np.float32).mean(),
                        np.asarray(image, dtype=np.float32).std(),
                    ]
                    for image in images
                ],
                dtype=np.float32,
            )

    policy = CatalogSupportAugmentationMetadata(
        views_per_source=2,
        seed=7,
        output_size=128,
        compiler_batch_size=2,
    )
    runtime = SimpleNamespace(
        metadata=SimpleNamespace(classifier_policy=SimpleNamespace(support_augmentation=policy))
    )
    records = [
        {"image_sha256": "1" * 64, "category_id": 1},
        {"image_sha256": "2" * 64, "category_id": 2},
    ]

    def images() -> list[Image.Image]:
        output = []
        for color in ((180, 70, 30), (30, 120, 190)):
            image = Image.new("RGB", (128, 128), "white")
            ImageDraw.Draw(image).ellipse((24, 18, 104, 112), fill=color)
            output.append(image)
        return output

    first_images = images()
    second_images = images()
    try:
        first = _adapter_features(FakeEmbedder(), first_images, records, runtime)
        second = _adapter_features(FakeEmbedder(), second_images, records, runtime)
    finally:
        for image in first_images + second_images:
            image.close()

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], np.asarray([0, 0, 0, 1, 1, 1]))
    assert first[1].tolist() == second[1].tolist()
    assert first[2] == second[2]
    assert first[2]["feature_count"] == 6


def test_ridge_pair_probability_blocks_ambiguous_head_disagreement_and_low_retrieval() -> None:
    classifier = object.__new__(OnnxCatalogClassifier)
    classifier.policy = CatalogDecisionPolicy(
        version="2.0.0-rc.8",
        prototype_weight=0.5,
        support_top_k=3,
        approval_minimum_similarity=1.0,
        approval_minimum_margin=0.1,
        ood_maximum_similarity=0.4,
        top3_minimum_similarity=-1.0,
        catalog_conflict_similarity=0.95,
        ridge_approval_metric="top2_pair_probability",
        ridge_approval_minimum_pair_probability=0.55,
        ridge_disagreement_minimum_pair_probability=0.65,
        ridge_top3_minimum_inverse_entropy=-2.9,
    )
    classifier.labels = [
        SimpleNamespace(class_id="bread_01"),
        SimpleNamespace(class_id="bread_02"),
        SimpleNamespace(class_id="bread_03"),
    ]
    classifier.restricted_ids = set()
    classifier.restricted_pairs = set()
    classifier.adapter_weight = np.eye(3, dtype=np.float32)
    classifier.adapter_bias = np.zeros(3, dtype=np.float32)

    result = classifier._classify_adapter(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.6, 0.5, 0.4],
                [0.4, 0.6, 0.3],
                [0.39, 0.38, 0.37],
            ],
            dtype=np.float32,
        ),
    )

    assert result.approval_scores[0] > 0.55
    assert not result.approval_blocked[0]
    assert result.segment_recapture_reasons[0] is None
    assert result.approval_scores[1] > 0.55
    assert result.approval_scores[1] < 0.65
    assert result.approval_blocked[1]
    assert result.unknown_reasons[1] == "CLASSIFIER_AMBIGUOUS_TOP2"
    assert result.segment_recapture_reasons[2] == "CLASSIFIER_OUT_OF_CATALOG"
    assert np.allclose(result.retrieval_logits[0], [0.6, 0.5, 0.4])


def test_ridge_margin_policy_requires_retrieval_agreement_and_minimum_similarity() -> None:
    classifier = object.__new__(OnnxCatalogClassifier)
    classifier.policy = CatalogDecisionPolicy(
        version="2.0.1-rc.3",
        prototype_weight=0.5,
        support_top_k=3,
        approval_minimum_similarity=1.0,
        approval_minimum_margin=0.1,
        ood_maximum_similarity=-1.0,
        top3_minimum_similarity=-1.0,
        catalog_conflict_similarity=0.95,
        ridge_approval_minimum_margin=0.2,
        ridge_top3_minimum_inverse_entropy=-3.0,
        ridge_require_retrieval_agreement=True,
        ridge_retrieval_minimum_similarity=0.4,
    )
    classifier.labels = [
        SimpleNamespace(class_id="bread_01"),
        SimpleNamespace(class_id="bread_02"),
        SimpleNamespace(class_id="bread_03"),
    ]
    classifier.restricted_ids = set()
    classifier.restricted_pairs = set()
    classifier.adapter_weight = np.eye(3, dtype=np.float32)
    classifier.adapter_bias = np.zeros(3, dtype=np.float32)

    result = classifier._classify_adapter(
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.7, 0.8, 0.1], [0.3, 0.2, 0.1]], dtype=np.float32),
    )

    assert result.approval_blocked.tolist() == [True, True]
    assert result.unknown_reasons == (
        "CLASSIFIER_AMBIGUOUS_TOP2",
        "BELOW_APPROVAL_THRESHOLD",
    )
    assert result.segment_recapture_reasons == (None, "CLASSIFIER_OUT_OF_CATALOG")
