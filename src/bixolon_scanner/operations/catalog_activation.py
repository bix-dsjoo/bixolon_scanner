from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts.catalog import (
    CatalogActivation,
    CatalogLabel,
    CatalogMetadata,
    CatalogRestrictedPair,
    CatalogSignature,
    sha256_file,
)
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..runtime.catalog import OnnxEmbedder, l2_normalize
from ..training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    direct_roi_recipe_sha256,
    prepare_direct_roi_source,
)

MINIMUM_SUPPORT_IMAGE_SIDE = 96


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _audit_records(dataset_root: Path, records: list[dict]) -> list[dict]:
    if not records:
        raise ValueError("Catalog manifest is empty")
    counts = Counter(str(row["class_id"]) for row in records)
    if set(counts.values()) != {10}:
        raise ValueError("Catalog requires exactly ten valid images per SKU")
    hashes = [str(row["image_sha256"]) for row in records]
    perceptual = [str(row["perceptual_group_id"]) for row in records]
    if len(hashes) != len(set(hashes)) or len(perceptual) != len(set(perceptual)):
        raise ValueError("Catalog support images must not contain exact or near duplicates")
    audited = []
    for row in records:
        relative = Path(str(row["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Catalog image paths must be relative and confined")
        path = (dataset_root / relative).resolve()
        if dataset_root.resolve() not in path.parents or not path.is_file():
            raise ValueError("Catalog support image is outside the dataset root")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Catalog support image checksum mismatch")
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"} or min(image.size) < MINIMUM_SUPPORT_IMAGE_SIDE:
                raise ValueError("Catalog support image format or size is invalid")
        audited.append({**row, "resolved_path": path})
    return audited


def fit_ridge_adapter(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the Catalog-only linear head with a deterministic closed-form solve."""
    if alpha <= 0 or class_count <= 0:
        raise ValueError("Catalog ridge configuration is invalid")
    features64 = np.asarray(features, dtype=np.float64)
    labels64 = np.asarray(labels, dtype=np.int64)
    if features64.ndim != 2 or labels64.shape != (len(features64),):
        raise ValueError("Catalog ridge inputs have invalid shapes")
    if len(features64) == 0 or np.any(labels64 < 0) or np.any(labels64 >= class_count):
        raise ValueError("Catalog ridge labels are invalid")
    design = np.concatenate([features64, np.ones((len(features64), 1), dtype=np.float64)], axis=1)
    targets = np.eye(class_count, dtype=np.float64)[labels64]
    regularization = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularization[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ targets,
    )
    return coefficients[:-1].astype(np.float32), coefficients[-1].astype(np.float32)


def _adapter_features(
    embedder: OnnxEmbedder,
    images: list[Image.Image],
    records: list[dict],
    runtime,
) -> tuple[np.ndarray, np.ndarray, dict]:
    policy = runtime.metadata.classifier_policy.support_augmentation
    recipe = DirectRoiRecipe(
        output_size=policy.output_size,
        canvas_scale_min=policy.canvas_scale_min,
        canvas_scale_max=policy.canvas_scale_max,
        rotation_degrees=policy.rotation_degrees,
        perspective_fraction=policy.perspective_fraction,
        brightness_min=policy.brightness_min,
        brightness_max=policy.brightness_max,
        contrast_min=policy.contrast_min,
        contrast_max=policy.contrast_max,
        saturation_min=policy.saturation_min,
        saturation_max=policy.saturation_max,
        blur_probability=policy.blur_probability,
        blur_radius_max=policy.blur_radius_max,
        jpeg_quality_min=policy.jpeg_quality_min,
        jpeg_quality_max=policy.jpeg_quality_max,
        crop_mode=policy.crop_mode,
        procedural_gradient=policy.procedural_gradient,
        procedural_shadow=policy.procedural_shadow,
    )
    adapter_images = []
    source_indices = []
    for source_index, (image, row) in enumerate(zip(images, records, strict=True)):
        adapter_images.append(image)
        source_indices.append(source_index)
        if policy.views_per_source == 0:
            continue
        prepared = prepare_direct_roi_source(image, recipe)
        for view in range(1, policy.views_per_source + 1):
            sample = augment_direct_roi(
                image,
                source_sha256=str(row["image_sha256"]),
                category_id=int(row["category_id"]),
                seed=policy.seed + source_index * 10_000 + view,
                recipe=recipe,
                prepared_cutout=prepared,
            )
            adapter_images.append(sample.image)
            source_indices.append(source_index)
    parts = []
    try:
        for start in range(0, len(adapter_images), policy.compiler_batch_size):
            parts.append(
                embedder.embed_images_raw(
                    adapter_images[start : start + policy.compiler_batch_size]
                )
            )
    finally:
        for image in adapter_images:
            if all(image is not source for source in images):
                image.close()
    return (
        l2_normalize(np.concatenate(parts)),
        np.asarray(source_indices, dtype=np.int64),
        {
            "views_per_source": policy.views_per_source,
            "feature_count": len(adapter_images),
            "seed": policy.seed,
            "recipe_sha256": direct_roi_recipe_sha256(recipe),
        },
    )


def build_catalog(
    runtime_dir: Path,
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    store_id: str,
    catalog_version: str,
    signing_key: bytes | None,
    key_id: str | None,
    authentication: str = "CHECKSUM-SHA256",
    provider: str,
    cuda_dll_dir: Path | None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    runtime = load_runtime_package_v2(runtime_dir)
    records = sorted(
        _audit_records(dataset_root, _records(manifest_path)),
        key=lambda row: (str(row["class_id"]), str(row["image_sha256"])),
    )
    embedder = OnnxEmbedder(runtime, provider, cuda_dll_dir)
    images = []
    for row in records:
        with Image.open(row["resolved_path"]) as image:
            images.append(image.convert("RGB"))
    raw_supports = embedder.embed_images_raw(images)
    supports = embedder.transform.apply(raw_supports)
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_class[str(row["class_id"])].append(index)
    class_ids = sorted(by_class)
    prototypes = l2_normalize(
        np.stack([supports[by_class[class_id]].mean(axis=0) for class_id in class_ids])
    )
    class_indices = {class_id: index for index, class_id in enumerate(class_ids)}
    labels_array = np.asarray(
        [class_indices[str(row["class_id"])] for row in records], dtype=np.int64
    )
    adapter_features, adapter_source_indices, augmentation_statistics = _adapter_features(
        embedder, images, records, runtime
    )
    weight, bias = fit_ridge_adapter(
        adapter_features,
        labels_array[adapter_source_indices],
        alpha=runtime.metadata.classifier_policy.ridge_alpha,
        class_count=len(class_ids),
    )
    similarities = prototypes @ prototypes.T
    restricted_pairs = []
    restricted_ids: set[str] = set()
    for left in range(len(class_ids)):
        for right in range(left + 1, len(class_ids)):
            value = float(similarities[left, right])
            if value >= runtime.metadata.classifier_policy.catalog_conflict_similarity:
                pair = (class_ids[left], class_ids[right])
                restricted_pairs.append(
                    CatalogRestrictedPair(class_ids=pair, prototype_similarity=value)
                )
                restricted_ids.update(pair)
    labels = []
    for class_index, class_id in enumerate(class_ids):
        indices = by_class[class_id]
        other = similarities[class_index].copy()
        other[class_index] = -1.0
        nearest_index = int(np.argmax(other))
        labels.append(
            CatalogLabel(
                class_id=class_id,
                class_name=str(records[indices[0]]["class_name"]),
                support_offset=indices[0],
                support_count=len(indices),
                compactness=float((supports[indices] @ prototypes[class_index]).mean()),
                nearest_class_id=class_ids[nearest_index],
                nearest_similarity=float(other[nearest_index]),
                approval_restricted=class_id in restricted_ids,
            )
        )
    output_dir.mkdir(parents=True)
    supports_path = output_dir / "supports.bin"
    prototypes_path = output_dir / "prototypes.bin"
    adapter_path = output_dir / "adapter.bin"
    with supports_path.open("wb") as stream:
        np.save(stream, supports)
    with prototypes_path.open("wb") as stream:
        np.save(stream, prototypes)
    with adapter_path.open("wb") as stream:
        np.savez(stream, weight=weight, bias=bias)
    sanitized = [
        {
            "support_index": index,
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "image_sha256": row["image_sha256"],
            "perceptual_group_id": row["perceptual_group_id"],
            "capture_session_id": row.get("capture_session_id"),
            "physical_item_id": row.get("physical_item_id"),
        }
        for index, row in enumerate(records)
    ]
    source_manifest_path = output_dir / "source-manifest.jsonl"
    source_manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sanitized),
        encoding="utf-8",
    )
    metadata = CatalogMetadata(
        authentication=authentication,
        catalog_version=catalog_version,
        store_id=store_id,
        embedder_id=runtime.metadata.embedder.embedder_id,
        embedder_version=runtime.metadata.embedder.version,
        classifier_policy_version=runtime.metadata.classifier_policy.version,
        embedding_dimension=supports.shape[1],
        support_count_per_class=10,
        support_count=len(records),
        labels=labels,
        source_manifest_sha256=sha256_file(source_manifest_path),
        decision_head="ridge_adapter",
        adapter_filename="adapter.bin",
    )
    activation = CatalogActivation(
        state="active_restricted" if restricted_ids else "active",
        restricted_class_ids=sorted(restricted_ids),
        restricted_pairs=restricted_pairs,
        reasons=["CATALOG_CONFUSABLE_PAIR"] if restricted_ids else [],
    )
    (output_dir / "catalog.json").write_bytes(
        _canonical_json(metadata.model_dump(mode="json", exclude_none=True))
    )
    (output_dir / "activation.json").write_bytes(
        _canonical_json(activation.model_dump(mode="json"))
    )
    statistics = {
        "schema_version": "2.0",
        "support_count": len(records),
        "class_count": len(class_ids),
        "compactness": {label.class_id: label.compactness for label in labels},
        "nearest_similarity": {label.class_id: label.nearest_similarity for label in labels},
        "adapter_fit": {
            "algorithm": "closed_form_ridge_with_unregularized_bias",
            "alpha": runtime.metadata.classifier_policy.ridge_alpha,
            "feature_space": "l2_normalized_frozen_embedder_output",
            "source_manifest_sha256": sha256_file(manifest_path),
            "support_augmentation": augmentation_statistics,
        },
        "automatic_activation": True,
        "customer_multi_object_test_required": False,
    }
    (output_dir / "statistics.json").write_bytes(_canonical_json(statistics))
    signed_files = {
        "catalog.json",
        "activation.json",
        "supports.bin",
        "prototypes.bin",
        "adapter.bin",
        "statistics.json",
        "source-manifest.jsonl",
    }
    checksums = {filename: sha256_file(output_dir / filename) for filename in sorted(signed_files)}
    checksum_bytes = _canonical_json(checksums)
    (output_dir / "checksums.json").write_bytes(checksum_bytes)
    if authentication == "HMAC-SHA256":
        if signing_key is None or len(signing_key) < 16 or not key_id:
            raise ValueError("HMAC Catalogs require a key ID and at least 16 key bytes")
        signature = CatalogSignature(
            key_id=key_id,
            digest=hmac.new(signing_key, checksum_bytes, hashlib.sha256).hexdigest(),
        )
        (output_dir / "signature.json").write_bytes(
            _canonical_json(signature.model_dump(mode="json"))
        )
    elif authentication != "CHECKSUM-SHA256":
        raise ValueError("unsupported Catalog authentication mode")
    return {
        "catalog_version": catalog_version,
        "state": activation.state,
        "class_count": len(class_ids),
        "support_count": len(records),
        "restricted_class_count": len(restricted_ids),
        "restricted_pair_count": len(restricted_pairs),
        "authentication": authentication,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and automatically activate a Store Catalog")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--catalog-version", default="2.0.1")
    parser.add_argument(
        "--authentication",
        choices=("CHECKSUM-SHA256", "HMAC-SHA256"),
        default="CHECKSUM-SHA256",
    )
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--key-id")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    args = parser.parse_args(argv)
    secret = os.environ.get(args.signing_key_env, "").encode() or None
    report = build_catalog(
        args.runtime,
        args.dataset_root,
        args.manifest,
        args.output_dir,
        store_id=args.store_id,
        catalog_version=args.catalog_version,
        signing_key=secret,
        key_id=args.key_id,
        authentication=args.authentication,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
