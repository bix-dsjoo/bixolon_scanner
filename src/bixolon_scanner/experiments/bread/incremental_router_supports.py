from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...runtime.catalog import OnnxEmbedder, l2_normalize, verification_runtime_package
from ...runtime.onnx_session import ExecutionProvider
from ...training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    direct_roi_recipe_sha256,
    prepare_direct_roi_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records.sort(key=lambda row: (str(row["class_id"]), str(row["image_sha256"])))
    counts = Counter(str(row["class_id"]) for row in records)
    if not records or len(set(counts.values())) != 1:
        raise ValueError("incremental router supports must be class balanced")
    return records


def _recipe(runtime) -> DirectRoiRecipe:
    policy = runtime.metadata.classifier_policy.support_augmentation
    return DirectRoiRecipe(
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


def build_incremental_router_supports(
    runtime_dir: Path,
    dataset_root: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    views_per_source: int,
    provider: ExecutionProvider,
    cuda_dll_dir: Path | None = None,
) -> dict:
    if views_per_source < 1:
        raise ValueError("incremental router support augmentation requires at least one view")
    runtime = load_runtime_package_v2(runtime_dir)
    independent_runtime = verification_runtime_package(runtime)
    records = _records(manifest_path)
    recipe = _recipe(runtime)
    primary_embedder = OnnxEmbedder(runtime, provider, cuda_dll_dir)
    independent_embedder = OnnxEmbedder(independent_runtime, provider, cuda_dll_dir)
    images: list[Image.Image] = []
    labels: list[int] = []
    source_indices: list[int] = []
    class_ids = sorted({str(row["class_id"]) for row in records})
    class_indices = {class_id: index for index, class_id in enumerate(class_ids)}
    try:
        for source_index, row in enumerate(records):
            source_path = (dataset_root / str(row["image_path"])).resolve()
            if _sha256(source_path) != str(row["image_sha256"]):
                raise ValueError("incremental router source checksum differs from its manifest")
            with Image.open(source_path) as source:
                image = source.convert("RGB")
            images.append(image)
            labels.append(class_indices[str(row["class_id"])])
            source_indices.append(source_index)
            prepared = prepare_direct_roi_source(image, recipe)
            for view in range(1, views_per_source + 1):
                sample = augment_direct_roi(
                    image,
                    source_sha256=str(row["image_sha256"]),
                    category_id=int(row["category_id"]),
                    seed=(
                        runtime.metadata.classifier_policy.support_augmentation.seed
                        + source_index * 10_000
                        + view
                    ),
                    recipe=recipe,
                    prepared_cutout=prepared,
                )
                images.append(sample.image)
                labels.append(class_indices[str(row["class_id"])])
                source_indices.append(source_index)

        primary_raw = primary_embedder.embed_images_raw(images)
        rotated_images = [image.transpose(Image.Transpose.ROTATE_180) for image in images]
        try:
            rotated_raw = primary_embedder.embed_images_raw(rotated_images)
        finally:
            for image in rotated_images:
                image.close()
        independent_raw = independent_embedder.embed_images_raw(images)
    finally:
        for image in images:
            image.close()

    primary = l2_normalize(primary_raw)
    rotation = l2_normalize((primary_raw + rotated_raw) * np.float32(0.5))
    independent = l2_normalize(independent_raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez(
            stream,
            primary=primary,
            rotation=rotation,
            independent=independent,
            labels=np.asarray(labels, dtype=np.int64),
            source_indices=np.asarray(source_indices, dtype=np.int64),
            class_ids=np.asarray(class_ids),
        )
    report = {
        "schema_version": "1.0",
        "operation": "build_incremental_router_augmented_supports",
        "runtime_version": runtime.metadata.worker_version,
        "provider": provider,
        "manifest_sha256": _sha256(manifest_path),
        "source_count": len(records),
        "class_count": len(class_ids),
        "views_per_source": views_per_source,
        "feature_count": len(labels),
        "recipe_sha256": direct_roi_recipe_sha256(recipe),
        "output_sha256": _sha256(output_path),
        "multi_object_images_used": False,
        "multi_object_product_labels_used": False,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build augmented supports for a new-SKU router")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-per-source", type=int, default=8)
    parser.add_argument("--provider", choices=("cpu", "cuda", "openvino"), default="openvino")
    parser.add_argument("--cuda-dll-dir", type=Path)
    args = parser.parse_args(argv)
    report = build_incremental_router_supports(
        args.runtime,
        args.dataset_root,
        args.manifest,
        args.output,
        views_per_source=args.views_per_source,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
