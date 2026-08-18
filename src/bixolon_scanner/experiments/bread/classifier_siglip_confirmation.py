from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ...training.bread_dataset import audit_bread_dataset

SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def training_views(image: Image.Image) -> tuple[Image.Image, ...]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    fill = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
    return (
        image,
        ImageOps.mirror(image),
        image.rotate(-12.0, resample=Image.Resampling.BILINEAR, fillcolor=fill),
        image.rotate(12.0, resample=Image.Resampling.BILINEAR, fillcolor=fill),
        ImageEnhance.Brightness(image).enhance(0.85),
        ImageEnhance.Brightness(image).enhance(1.15),
    )


def normalized_tensor_to_image(tensor: np.ndarray) -> Image.Image:
    values = np.asarray(tensor, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("evaluation tensor must have shape [3, height, width]")
    rgb = values.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    pixels = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def batches(values: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def image_features(model: Any, pixel_values: Any) -> Any:
    if hasattr(model, "get_image_features"):
        output = model.get_image_features(pixel_values=pixel_values)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
    else:
        output = model(pixel_values=pixel_values)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            features = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            features = output.last_hidden_state[:, 0]
        else:
            raise ValueError("pretrained image model has no pooled representation")
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def embed(
    images: Iterable[Image.Image],
    *,
    model: Any,
    processor: Any,
    torch: Any,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    outputs: list[np.ndarray] = []
    elapsed = 0.0
    with torch.inference_mode():
        for batch in batches(images, batch_size):
            pixels = processor(images=batch, return_tensors="pt").pixel_values.to("cuda")
            started = time.perf_counter()
            features = image_features(model, pixels)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            outputs.append(features.float().cpu().numpy())
    return np.concatenate(outputs), elapsed


def metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1, kind="stable")
    return {
        "sample_count": len(targets),
        "top1_error_count": int(np.count_nonzero(order[:, 0] != targets)),
        "top3_miss_count": int(np.count_nonzero(~np.any(order[:, :3] == targets[:, None], axis=1))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from sklearn.svm import LinearSVC
    from transformers import AutoModel, AutoProcessor

    root = args.dataset_root.resolve()
    records, metadata = audit_bread_dataset(root, training_source="single_objects")
    training_images: list[Image.Image] = []
    training_targets: list[int] = []
    for row in records:
        with Image.open(root / row["image_path"]) as opened:
            views = training_views(opened)
        training_images.extend(views)
        training_targets.extend([int(row["category_id"]) - 1] * len(views))

    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(tensors) != len(rows):
        raise ValueError("evaluation tensors and records are not aligned")
    evaluation_targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    evaluation_images = (normalized_tensor_to_image(tensor) for tensor in tensors)

    model = (
        AutoModel.from_pretrained(
            args.model_name,
            revision=args.revision,
            local_files_only=True,
        )
        .eval()
        .to("cuda")
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=True,
    )
    training_features, training_seconds = embed(
        training_images,
        model=model,
        processor=processor,
        torch=torch,
        batch_size=args.batch_size,
    )
    evaluation_features, evaluation_seconds = embed(
        evaluation_images,
        model=model,
        processor=processor,
        torch=torch,
        batch_size=args.batch_size,
    )
    classifier = LinearSVC(C=args.svm_c, dual="auto", max_iter=20_000)
    classifier.fit(training_features, np.asarray(training_targets, dtype=np.int64))
    if not np.array_equal(classifier.classes_, np.arange(20)):
        raise ValueError("SigLIP confirmation classifier does not cover all 20 classes")
    scores = classifier.decision_function(evaluation_features).astype(np.float32)
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.logits_output,
        targets=evaluation_targets,
        **{args.logit_name: scores},
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_siglip2_classifier_confirmation_probe",
        "selection_scope": "fixed_recipe_development_diagnostic_not_locked_test",
        "training_source": "single_objects",
        "mixed_support_sources": False,
        "dataset_version": metadata["dataset_version"],
        "training_original_count": len(records),
        "training_view_count": len(training_images),
        "evaluation_sample_count": len(evaluation_targets),
        "evaluation_used_for_training": False,
        "model": {
            "name": args.model_name,
            "revision": args.revision,
            "method": "frozen_image_encoder_plus_linear_svc",
            "svm_c": args.svm_c,
        },
        "metrics": metrics(scores, evaluation_targets),
        "encoder_cuda_ms_per_crop": {
            "training": training_seconds * 1000.0 / len(training_images),
            "evaluation": evaluation_seconds * 1000.0 / len(evaluation_targets),
            "batch_size": args.batch_size,
        },
        "promotion_ready": False,
        "promotion_blocker": "independent locked test and ONNX/Worker validation pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed SigLIP2 confirmation head on detector ROIs"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="google/siglip2-base-patch16-224")
    parser.add_argument("--revision", default=SIGLIP2_REVISION)
    parser.add_argument("--logit-name", default="siglip2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--svm-c", type=float, default=1.0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
