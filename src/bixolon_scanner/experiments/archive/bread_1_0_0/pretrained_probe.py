from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ....training.bread_dataset import TRAINING_SOURCES, audit_bread_dataset

SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def _training_views(image: Image.Image) -> tuple[Image.Image, ...]:
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


def _evaluation_samples(
    root: Path,
    evaluation_sets: tuple[str, ...] = ("multi_object_scenes",),
) -> Iterator[tuple[Image.Image, int, str]]:
    available = {
        "multi_object_scenes": "multi_object_instances.json",
        "scan_log_samples": "scan_log_instances.json",
    }
    for dataset_name in evaluation_sets:
        annotation_name = available[dataset_name]
        annotation_path = root / "annotations" / annotation_name
        coco = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        images = {int(row["id"]): row for row in coco["images"]}
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in coco["annotations"]:
            annotations[int(row["image_id"])].append(row)
        for image_id, rows in sorted(annotations.items()):
            image_row = images[image_id]
            if str(image_row.get("status", "ANNOTATED")) == "RECAPTURE":
                continue
            source_path = (annotation_path.parent / str(image_row["file_name"])).resolve()
            source_path.relative_to(root)
            with Image.open(source_path) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
            for row in sorted(rows, key=lambda value: int(value["id"])):
                x, y, width, height = (float(value) for value in row["bbox"])
                margin_x = width * 0.05
                margin_y = height * 0.05
                box = (
                    max(0, int(np.floor(x - margin_x))),
                    max(0, int(np.floor(y - margin_y))),
                    min(source.width, int(np.ceil(x + width + margin_x))),
                    min(source.height, int(np.ceil(y + height + margin_y))),
                )
                yield source.crop(box), int(row["category_id"]) - 1, dataset_name


def _batched(
    rows: Iterable[tuple[Image.Image, int, str]], batch_size: int
) -> Iterator[list[tuple[Image.Image, int, str]]]:
    batch: list[tuple[Image.Image, int, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _image_features(model: Any, pixel_values: Any) -> Any:
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
            raise ValueError("pretrained image model does not expose a pooled representation")
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _embed(
    rows: Iterable[tuple[Image.Image, int, str]],
    *,
    model: Any,
    processor: Any,
    torch: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    elapsed = 0.0
    with torch.inference_mode():
        for batch in _batched(rows, batch_size):
            values = processor(
                images=[row[0] for row in batch], return_tensors="pt"
            ).pixel_values.to("cuda")
            started = time.perf_counter()
            embedded = _image_features(model, values)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            features.append(embedded.float().cpu().numpy())
            labels.extend(row[1] for row in batch)
            groups.extend(row[2] for row in batch)
    return np.concatenate(features), np.asarray(labels), groups, elapsed


def _metrics(scores: np.ndarray, labels: np.ndarray, groups: list[str]) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1, kind="stable")
    top1 = order[:, 0] == labels
    top3 = np.any(order[:, :3] == labels[:, None], axis=1)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        return {
            "sample_count": int(mask.sum()),
            "top1_accuracy": float(top1[mask].mean()),
            "top3_accuracy": float(top3[mask].mean()),
            "top1_error_count": int((~top1[mask]).sum()),
        }

    result = {"ALL": summarize(np.ones(len(labels), dtype=bool))}
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        result[group.upper()] = summarize(group_array == group)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from sklearn.svm import LinearSVC
    from transformers import AutoModel, AutoProcessor

    root = args.dataset_root.resolve()
    training_records, metadata = audit_bread_dataset(
        root,
        training_source=args.training_source,
    )
    training_rows: list[tuple[Image.Image, int, str]] = []
    for row in training_records:
        with Image.open(root / row["image_path"]) as opened:
            views = _training_views(opened)
        training_rows.extend((view, int(row["category_id"]) - 1, "train") for view in views)

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
    train_features, train_labels, _, train_seconds = _embed(
        training_rows,
        model=model,
        processor=processor,
        torch=torch,
        batch_size=args.batch_size,
    )
    evaluation_features, evaluation_labels, groups, evaluation_seconds = _embed(
        _evaluation_samples(root, tuple(args.evaluation_set)),
        model=model,
        processor=processor,
        torch=torch,
        batch_size=args.batch_size,
    )
    classifier = LinearSVC(C=args.svm_c, dual="auto", max_iter=20_000)
    classifier.fit(train_features, train_labels)
    scores = classifier.decision_function(evaluation_features)
    result = {
        "schema_version": "1.0",
        "evaluation": "pretrained_classifier_upper_bound_probe",
        "promotion_status": "diagnostic_only",
        "dataset_version": metadata["dataset_version"],
        "training_source": args.training_source,
        "evaluation_sets": list(args.evaluation_set),
        "training_contract": {
            "original_count": len(training_records),
            "derived_views_per_original": 6,
            "total_training_views": len(training_rows),
            "evaluation_images_used_for_training": False,
        },
        "model": {
            "name": args.model_name,
            "revision": args.revision,
            "method": "frozen_image_encoder_plus_linear_svc",
            "svm_c": args.svm_c,
        },
        "crop_contract": "ground_truth_bbox_plus_5_percent_margin_box_resize",
        "metrics": _metrics(scores, evaluation_labels, groups),
        "encoder_per_image_ms": {
            "training_views_mean": train_seconds * 1000.0 / len(training_rows),
            "evaluation_crops_mean": evaluation_seconds * 1000.0 / len(evaluation_labels),
            "batch_size": args.batch_size,
            "scope": "CUDA encoder only; excludes decode, crop, processor and linear head",
        },
        "limitations": [
            "Uses ground-truth crops, so this is a classifier ceiling diagnostic rather than a Worker gate.",
            "The available evaluation set is not an independent locked test set.",
            "No ONNX parity or end-to-end latency evidence is produced by this probe.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a frozen pretrained bread classifier")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="google/siglip2-base-patch16-224")
    parser.add_argument("--revision", default=SIGLIP2_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--training-source", choices=tuple(TRAINING_SOURCES), default="single_objects"
    )
    parser.add_argument(
        "--evaluation-set",
        action="append",
        choices=("multi_object_scenes", "scan_log_samples"),
        default=None,
    )
    args = parser.parse_args()
    if args.evaluation_set is None:
        args.evaluation_set = ["multi_object_scenes"]
    evaluate(args)


if __name__ == "__main__":
    main()
