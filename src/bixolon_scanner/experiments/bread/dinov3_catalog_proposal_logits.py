from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ...contracts.catalog import load_store_catalog_package
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...runtime.catalog import OnnxCatalogClassifier, OnnxEmbedder
from ...runtime.preprocessing import prepare_rgb
from ...training.data import read_manifest


def catalog_target_indices(class_ids: list[str], category_ids: np.ndarray) -> np.ndarray:
    """Map manifest category ids to the catalog order without assuming sorted labels."""
    lookup = {class_id: index for index, class_id in enumerate(class_ids)}
    targets = []
    for category_id in np.asarray(category_ids, dtype=np.int64):
        class_id = f"bread_{int(category_id):02d}"
        if class_id not in lookup:
            raise ValueError(f"manifest category is absent from Catalog: {class_id}")
        targets.append(lookup[class_id])
    return np.asarray(targets, dtype=np.int64)


def clipped_crop_box(box: np.ndarray, *, width: int, height: int) -> tuple[int, int, int, int]:
    values = np.asarray(box, dtype=np.float32)
    if values.shape != (4,) or np.any(~np.isfinite(values)):
        raise ValueError("proposal box must contain four finite coordinates")
    left = int(np.floor(np.clip(values[0], 0.0, float(width))))
    top = int(np.floor(np.clip(values[1], 0.0, float(height))))
    right = int(np.ceil(np.clip(values[2], 0.0, float(width))))
    bottom = int(np.ceil(np.clip(values[3], 0.0, float(height))))
    if right <= left or bottom <= top:
        raise ValueError("proposal box is empty after clipping")
    return left, top, right, bottom


def _prediction_map(template: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for fold in (0, 1, 2):
        path = Path(str(template).format(fold=fold))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in rows:
                raise ValueError(f"duplicate proposal prediction for image {image_id}")
            rows[image_id] = row
    return rows


def _positive_metrics(
    logits: np.ndarray,
    retrieval_logits: np.ndarray,
    max_iou: np.ndarray,
    target_indices: np.ndarray,
) -> dict:
    positive = np.asarray(max_iou) >= 0.5
    selected_logits = np.asarray(logits)[positive]
    selected_retrieval = np.asarray(retrieval_logits)[positive]
    selected_targets = np.asarray(target_indices)[positive]
    if not len(selected_targets):
        raise ValueError("proposal cache contains no IoU>=0.5 positive")
    order = np.argsort(-selected_logits, axis=1, kind="stable")
    retrieval_order = np.argsort(-selected_retrieval, axis=1, kind="stable")
    return {
        "positive_iou_0_5_count": int(positive.sum()),
        "adapter_top1_accuracy": float(np.mean(order[:, 0] == selected_targets)),
        "adapter_top3_accuracy": float(
            np.mean(np.any(order[:, :3] == selected_targets[:, None], axis=1))
        ),
        "retrieval_top1_accuracy": float(np.mean(retrieval_order[:, 0] == selected_targets)),
        "retrieval_top3_accuracy": float(
            np.mean(
                np.any(
                    retrieval_order[:, :3] == selected_targets[:, None],
                    axis=1,
                )
            )
        ),
    }


def run(args: argparse.Namespace) -> dict:
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and not row.get("exclude_from_detector_training", False)
    ]
    predictions = _prediction_map(args.predictions_template)
    expected_ids = {int(row["image_id"]) for row in records}
    if set(predictions) != expected_ids:
        raise ValueError("proposal predictions do not align with development records")

    with np.load(args.proposal_features, allow_pickle=False) as proposal_cache:
        cache_image_ids = proposal_cache["image_id"].copy()
        max_iou = proposal_cache["max_iou"].copy()
        target_class = proposal_cache["target_class"].copy()

    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(
        args.catalog,
        expected_store_id=args.store_id,
    )
    embedder = OnnxEmbedder(runtime, args.provider, args.cuda_dll_dir)
    classifier = OnnxCatalogClassifier(runtime, catalog, embedder)
    class_ids = [label.class_id for label in classifier.labels]

    image_rows = []
    fold_rows = []
    logits_rows = []
    retrieval_rows = []
    approval_rows = []
    blocked_rows = []
    recapture_rows = []
    started = time.perf_counter()
    try:
        for record_index, record in enumerate(records):
            image_id = int(record["image_id"])
            prediction = predictions[image_id]
            boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            image_rows.append(np.full(len(boxes), image_id, dtype=np.int64))
            fold_rows.append(np.full(len(boxes), int(record["fold"]), dtype=np.int8))
            with Image.open(args.dataset_root / record["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                try:
                    if image.size != (int(record["width"]), int(record["height"])):
                        raise ValueError(f"manifest image size mismatch for image {image_id}")
                    for start in range(0, len(boxes), args.batch_size):
                        batch_boxes = boxes[start : start + args.batch_size]
                        tensors = np.stack(
                            [
                                prepare_rgb(
                                    image.crop(
                                        clipped_crop_box(
                                            box,
                                            width=image.width,
                                            height=image.height,
                                        )
                                    ),
                                    embedder.metadata.input_size,
                                    embedder.metadata.mean,
                                    embedder.metadata.std,
                                    reducing_gap=embedder.metadata.resize_reducing_gap,
                                )
                                for box in batch_boxes
                            ]
                        )
                        raw = embedder.embed_prepared_tensors_raw(batch=tensors)
                        result = classifier.classify_embeddings(raw)
                        if result.retrieval_logits is None:
                            raise ValueError("Catalog classifier did not return retrieval logits")
                        logits_rows.append(np.asarray(result.logits, dtype=np.float16))
                        retrieval_rows.append(np.asarray(result.retrieval_logits, dtype=np.float16))
                        approval_rows.append(np.asarray(result.approval_scores, dtype=np.float16))
                        blocked_rows.append(np.asarray(result.approval_blocked, dtype=np.bool_))
                        recapture_rows.append(
                            np.asarray(
                                [reason is not None for reason in result.segment_recapture_reasons],
                                dtype=np.bool_,
                            )
                        )
                finally:
                    image.close()
            if (record_index + 1) % 25 == 0:
                print(json.dumps({"images_processed": record_index + 1}), flush=True)
    finally:
        classifier.close()

    image_ids = np.concatenate(image_rows)
    folds = np.concatenate(fold_rows)
    logits = np.concatenate(logits_rows).astype(np.float32)
    retrieval_logits = np.concatenate(retrieval_rows).astype(np.float32)
    approval_scores = np.concatenate(approval_rows).astype(np.float32)
    approval_blocked = np.concatenate(blocked_rows)
    segment_recapture = np.concatenate(recapture_rows)
    if not np.array_equal(image_ids, cache_image_ids):
        raise ValueError("Catalog logits and proposal feature cache order differ")
    targets = catalog_target_indices(class_ids, target_class)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        logits=logits.astype(np.float16),
        retrieval_logits=retrieval_logits.astype(np.float16),
        approval_scores=approval_scores.astype(np.float16),
        approval_blocked=approval_blocked,
        segment_recapture=segment_recapture,
        target_index=targets,
        image_id=image_ids,
        fold=folds,
        class_ids=np.asarray(class_ids),
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_catalog_oof_proposal_logits",
        "selection_scope": "DINOv3 detector group-aware OOF proposals; Catalog is fixed",
        "image_count": len(records),
        "proposal_count": len(image_ids),
        "class_count": len(class_ids),
        "neighbor_mask_applied": False,
        "neighbor_mask_reason": (
            "raw recall-first proposals overlap heavily; contextual masking is deferred "
            "until after proposal selection"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
        **_positive_metrics(logits, retrieval_logits, max_iou, targets),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cache fixed DINOv3 Catalog logits for group-OOF proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-template", type=Path, required=True)
    parser.add_argument("--proposal-features", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
