from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from ...configuration import load_json_config, resolve_config_path
from ...contracts.model_package import sha256_file
from ...training.calibration import (
    fit_temperature,
    select_approval_threshold,
    softmax,
    topk_accuracy,
)
from ...training.models import build_dino_classifier, require_torch, set_frozen_backbone
from .worker_gate import (
    _detector_phase_complete,
    _final_detector_artifacts,
    load_worker_gated_records,
    prepare_detector_domain_adaptation,
    prepare_detector_phase,
    prepare_final_test_records,
    train_final_detector,
)

SCHEMA_VERSION = "2.0"
CLASSIFIER_STAGE_SELECTION_POLICY = "calibration_risk_then_coverage_else_top3_top1_v2"
LEVELS = ("easy", "medium", "hard")
TRAIN_NAME = re.compile(r"^(?P<barcode>.+)_camera(?P<camera>\d+)-")
CHECKOUT_GROUP = re.compile(r"-(?P<group>\d+)\.[^.]+$")
TEST_ACCESS_SEAL_FIELDS = (
    "test_access_started_at",
    "test_access_model_lock_sha256",
    "test_access_final_detector_complete_sha256",
    "test_access_final_detector_checkpoint_sha256",
)
TEST_RESULT_SEAL_FIELDS = (
    "final_test_report_sha256",
    "final_test_detector_report_sha256",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_json_durable(path: Path, value: Any) -> None:
    """Atomically persist a seal before any protected dataset access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    finally:
        temporary.unlink(missing_ok=True)


def _write_experiment_metadata(path: Path, value: dict[str, Any], *, durable: bool = False) -> None:
    """Persist experiment metadata without allowing a test seal downgrade."""
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if previous.get("test_accessed") is True:
        if value.get("test_accessed") is not True:
            raise ValueError("test-access seal cannot be downgraded or removed")
        for field in TEST_ACCESS_SEAL_FIELDS:
            if previous.get(field) is None or value.get(field) != previous.get(field):
                raise ValueError(f"test-access seal field changed or missing: {field}")
    if value.get("test_accessed") is True and any(
        value.get(field) is None for field in TEST_ACCESS_SEAL_FIELDS
    ):
        raise ValueError("complete test-access seal evidence is required")
    for field in TEST_RESULT_SEAL_FIELDS:
        if previous.get(field) is not None and value.get(field) != previous.get(field):
            raise ValueError(f"test-result seal field changed or missing: {field}")
    if durable:
        _write_json_durable(path, value)
    else:
        _write_json(path, value)


def _sealed_test_metadata(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "prepared" / "experiment.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return metadata if metadata.get("test_accessed") is True else None


def _reject_sealed_mutation(output_dir: Path, operation: str) -> None:
    if _sealed_test_metadata(output_dir) is not None:
        raise RuntimeError(
            f"post-test output is immutable; {operation} requires a fresh output directory"
        )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical_json(record) + "\n" for record in records), encoding="utf-8")


def _environment_snapshot(config: dict[str, Any], weights: Path) -> dict[str, Any]:
    torch = require_torch()

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    cuda_available = bool(torch.cuda.is_available())
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": package_version("numpy"),
            "pillow": package_version("pillow"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
        },
        "weights": {
            "path_name": weights.name,
            "sha256": sha256_file(weights),
        },
        "config": config,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_config(path: Path) -> dict[str, Any]:
    raw = load_json_config(path)
    required = ("experiment", "detector", "sampling", "training")
    if any(not isinstance(raw.get(key), dict) for key in required):
        raise ValueError(f"config must contain objects: {', '.join(required)}")
    mode = str(raw["experiment"].get("mode", "data_scale"))
    if mode not in {"data_scale", "full_dataset"}:
        raise ValueError(f"unsupported experiment mode: {mode}")
    if mode == "full_dataset" and [int(value) for value in raw["experiment"].get("seeds", [])] != [
        20260810
    ]:
        raise ValueError("full_dataset mode requires the single seed 20260810")
    return raw


def _is_full_dataset(config: dict[str, Any]) -> bool:
    return str(config["experiment"].get("mode", "data_scale")) == "full_dataset"


def _group_from_filename(filename: str) -> str:
    match = CHECKOUT_GROUP.search(filename)
    if match is None:
        raise ValueError(f"checkout filename has no terminal group id: {filename}")
    return match.group("group")


def _validate_bbox(bbox: list[float], width: int, height: int, identity: str) -> None:
    if len(bbox) != 4:
        raise ValueError(f"invalid bbox length: {identity}")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"invalid bbox values: {identity}")
    if x + box_width > width + 1.0 or y + box_height > height + 1.0:
        raise ValueError(f"bbox exceeds image dimensions: {identity}")


def _load_coco(
    dataset_root: Path,
    split: str,
    *,
    expected_categories: list[dict[str, Any]] | None = None,
    verify_files: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_path = dataset_root / f"instances_{split}2019.json"
    image_dir = dataset_root / f"{split}2019"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda row: int(row["id"]))
    category_ids = [int(row["id"]) for row in categories]
    if category_ids != list(range(1, len(categories) + 1)):
        raise ValueError(f"{split} category ids must be contiguous and one-based")
    if expected_categories is not None and categories != expected_categories:
        raise ValueError(f"{split} categories do not match train categories")

    images = {int(row["id"]): row for row in payload["images"]}
    if len(images) != len(payload["images"]):
        raise ValueError(f"{split} contains duplicate image ids")
    if verify_files:
        missing = [
            row["file_name"]
            for row in images.values()
            if not (image_dir / row["file_name"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{split} is missing {len(missing)} image files; first={missing[0]}"
            )

    records: list[dict[str, Any]] = []
    annotation_ids: set[int] = set()
    for annotation in payload["annotations"]:
        annotation_id = int(annotation["id"])
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"{split} contains duplicate annotation id {annotation_id}")
        annotation_ids.add(annotation_id)
        if image_id not in images or category_id not in category_ids:
            raise ValueError(f"{split} annotation {annotation_id} has an invalid reference")
        image = images[image_id]
        identity = f"{split}:{image_id}:{annotation_id}"
        _validate_bbox(annotation["bbox"], int(image["width"]), int(image["height"]), identity)
        record: dict[str, Any] = {
            "sample_id": identity,
            "split": split,
            "image_id": image_id,
            "annotation_id": annotation_id,
            "image_path": f"{split}2019/{image['file_name']}",
            "width": int(image["width"]),
            "height": int(image["height"]),
            "bbox_xywh": [float(value) for value in annotation["bbox"]],
            "category_id": category_id,
            "target": category_id - 1,
        }
        if split == "train":
            match = TRAIN_NAME.match(image["file_name"])
            if match is None:
                raise ValueError(
                    f"train filename does not encode barcode/camera: {image['file_name']}"
                )
            record["barcode"] = match.group("barcode")
            record["camera"] = int(match.group("camera"))
        else:
            level = image.get("level")
            if level not in LEVELS:
                raise ValueError(f"invalid checkout difficulty for {identity}: {level}")
            record["level"] = level
            record["group_id"] = _group_from_filename(image["file_name"])
        records.append(record)

    if split == "train":
        counts = defaultdict(int)
        for record in records:
            counts[record["image_id"]] += 1
        if any(count != 1 for count in counts.values()) or len(counts) != len(images):
            raise ValueError("train2019 must contain exactly one annotation per image")
    return records, categories


def _balanced_training_order(records: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[int(record["category_id"])].append(record)
    result: dict[str, list[str]] = {}
    for category_id, category_records in sorted(by_category.items()):
        rng = random.Random(seed * 1009 + category_id)
        cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for record in category_records:
            cells[(str(record["barcode"]), int(record["camera"]))].append(record)
        for values in cells.values():
            values.sort(key=lambda row: row["sample_id"])
            rng.shuffle(values)
        cameras = sorted({cell[1] for cell in cells})
        rng.shuffle(cameras)
        cells_by_camera: dict[int, list[tuple[str, int]]] = {}
        for camera in cameras:
            camera_cells = sorted(cell for cell in cells if cell[1] == camera)
            rng.shuffle(camera_cells)
            cells_by_camera[camera] = camera_cells
        cursors = {camera: 0 for camera in cameras}
        ordered: list[str] = []
        while len(ordered) < len(category_records):
            made_progress = False
            for camera in cameras:
                camera_cells = cells_by_camera[camera]
                for offset in range(len(camera_cells)):
                    index = (cursors[camera] + offset) % len(camera_cells)
                    cell = camera_cells[index]
                    if cells[cell]:
                        ordered.append(str(cells[cell].pop()["sample_id"]))
                        cursors[camera] = (index + 1) % len(camera_cells)
                        made_progress = True
                        break
            if not made_progress:
                break
        if len(ordered) != len(category_records) or len(set(ordered)) != len(ordered):
            raise RuntimeError(f"failed to order category {category_id}")
        result[str(category_id)] = ordered
    return result


def _deduplicate_roi_hashes(
    records: list[dict[str, Any]], roi_hashes: list[str]
) -> tuple[list[dict[str, Any]], list[int], dict[str, list[str]]]:
    if len(records) != len(roi_hashes):
        raise ValueError("record/hash length mismatch")
    seen: dict[tuple[int, str], str] = {}
    kept_records: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    duplicates: dict[str, list[str]] = defaultdict(list)
    for index, (record, roi_hash) in enumerate(zip(records, roi_hashes)):
        key = (int(record["category_id"]), str(roi_hash))
        if key in seen:
            duplicates[seen[key]].append(str(record["sample_id"]))
            continue
        seen[key] = str(record["sample_id"])
        kept_records.append(record)
        kept_indices.append(index)
    return kept_records, kept_indices, dict(duplicates)


def _deduplicate_full_dataset_roi_hashes(
    records: list[dict[str, Any]], roi_hashes: list[str]
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Deduplicate normalized ROIs globally while rejecting conflicting exact labels."""
    if len(records) != len(roi_hashes):
        raise ValueError("record/hash length mismatch")
    seen: dict[str, dict[str, Any]] = {}
    kept_records: list[dict[str, Any]] = []
    duplicates: dict[str, list[str]] = defaultdict(list)
    for record, roi_hash in zip(records, roi_hashes):
        digest = str(roi_hash)
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = record
            kept_records.append(record)
            continue
        if int(previous["category_id"]) != int(record["category_id"]):
            raise ValueError(
                "identical normalized ROI has conflicting categories: "
                f"{previous['sample_id']} and {record['sample_id']}"
            )
        duplicates[str(previous["sample_id"])].append(str(record["sample_id"]))
    return kept_records, dict(duplicates)


def _extract_exact_roi_hashes(
    records: list[dict[str, Any]], dataset_root: Path, image_size: int
) -> list[str]:
    """Hash the exact RGB bytes of each zero-margin ROI normalized to model input geometry."""
    hashes: list[str] = []
    for index, record in enumerate(records):
        with Image.open(dataset_root / str(record["image_path"])) as source:
            image = source.convert("RGB")
        normalized = _crop(image, list(record["bbox_xywh"]), 0.0).resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        hashes.append(hashlib.sha256(np.asarray(normalized, dtype=np.uint8).tobytes()).hexdigest())
        if index and index % 2000 == 0:
            print(json.dumps({"roi_hash_records": index, "total": len(records)}), flush=True)
    return hashes


def _class_imbalance(counts: Counter[int], category_count: int) -> dict[str, Any]:
    values = np.asarray([counts[category_id] for category_id in range(1, category_count + 1)])
    positive = values[values > 0]
    return {
        "minimum": int(values.min()) if len(values) else 0,
        "maximum": int(values.max()) if len(values) else 0,
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "max_to_min_ratio": float(positive.max() / positive.min()) if len(positive) else None,
        "missing_category_ids": [
            category_id for category_id in range(1, category_count + 1) if counts[category_id] == 0
        ],
    }


def _seed_rank(seed: int, category_id: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category_id}:{sample_id}".encode()).hexdigest()


def _visual_farthest_order(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    seed: int,
    anchor_pool_size: int,
    tie_tolerance: float,
) -> list[str]:
    if not records:
        return []
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(records):
        raise ValueError("embedding shape does not match records")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    centroid = vectors.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    center_distance = 1.0 - vectors @ centroid
    pool_size = min(max(1, int(anchor_pool_size)), len(records))
    central = np.argsort(center_distance, kind="stable")[:pool_size].tolist()
    category_id = int(records[0]["category_id"])
    anchor = min(
        central,
        key=lambda index: _seed_rank(seed, category_id, str(records[index]["sample_id"])),
    )
    selected = [anchor]
    remaining = set(range(len(records))) - {anchor}
    minimum_distance = 1.0 - vectors @ vectors[anchor]

    def metadata_score(index: int) -> tuple[int, int, int, int, str]:
        selected_records = [records[value] for value in selected]
        used_camera = {int(value["camera"]) for value in selected_records}
        used_surface = {str(value["surface"]) for value in selected_records}
        used_barcode = {str(value["barcode"]) for value in selected_records}
        used_view = {int(value["view_id"]) for value in selected_records}
        record = records[index]
        return (
            int(int(record["camera"]) not in used_camera),
            int(str(record["surface"]) not in used_surface),
            int(str(record["barcode"]) not in used_barcode),
            int(int(record["view_id"]) not in used_view),
            _seed_rank(seed, category_id, str(record["sample_id"])),
        )

    while remaining:
        best_distance = max(float(minimum_distance[index]) for index in remaining)
        tied = [
            index
            for index in remaining
            if best_distance - float(minimum_distance[index]) <= tie_tolerance
        ]
        chosen = max(tied, key=metadata_score)
        selected.append(chosen)
        remaining.remove(chosen)
        minimum_distance = np.minimum(minimum_distance, 1.0 - vectors @ vectors[chosen])
    return [str(records[index]["sample_id"]) for index in selected]


def _visual_training_orders(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    seeds: list[int],
    sampling: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    by_category: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_category[int(record["category_id"])].append(index)
    result: dict[str, dict[str, list[str]]] = {}
    for seed in seeds:
        orders: dict[str, list[str]] = {}
        for category_id, indices in sorted(by_category.items()):
            category_records = [records[index] for index in indices]
            orders[str(category_id)] = _visual_farthest_order(
                category_records,
                embeddings[indices],
                seed=int(seed),
                anchor_pool_size=int(sampling["anchor_pool_size"]),
                tie_tolerance=float(sampling["tie_tolerance"]),
            )
        result[str(seed)] = orders
    return result


def _extract_visual_embeddings(
    records: list[dict[str, Any]],
    dataset_root: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    torch = require_torch()
    import torchvision.transforms as transforms

    if not torch.cuda.is_available():
        raise RuntimeError("visual embedding extraction requires CUDA")
    sampling = config["sampling"]
    training = config["training"]
    size = int(training["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    model = (
        build_dino_classifier(
            str(training["backbone_kind"]),
            1,
            weights_path=args.weights,
            hub_repository=str(training["hub_repository"]),
        )
        .backbone.to("cuda")
        .eval()
    )
    arrays: list[np.ndarray] = []
    roi_hashes: list[str] = []
    batch: list[Any] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.stack(batch).to("cuda", non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(tensor)
        normalized = torch.nn.functional.normalize(output.float(), dim=1)
        arrays.append(normalized.cpu().numpy())
        batch.clear()

    for index, record in enumerate(records):
        with Image.open(dataset_root / str(record["image_path"])) as source:
            image = source.convert("RGB")
        crop = _crop(image, list(record["bbox_xywh"]), 0.0).resize(
            (size, size), Image.Resampling.BILINEAR
        )
        roi_hashes.append(hashlib.sha256(np.asarray(crop, dtype=np.uint8).tobytes()).hexdigest())
        batch.append(transform(crop))
        if len(batch) >= int(sampling["embedding_batch_size"]):
            flush()
        if index and index % 2000 == 0:
            print(json.dumps({"embedding_records": index, "total": len(records)}), flush=True)
    flush()
    torch.cuda.empty_cache()
    return np.concatenate(arrays).astype(np.float32), roi_hashes


def _render_sampling_audit(
    dataset_root: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    orders: dict[str, dict[str, list[str]]],
    embeddings: np.ndarray,
    first_n: int,
) -> None:
    by_id = {
        str(record["sample_id"]): (record, embeddings[index])
        for index, record in enumerate(records)
    }
    rows: list[dict[str, Any]] = []
    sheets_dir = output_dir / "prepared" / "sampling_contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for seed, categories in orders.items():
        seed_dir = sheets_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for category_id, sample_ids in categories.items():
            selected_ids = sample_ids[:first_n]
            selected = [by_id[sample_id] for sample_id in selected_ids]
            vectors = np.stack([value[1] for value in selected])
            pairwise = 1.0 - vectors @ vectors.T
            triangle = pairwise[np.triu_indices(len(selected), 1)]
            rows.append(
                {
                    "seed": int(seed),
                    "category_id": int(category_id),
                    "sample_count": len(selected),
                    "min_pairwise_cosine_distance": float(triangle.min()) if len(triangle) else 0.0,
                    "mean_pairwise_cosine_distance": float(triangle.mean())
                    if len(triangle)
                    else 0.0,
                    "camera_count": len({int(value[0]["camera"]) for value in selected}),
                    "surface_count": len({str(value[0]["surface"]) for value in selected}),
                    "barcode_count": len({str(value[0]["barcode"]) for value in selected}),
                    "view_count": len({int(value[0]["view_id"]) for value in selected}),
                    "sample_ids": "|".join(selected_ids),
                }
            )
            canvas = Image.new("RGB", (1100, first_n * 250), "white")
            draw = ImageDraw.Draw(canvas)
            for row_index, (record, _) in enumerate(selected):
                with Image.open(dataset_root / str(record["image_path"])) as source:
                    original = source.convert("RGB")
                x, y, width, height = [float(value) for value in record["bbox_xywh"]]
                overlay = original.copy()
                ImageDraw.Draw(overlay).rectangle(
                    (x, y, x + width, y + height), outline="red", width=8
                )
                overlay.thumbnail((430, 220))
                crop = _crop(original, list(record["bbox_xywh"]), 0.05)
                crop.thumbnail((300, 220))
                top = row_index * 250
                canvas.paste(overlay, (10, top + 10))
                canvas.paste(crop, (450, top + 10))
                draw.text(
                    (770, top + 20),
                    f"#{row_index + 1} camera={record['camera']}\n"
                    f"surface={record['surface']} view={record['view_id']}\n"
                    f"score={record['detector_score']:.4f}\n{Path(record['image_path']).name}",
                    fill="black",
                )
            canvas.save(seed_dir / f"class_{int(category_id):03d}.jpg", quality=90)
    audit_path = output_dir / "prepared" / "sampling_audit.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validation_partition(
    records: list[dict[str, Any]], category_count: int, seed: int, calibration_fraction: float
) -> dict[str, str]:
    if calibration_fraction != 0.5:
        raise ValueError("the RPC experiment currently requires a 0.5 calibration fraction")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["group_id"])].append(record)
    totals = np.zeros(category_count + len(LEVELS), dtype=np.float64)
    vectors: dict[str, np.ndarray] = {}
    for group_id, group_records in grouped.items():
        vector = np.zeros_like(totals)
        seen_images: set[int] = set()
        for record in group_records:
            vector[int(record["target"])] += 1
            if int(record["image_id"]) not in seen_images:
                vector[category_count + LEVELS.index(str(record["level"]))] += 1
                seen_images.add(int(record["image_id"]))
        vectors[group_id] = vector
        totals += vector
    target = totals / 2.0
    normalizer = np.maximum(target, 1.0)

    def digest(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    order = sorted(grouped, key=lambda key: (-len(grouped[key]), digest(key)))
    allocations = {"calibration": np.zeros_like(totals), "selection": np.zeros_like(totals)}
    counts = {"calibration": 0, "selection": 0}
    capacity = {"calibration": math.ceil(len(order) / 2), "selection": len(order) // 2}
    partition: dict[str, str] = {}
    for group_id in order:
        scores: dict[str, float] = {}
        for side, other in (("calibration", "selection"), ("selection", "calibration")):
            if counts[side] >= capacity[side]:
                scores[side] = float("inf")
                continue
            proposed = allocations[side] + vectors[group_id]
            scores[side] = float((((proposed - target) / normalizer) ** 2).mean())
            scores[side] += float((((allocations[other] - target) / normalizer) ** 2).mean())
        side = min(scores, key=lambda key: (scores[key], key))
        partition[group_id] = side
        allocations[side] += vectors[group_id]
        counts[side] += 1
    if set(grouped) != set(partition) or set(partition.values()) != {"calibration", "selection"}:
        raise RuntimeError("failed to partition validation groups")
    return partition


def _crop(image: Image.Image, bbox: list[float], margin: float) -> Image.Image:
    x, y, width, height = bbox
    margin_x = width * margin
    margin_y = height * margin
    return image.crop(
        (
            max(0, int(np.floor(x - margin_x))),
            max(0, int(np.floor(y - margin_y))),
            min(image.width, int(np.ceil(x + width + margin_x))),
            min(image.height, int(np.ceil(y + height + margin_y))),
        )
    )


def _cache_fingerprint(
    metadata: dict[str, Any], records: list[dict[str, Any]], options: dict[str, Any]
) -> str:
    value = {
        "metadata": {
            "schema_version": metadata["schema_version"],
            "mode": metadata.get("mode", "data_scale"),
            "category_count": metadata.get("category_count"),
            "sample_sizes": metadata.get("sample_sizes"),
            "seeds": metadata.get("seeds"),
            "train_counts": metadata.get("train_counts"),
            "source_hashes": metadata.get("source_hashes"),
            "detector_complete_sha256": metadata.get("detector_complete_sha256"),
        },
        "sample_ids": [record["sample_id"] for record in records],
        "bboxes": [record["bbox_xywh"] for record in records],
        "cache_size": int(options["cache_size"]),
        "train_margin_ratio": float(options["train_margin_ratio"]),
        "eval_margin_ratio": float(options["eval_margin_ratio"]),
    }
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _build_cache(
    dataset_root: Path,
    cache_dir: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    options: dict[str, Any],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint(metadata, records, options)
    metadata_path = cache_dir / "metadata.json"
    records_path = cache_dir / "records.jsonl"
    array_path = cache_dir / "images.npy"
    if resume and metadata_path.is_file() and records_path.is_file() and array_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete") is True:
            cached_records = _read_jsonl(records_path)
            if len(cached_records) == len(records):
                return cached_records

    size = int(options["cache_size"])
    partial_path = cache_dir / "images.partial.npy"
    images = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.uint8, shape=(len(records), size, size, 3)
    )
    by_path: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    cached_records: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        cached = dict(record)
        cached["cache_row"] = row
        cached_records.append(cached)
        by_path[str(record["image_path"])].append((row, record))

    def process_image(item: tuple[str, list[tuple[int, dict[str, Any]]]]):
        image_path, entries = item
        with Image.open(dataset_root / image_path) as source:
            image = source.convert("RGB")
        result: list[tuple[int, np.ndarray]] = []
        for row, record in entries:
            margin = (
                float(options["train_margin_ratio"])
                if record["split"] == "train"
                else float(options["eval_margin_ratio"])
            )
            roi = _crop(image, record["bbox_xywh"], margin)
            resized = roi.resize((size, size), Image.Resampling.BILINEAR)
            result.append((row, np.asarray(resized, dtype=np.uint8)))
        return result

    items = iter(sorted(by_path.items()))
    worker_count = max(1, int(options.get("workers", 4)))
    processed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending: set[Future] = set()
        for _ in range(worker_count * 2):
            try:
                pending.add(executor.submit(process_image, next(items)))
            except StopIteration:
                break
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                entries = future.result()
                for row, array in entries:
                    images[row] = array
                processed += len(entries)
                if processed and processed % 1000 < len(entries):
                    print(
                        json.dumps({"cache_entries": processed, "cache_total": len(records)}),
                        flush=True,
                    )
                try:
                    pending.add(executor.submit(process_image, next(items)))
                except StopIteration:
                    pass
    images.flush()
    del images
    partial_path.replace(array_path)
    _write_jsonl(records_path, cached_records)
    _write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "complete": True,
            "entry_count": len(records),
            "cache_size": size,
            "array_filename": array_path.name,
        },
    )
    return cached_records


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    _reject_sealed_mutation(args.output_dir, "prepare")
    experiment_options = config["experiment"]
    training_options = config["training"]
    prepared_dir = args.output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    detector_complete = args.output_dir / "detector" / "complete.json"
    if not detector_complete.is_file():
        raise FileNotFoundError("detector phase has not completed")
    if not _detector_phase_complete(args.output_dir / "detector", args.dataset_root, config):
        raise ValueError("detector complete marker or artifact checksum is invalid")
    _, categories = _load_coco(args.dataset_root, "train")
    expected_count = int(experiment_options["expected_num_classes"])
    if len(categories) != expected_count:
        raise ValueError(f"expected {expected_count} categories, found {len(categories)}")
    train_records, val_records, worker_gate_report = load_worker_gated_records(
        args.dataset_root, args.output_dir, config
    )
    _write_json(prepared_dir / "worker_gate_report.json", worker_gate_report)
    positive_train_records = [record for record in train_records if int(record["target"]) >= 0]
    hard_negative_records = [record for record in train_records if int(record["target"]) < 0]
    full_dataset = _is_full_dataset(config)
    counts = defaultdict(int)
    for record in positive_train_records:
        counts[int(record["category_id"])] += 1
    if full_dataset:
        hashes_path = prepared_dir / "roi_hashes.json"
        hash_metadata_path = prepared_dir / "roi_hashes_metadata.json"
        hash_fingerprint = hashlib.sha256(
            _canonical_json(
                {
                    "sample_ids": [record["sample_id"] for record in positive_train_records],
                    "bboxes": [record["bbox_xywh"] for record in positive_train_records],
                    "detector_complete_sha256": sha256_file(detector_complete),
                    "image_size": int(training_options["image_size"]),
                    "normalization": "rgb_zero_margin_bilinear_resize",
                }
            ).encode()
        ).hexdigest()
        reuse_hashes = False
        if args.resume and hashes_path.is_file() and hash_metadata_path.is_file():
            hash_metadata = json.loads(hash_metadata_path.read_text(encoding="utf-8"))
            reuse_hashes = hash_metadata.get("fingerprint") == hash_fingerprint
        if reuse_hashes:
            roi_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        else:
            roi_hashes = _extract_exact_roi_hashes(
                positive_train_records,
                args.dataset_root,
                int(training_options["image_size"]),
            )
            _write_json(hashes_path, roi_hashes)
            _write_json(
                hash_metadata_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": hash_fingerprint,
                    "record_count": len(train_records),
                    "normalization": "rgb_zero_margin_bilinear_resize",
                },
            )
        unique_records, duplicate_groups = _deduplicate_full_dataset_roi_hashes(
            positive_train_records, list(roi_hashes)
        )
        unique_counts = Counter(int(record["category_id"]) for record in unique_records)
        imbalance = _class_imbalance(unique_counts, expected_count)
        if imbalance["missing_category_ids"]:
            raise ValueError(
                "full_dataset has no eligible unique ROI for categories: "
                f"{imbalance['missing_category_ids']}"
            )
        orders: dict[str, dict[str, list[str]]] = {}
        selected_train = [
            dict(record, role="train")
            for record in sorted(unique_records, key=lambda row: str(row["sample_id"]))
        ]
        hard_negative_hashes = _extract_exact_roi_hashes(
            hard_negative_records,
            args.dataset_root,
            int(training_options["image_size"]),
        )
        positive_hashes = set(roi_hashes)
        seen_hard_hashes: set[str] = set()
        selected_hard_negatives: list[dict[str, Any]] = []
        hard_negative_duplicates: dict[str, list[str]] = defaultdict(list)
        for record, digest in zip(hard_negative_records, hard_negative_hashes):
            if digest in positive_hashes:
                hard_negative_duplicates["conflicts_with_positive"].append(str(record["sample_id"]))
                continue
            if digest in seen_hard_hashes:
                hard_negative_duplicates["duplicate_hard_negative"].append(str(record["sample_id"]))
                continue
            seen_hard_hashes.add(digest)
            selected_hard_negatives.append(record)
        selected_train.extend(selected_hard_negatives)
        duplicate_groups.update(hard_negative_duplicates)
    else:
        max_size = max(int(value) for value in experiment_options["sample_sizes"])
        undersized = {key: value for key, value in counts.items() if value < max_size}
        if undersized:
            raise ValueError(f"categories below max sample size {max_size}: {undersized}")
        embeddings_dir = prepared_dir / "embeddings"
        embeddings_path = embeddings_dir / "train.npy"
        hashes_path = embeddings_dir / "roi_hashes.json"
        embedding_metadata_path = embeddings_dir / "metadata.json"
        embedding_fingerprint = hashlib.sha256(
            _canonical_json(
                {
                    "sample_ids": [record["sample_id"] for record in train_records],
                    "bboxes": [record["bbox_xywh"] for record in train_records],
                    "detector_complete_sha256": sha256_file(detector_complete),
                    "weights_sha256": sha256_file(args.weights),
                    "image_size": int(training_options["image_size"]),
                }
            ).encode()
        ).hexdigest()
        reuse_embeddings = False
        if (
            args.resume
            and embeddings_path.is_file()
            and hashes_path.is_file()
            and embedding_metadata_path.is_file()
        ):
            embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
            reuse_embeddings = embedding_metadata.get("fingerprint") == embedding_fingerprint
        if reuse_embeddings:
            embeddings = np.load(embeddings_path)
            roi_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        else:
            embeddings, roi_hashes = _extract_visual_embeddings(
                train_records, args.dataset_root, args, config
            )
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)
            _write_json(hashes_path, roi_hashes)
            _write_json(
                embedding_metadata_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": embedding_fingerprint,
                    "record_count": len(train_records),
                    "embedding_size": int(embeddings.shape[1]),
                },
            )
        unique_records, kept_indices, duplicate_groups = _deduplicate_roi_hashes(
            train_records, list(roi_hashes)
        )
        unique_embeddings = embeddings[kept_indices]
        unique_counts = Counter(int(record["category_id"]) for record in unique_records)
        unique_undersized = {
            category_id: unique_counts[category_id]
            for category_id in range(1, expected_count + 1)
            if unique_counts[category_id] < max_size
        }
        if unique_undersized:
            raise ValueError(
                f"categories below max sample size after normal/dedup filtering {max_size}: "
                f"{unique_undersized}"
            )
        orders = _visual_training_orders(
            unique_records,
            unique_embeddings,
            [int(value) for value in experiment_options["seeds"]],
            config["sampling"],
        )
        imbalance = _class_imbalance(unique_counts, expected_count)
        train_by_id = {record["sample_id"]: record for record in train_records}
        selected_ids = {
            sample_id
            for seed_orders in orders.values()
            for category_order in seed_orders.values()
            for sample_id in category_order[:max_size]
        }
        selected_train = [
            dict(train_by_id[sample_id], role="train") for sample_id in sorted(selected_ids)
        ]
    partition = {str(record["group_id"]): str(record["role"]) for record in val_records}
    cache_records = selected_train + sorted(val_records, key=lambda row: row["sample_id"])
    source_hashes = {
        "instances_train2019.json": sha256_file(args.dataset_root / "instances_train2019.json"),
        "instances_val2019.json": sha256_file(args.dataset_root / "instances_val2019.json"),
    }
    previous: dict[str, Any] = {}
    metadata_path = prepared_dir / "experiment.json"
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": previous.get("created_at", datetime.now(UTC).isoformat()),
        "mode": "full_dataset" if full_dataset else "data_scale",
        "category_count": len(categories),
        "categories": categories,
        "sample_sizes": []
        if full_dataset
        else [int(value) for value in experiment_options["sample_sizes"]],
        "seeds": [int(value) for value in experiment_options["seeds"]],
        "source_hashes": source_hashes,
        "detector_complete_sha256": sha256_file(detector_complete),
        "worker_gate_report_sha256": sha256_file(prepared_dir / "worker_gate_report.json"),
        "train_counts_before_dedup": {str(key): counts[key] for key in sorted(counts)},
        "train_counts": {str(key): unique_counts[key] for key in sorted(unique_counts)},
        "train_class_imbalance": imbalance,
        "duplicate_group_count": len(duplicate_groups),
        "train_union_count": len(selected_train),
        "positive_train_count": len(selected_train) - len(selected_hard_negatives)
        if full_dataset
        else len(selected_train),
        "hard_negative_train_count": len(selected_hard_negatives) if full_dataset else 0,
        "validation_annotation_count": len(val_records),
        "validation_groups": partition,
        "orders": orders,
        "test_accessed": bool(previous.get("test_accessed", False)),
    }
    if previous.get("test_source_hash"):
        metadata["test_source_hash"] = previous["test_source_hash"]
    if previous.get("test_accessed") is True:
        for field in TEST_ACCESS_SEAL_FIELDS:
            metadata[field] = previous.get(field)
    _write_experiment_metadata(metadata_path, metadata)
    cached_records = _build_cache(
        args.dataset_root,
        prepared_dir / "cache",
        cache_records,
        metadata,
        training_options,
        resume=args.resume,
    )
    role_counts = defaultdict(int)
    for record in cached_records:
        role_counts[str(record["role"])] += 1
    metadata["cache_role_counts"] = dict(sorted(role_counts.items()))
    _write_experiment_metadata(metadata_path, metadata)
    _write_json(prepared_dir / "duplicate_groups.json", duplicate_groups)
    if not full_dataset:
        _render_sampling_audit(
            args.dataset_root,
            args.output_dir,
            unique_records,
            orders,
            unique_embeddings,
            int(config["sampling"]["contact_sheet_first_n"]),
        )
    return metadata


class RpcCachedDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        array_path: Path,
        *,
        image_size: int,
        training: bool,
        include_metadata: bool = False,
        crop_scale_min: float = 0.78,
        crop_ratio_min: float = 0.9,
        crop_ratio_max: float = 1.1,
    ) -> None:
        import torchvision.transforms as transforms

        self.records = records
        self.array_path = array_path
        self.images = None
        self.include_metadata = include_metadata
        if not 0.0 < crop_scale_min <= 1.0:
            raise ValueError("training crop scale minimum must be in (0, 1]")
        if not 0.0 < crop_ratio_min <= crop_ratio_max:
            raise ValueError("training crop ratio range is invalid")
        augmentation = (
            [
                transforms.RandomRotation(180, fill=(255, 255, 255)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(crop_scale_min, 1.0),
                    ratio=(crop_ratio_min, crop_ratio_max),
                ),
            ]
            if training
            else [transforms.Resize((image_size, image_size))]
        )
        self.transform = transforms.Compose(
            augmentation
            + [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        if self.images is None:
            self.images = np.load(self.array_path, mmap_mode="r")
        record = self.records[index]
        image = Image.fromarray(np.asarray(self.images[int(record["cache_row"])]), mode="RGB")
        tensor = self.transform(image)
        target = int(record["target"])
        if not self.include_metadata:
            return tensor, target
        return (
            tensor,
            target,
            str(record.get("level", "none")),
            str(record.get("group_id", "none")),
            str(record["sample_id"]),
            int(record.get("image_id", -1)),
            bool(record.get("touches_border", False)),
        )


def _seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer(model, loader, device) -> dict[str, np.ndarray]:
    torch = require_torch()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    levels: list[str] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    image_ids: list[np.ndarray] = []
    touches_border: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (
            images,
            labels,
            batch_levels,
            batch_groups,
            batch_ids,
            batch_image_ids,
            batch_touches_border,
        ) in loader:
            output = model(images.to(device, non_blocking=True)).float().cpu().numpy()
            logits.append(output)
            targets.append(labels.numpy())
            levels.extend(batch_levels)
            groups.extend(batch_groups)
            sample_ids.extend(batch_ids)
            image_ids.append(batch_image_ids.numpy())
            touches_border.append(batch_touches_border.numpy())
    return {
        "logits": np.concatenate(logits),
        "targets": np.concatenate(targets),
        "levels": np.asarray(levels),
        "groups": np.asarray(groups),
        "sample_ids": np.asarray(sample_ids),
        "image_ids": np.concatenate(image_ids),
        "touches_border": np.concatenate(touches_border).astype(bool),
    }


def _save_predictions(path: Path, predictions: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **predictions)


def _calibration(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    matched = targets >= 0
    if not matched.any():
        raise ValueError("calibration has no matched detector boxes")
    temperature = fit_temperature(logits[matched], targets[matched])
    probabilities = softmax(logits, temperature)
    threshold = select_approval_threshold(probabilities, targets)
    return {
        "temperature": temperature,
        "approval_threshold": threshold.threshold,
        "approved_count": threshold.approved_count,
        "approved_precision": threshold.approved_precision,
        "approval_coverage": threshold.coverage,
        "approved_false_rate_upper_95": threshold.false_approval_rate_upper,
        "risk_control_satisfied": threshold.risk_control_satisfied,
        "accuracy": float((probabilities[matched].argmax(axis=1) == targets[matched]).mean()),
        "top3_accuracy": topk_accuracy(probabilities[matched], targets[matched]),
        "matched_count": int(matched.sum()),
        "unmatched_detector_count": int((~matched).sum()),
    }


def _bootstrap_top1(
    correct: np.ndarray, groups: np.ndarray, repetitions: int, seed: int
) -> list[float]:
    unique, inverse = np.unique(groups, return_inverse=True)
    successes = np.bincount(inverse, weights=correct.astype(np.float64))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    rates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        rates[index] = successes[sampled].sum() / counts[sampled].sum()
    return [float(value) for value in np.quantile(rates, [0.025, 0.975])]


def evaluate_logits(
    predictions: dict[str, np.ndarray],
    calibration: dict[str, Any],
    *,
    category_count: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    logits = predictions["logits"]
    targets = predictions["targets"].astype(np.int64)
    levels = predictions["levels"].astype(str)
    groups = predictions["groups"].astype(str)
    image_ids = predictions.get("image_ids", np.arange(len(targets))).astype(np.int64)
    touches_border = predictions.get("touches_border", np.zeros(len(targets), dtype=bool)).astype(
        bool
    )
    probabilities = softmax(logits, float(calibration["temperature"]))
    predicted = probabilities.argmax(axis=1)
    matched = targets >= 0
    correct = matched & (predicted == targets)
    confidence = probabilities.max(axis=1)
    threshold = float(calibration["approval_threshold"])
    border_low_confidence = touches_border & (confidence < threshold)
    recapture_image_ids = set(image_ids[border_low_confidence].tolist())
    normal = np.asarray([int(image_id) not in recapture_image_ids for image_id in image_ids])
    metric = normal & matched
    if not metric.any():
        raise ValueError("selection contains no matched normal detector boxes")
    # A failed calibration uses a fail-closed threshold sentinel.  Do not rely on
    # floating-point probabilities staying strictly below that sentinel: an
    # extreme logit can produce an exact probability of 1.0.
    approved = (
        normal & (confidence >= threshold)
        if bool(calibration["risk_control_satisfied"])
        else np.zeros(len(targets), dtype=bool)
    )
    approved_count = int(approved.sum())
    approved_correct = int((correct & approved).sum())
    approved_matched = approved & matched
    approved_wrong_matched = int((approved_matched & ~correct).sum())
    approved_unmatched = int((approved & ~matched).sum())
    unknown = normal & ~approved
    class_top1: list[float] = []
    class_top3: list[float] = []
    for target in range(category_count):
        mask = metric & (targets == target)
        if not mask.any():
            raise ValueError(f"evaluation split is missing target {target}")
        class_top1.append(float(correct[mask].mean()))
        class_top3.append(topk_accuracy(probabilities[mask], targets[mask]))
    difficulty: dict[str, Any] = {}
    for level in LEVELS:
        mask = metric & (levels == level)
        level_unknown_matched = unknown & matched & (levels == level)
        if level_unknown_matched.any():
            level_unknown_top = np.argsort(-probabilities[level_unknown_matched], axis=1)[:, :3]
            candidate_in_count = int(
                np.any(level_unknown_top == targets[level_unknown_matched, None], axis=1).sum()
            )
        else:
            candidate_in_count = 0
        level_approved = approved & (levels == level)
        level_approved_count = int(level_approved.sum())
        level_misrecognized = int((level_approved & ~correct).sum())
        difficulty[level] = {
            "sample_count": int(mask.sum()),
            "top1_accuracy": float(correct[mask].mean()) if mask.any() else None,
            "top3_accuracy": topk_accuracy(probabilities[mask], targets[mask])
            if mask.any()
            else None,
            "recognition_rate": float(correct[mask].mean()) if mask.any() else None,
            "candidate_sample_count": int(level_unknown_matched.sum()),
            "candidate_in_count": candidate_in_count,
            "candidate_in_rate": (
                candidate_in_count / int(level_unknown_matched.sum())
                if level_unknown_matched.any()
                else None
            ),
            "candidate_out_count": int(level_unknown_matched.sum()) - candidate_in_count,
            "candidate_out_rate": (
                1.0 - candidate_in_count / int(level_unknown_matched.sum())
                if level_unknown_matched.any()
                else None
            ),
            "approved_count": level_approved_count,
            "misrecognition_count": level_misrecognized,
            "misrecognition_rate": (
                level_misrecognized / level_approved_count if level_approved_count else 0.0
            ),
            "processing_speed_p95_ms": None,
        }
    matched_unknown = unknown & matched
    unknown_top3 = (
        topk_accuracy(probabilities[matched_unknown], targets[matched_unknown])
        if matched_unknown.any()
        else None
    )
    if matched_unknown.any():
        unknown_top = np.argsort(-probabilities[matched_unknown], axis=1)[:, :3]
        unknown_top3_correct_count = int(
            np.any(unknown_top == targets[matched_unknown, None], axis=1).sum()
        )
    else:
        unknown_top3_correct_count = 0
    return {
        "sample_count": int(normal.sum()),
        "matched_sample_count": int(metric.sum()),
        "unmatched_detector_count": int((normal & ~matched).sum()),
        "classifier_border_recapture_images": len(recapture_image_ids),
        "classifier_border_recapture_image_ids": sorted(
            int(value) for value in recapture_image_ids
        ),
        "overall_top1_accuracy": float(correct[metric].mean()),
        "overall_top3_accuracy": topk_accuracy(probabilities[metric], targets[metric]),
        "macro_top1_accuracy": float(np.mean(class_top1)),
        "macro_top3_accuracy": float(np.mean(class_top3)),
        "class_top1_min": float(np.min(class_top1)),
        "class_top1_p05": float(np.quantile(class_top1, 0.05)),
        "class_top3_min": float(np.min(class_top3)),
        "class_top3_p05": float(np.quantile(class_top3, 0.05)),
        "per_class_top1": class_top1,
        "per_class_top3": class_top3,
        "difficulty": difficulty,
        "top1_cluster_bootstrap_95ci": _bootstrap_top1(
            correct[metric], groups[metric], bootstrap_repetitions, bootstrap_seed
        ),
        "temperature": float(calibration["temperature"]),
        "approval_threshold": float(calibration["approval_threshold"]),
        "calibration_risk_control_satisfied": bool(calibration["risk_control_satisfied"]),
        "approved_count": approved_count,
        "approved_matched_count": int(approved_matched.sum()),
        "approval_coverage": approved_count / int(normal.sum()),
        "approved_precision": approved_correct / approved_count if approved_count else 1.0,
        "approved_correct_count": approved_correct,
        "approved_wrong_matched_count": approved_wrong_matched,
        "approved_unmatched_count": approved_unmatched,
        "unknown_count": int(unknown.sum()),
        "unknown_matched_count": int(matched_unknown.sum()),
        "unknown_unmatched_count": int((unknown & ~matched).sum()),
        "unknown_top3_correct_count": unknown_top3_correct_count,
        "unknown_top3_missing_count": int(matched_unknown.sum()) - unknown_top3_correct_count,
        "unknown_top3_accuracy": unknown_top3,
    }


def _ground_truth_worker_outcomes(
    classifier_report: dict[str, Any], detector_report: dict[str, Any], *, role: str | None
) -> dict[str, Any]:
    outcomes = (
        detector_report["validation_image_outcomes"]
        if role is not None
        else detector_report["outcomes"]
    )
    if role is not None:
        outcomes = [row for row in outcomes if row["role"] == role]
    border_ids = {
        int(value) for value in classifier_report["classifier_border_recapture_image_ids"]
    }
    static_recapture = [row for row in outcomes if row["recapture_reasons"]]
    border_recapture = [
        row
        for row in outcomes
        if not row["recapture_reasons"] and int(row["image_id"]) in border_ids
    ]
    unblocked = [
        row
        for row in outcomes
        if not row["recapture_reasons"] and int(row["image_id"]) not in border_ids
    ]
    counts = {
        "approved_correct": int(classifier_report["approved_correct_count"]),
        "approved_misclassification": int(classifier_report["approved_wrong_matched_count"]),
        "unknown_top3_candidate": int(classifier_report["unknown_top3_correct_count"]),
        "unknown_candidate_out": int(classifier_report["unknown_top3_missing_count"]),
        "detector_recapture": sum(int(row["ground_truth_count"]) for row in static_recapture),
        "classifier_border_recapture": sum(
            int(row["ground_truth_count"]) for row in border_recapture
        ),
        "unblocked_detector_missed": sum(int(row["missed_count"]) for row in unblocked),
    }
    denominator = sum(int(row["ground_truth_count"]) for row in outcomes)
    if sum(counts.values()) != denominator:
        raise ValueError(
            "Worker ground-truth outcomes do not partition the detector role: "
            f"outcomes={sum(counts.values())}, denominator={denominator}"
        )
    return {
        "denominator": denominator,
        "counts": counts,
        "rates": {key: value / denominator for key, value in counts.items()},
        "static_recapture_images": len(static_recapture),
        "classifier_border_recapture_images": len(border_recapture),
        "unblocked_images": len(unblocked),
        "unblocked_false_positive_detections": int(classifier_report["unmatched_detector_count"]),
        "approved_unmatched_detections": int(classifier_report["approved_unmatched_count"]),
        "unknown_unmatched_detections": int(classifier_report["unknown_unmatched_count"]),
    }


def _difficulty_worker_metrics(
    classifier_report: dict[str, Any],
    detector_report: dict[str, Any],
    *,
    role: str,
) -> dict[str, dict[str, Any]]:
    """Join classifier KPIs with detector misses using fixed difficulty denominators."""
    outcomes = [row for row in detector_report["validation_image_outcomes"] if row["role"] == role]
    result: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        level_outcomes = [row for row in outcomes if row["level"] == level]
        ground_truth_count = sum(int(row["ground_truth_count"]) for row in level_outcomes)
        missed_count = sum(int(row["missed_count"]) for row in level_outcomes)
        metrics = dict(classifier_report["difficulty"][level])
        metrics.update(
            {
                "ground_truth_count": ground_truth_count,
                "segmentation_failure_count": missed_count,
                "segmentation_failure_rate": (
                    missed_count / ground_truth_count if ground_truth_count else None
                ),
            }
        )
        result[level] = metrics
    return result


def evaluate_worker_taxonomy(
    predictions: dict[str, np.ndarray],
    calibration: dict[str, Any],
    detector_report: dict[str, Any],
    *,
    role: str,
    segment_quality_scores: np.ndarray | None = None,
    segment_quality_threshold: float | None = None,
    force_unknown_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate RPC image/segment outcomes without double-counting recapture misses."""
    logits = predictions["logits"]
    targets = predictions["targets"].astype(np.int64)
    image_ids = predictions["image_ids"].astype(np.int64)
    touches_border = predictions.get("touches_border", np.zeros(len(targets), dtype=bool)).astype(
        bool
    )
    probabilities = softmax(logits, float(calibration["temperature"]))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    if (segment_quality_scores is None) != (segment_quality_threshold is None):
        raise ValueError("segment quality scores and threshold must be provided together")
    if segment_quality_scores is None:
        quality_recapture = np.zeros(len(targets), dtype=bool)
    else:
        quality_scores = np.asarray(segment_quality_scores, dtype=np.float64)
        if quality_scores.shape != targets.shape:
            raise ValueError("segment quality scores must align with predictions")
        quality_recapture = quality_scores < float(segment_quality_threshold)
    if force_unknown_mask is None:
        force_unknown = np.zeros(len(targets), dtype=bool)
    else:
        force_unknown = np.asarray(force_unknown_mask, dtype=bool)
        if force_unknown.shape != targets.shape:
            raise ValueError("force unknown mask must align with predictions")
    approval_enabled = bool(calibration["risk_control_satisfied"])
    threshold = float(calibration["approval_threshold"])
    outcomes = [row for row in detector_report["validation_image_outcomes"] if row["role"] == role]
    result: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        level_outcomes = [row for row in outcomes if row["level"] == level]
        level_image_ids = {int(row["image_id"]) for row in level_outcomes}
        image_recapture_ids = {
            int(row["image_id"]) for row in level_outcomes if row["recapture_reasons"]
        }
        level_detection = np.asarray([int(image_id) in level_image_ids for image_id in image_ids])
        image_normal = np.asarray(
            [int(image_id) not in image_recapture_ids for image_id in image_ids]
        )
        segment_recapture = (
            level_detection
            & image_normal
            & ((touches_border & (confidence < threshold)) | quality_recapture)
        )
        recognition_target = level_detection & image_normal & ~segment_recapture & (targets >= 0)
        approved = (
            level_detection
            & image_normal
            & ~segment_recapture
            & ~force_unknown
            & (confidence >= threshold)
            if approval_enabled
            else np.zeros(len(targets), dtype=bool)
        )
        correct = (targets >= 0) & (predicted == targets)
        correct_approved = approved & correct
        wrong_approved = approved & ~correct
        unknown = recognition_target & ~approved
        if unknown.any():
            unknown_top3 = np.argsort(-probabilities[unknown], axis=1)[:, :3]
            top3_in_count = int(np.any(unknown_top3 == targets[unknown, None], axis=1).sum())
        else:
            top3_in_count = 0
        non_image_recapture = [row for row in level_outcomes if not row["recapture_reasons"]]
        ground_truth_count = sum(int(row["ground_truth_count"]) for row in level_outcomes)
        recognition_count = int(recognition_target.sum())
        approved_count = int(approved.sum())
        detected_non_image_recapture = int((level_detection & image_normal).sum())
        missed_count = sum(int(row["missed_count"]) for row in non_image_recapture)
        non_recapture_gt_count = sum(int(row["ground_truth_count"]) for row in non_image_recapture)
        unmatched_count = int((level_detection & image_normal & (targets < 0)).sum())
        segment_recapture_image_ids = set(
            int(value) for value in image_ids[segment_recapture].tolist()
        )
        approved_unmatched_image_ids = set(
            int(value) for value in image_ids[approved & (targets < 0)].tolist()
        )
        segmentation_failure_image_ids = (
            image_recapture_ids | segment_recapture_image_ids | approved_unmatched_image_ids
        )
        result[level] = {
            "image_count": len(level_outcomes),
            "ground_truth_count": ground_truth_count,
            "recognition_target_count": recognition_count,
            "correct_approved_count": int(correct_approved.sum()),
            "recognition_rate": (
                int(correct_approved.sum()) / recognition_count if recognition_count else None
            ),
            "approved_count": approved_count,
            "wrong_approved_count": int(wrong_approved.sum()),
            "misrecognition_rate": (
                int(wrong_approved.sum()) / approved_count if approved_count else 0.0
            ),
            "unknown_count": int(unknown.sum()),
            "unknown_top3_in_count": top3_in_count,
            "unknown_top3_in_rate": (top3_in_count / int(unknown.sum()) if unknown.any() else None),
            "image_recapture_count": len(image_recapture_ids),
            "image_recapture_rate": (
                len(image_recapture_ids) / len(level_outcomes) if level_outcomes else None
            ),
            "segment_recapture_count": int(segment_recapture.sum()),
            "segment_recapture_image_count": len(segment_recapture_image_ids),
            "segment_recapture_rate": (
                int(segment_recapture.sum()) / detected_non_image_recapture
                if detected_non_image_recapture
                else None
            ),
            "segmentation_missed_count": missed_count,
            "segmentation_missed_rate": (
                missed_count / non_recapture_gt_count if non_recapture_gt_count else None
            ),
            "segmentation_false_positive_count": unmatched_count,
            "segmentation_false_positive_rate": (
                unmatched_count / detected_non_image_recapture
                if detected_non_image_recapture
                else None
            ),
            "approved_unmatched_image_count": len(approved_unmatched_image_ids),
            "segmentation_failure_image_count": len(segmentation_failure_image_ids),
            "segmentation_failure_image_rate": (
                len(segmentation_failure_image_ids) / len(level_outcomes)
                if level_outcomes
                else None
            ),
            "end_to_end_success_rate": (
                int(correct_approved.sum()) / ground_truth_count if ground_truth_count else None
            ),
            "processing_time_mean_ms": None,
            "processing_time_p95_ms": None,
        }
    return result


def _checkpoint(
    model,
    path: Path,
    *,
    stage: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    weights: Path,
    category_count: int,
) -> None:
    torch = require_torch()
    training = config["training"]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_kind": training["backbone_kind"],
            "pretrained_name": None,
            "backbone_architecture": training["backbone_kind"],
            "source_revision": str(training["hub_repository"]).split(":", 1)[-1],
            "source_weight_filename": weights.name,
            "source_weight_sha256": sha256_file(weights),
            "num_classes": category_count,
            "image_size": int(training["image_size"]),
            "classifier_head_kind": str(training.get("classifier_head_kind", "linear")),
            "cosine_scale": float(training.get("cosine_scale", 16.0)),
            "stage": stage,
            "metrics": metrics,
        },
        path,
    )


def _save_stage_progress(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    generator,
    stage: str,
    completed_epochs: int,
    total_epochs: int,
    history: list[dict[str, Any]],
    sample_size: int | None,
    seed: int,
    run_fingerprint: str | None = None,
) -> None:
    torch = require_torch()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "sample_size": sample_size,
        "seed": seed,
        "run_fingerprint": run_fingerprint,
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "history": history,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator": generator.get_state(),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_stage_progress(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    generator,
    stage: str,
    total_epochs: int,
    sample_size: int | None,
    seed: int,
    run_fingerprint: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("stage") == stage
        and int(payload.get("total_epochs", -1)) == total_epochs
        and payload.get("sample_size") == sample_size
        and int(payload.get("seed", -1)) == seed
        and (run_fingerprint is None or payload.get("run_fingerprint") == run_fingerprint)
    )
    if not identity:
        raise ValueError(f"stage progress identity mismatch: {path}")
    completed_epochs = int(payload["completed_epochs"])
    history = list(payload["history"])
    if completed_epochs < 0 or completed_epochs > total_epochs or len(history) != completed_epochs:
        raise ValueError(f"invalid stage progress epoch state: {path}")
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    generator.set_state(rng["loader_generator"])
    return completed_epochs, history


def _run_complete(
    run_dir: Path, expected_selection: int, expected_fingerprint: str | None = None
) -> bool:
    required = (
        run_dir / "complete.json",
        run_dir / "best.pt",
        run_dir / "calibration.json",
        run_dir / "selection_predictions.npz",
        run_dir / "selection_report.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        completion = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
        if completion.get("complete") is not True:
            return False
        if (
            expected_fingerprint is not None
            and completion.get("run_fingerprint") != expected_fingerprint
        ):
            return False
        for filename, expected_hash in completion.get("artifact_sha256", {}).items():
            artifact = run_dir / str(filename)
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                return False
        archive = np.load(run_dir / "selection_predictions.npz")
        return len(archive["targets"]) == expected_selection
    except Exception:
        return False


def _classifier_stage_rank(
    metrics: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Rank classifier stages using calibration data only."""
    risk_controlled = bool(metrics["risk_control_satisfied"])
    if risk_controlled:
        return (
            1.0,
            float(metrics["approval_coverage"]),
            float(metrics["top3_accuracy"]),
            float(metrics["accuracy"]),
        )
    return (
        0.0,
        float(metrics["top3_accuracy"]),
        float(metrics["accuracy"]),
        float(metrics["approval_coverage"]),
    )


def _classifier_training_loss(
    logits,
    labels,
    *,
    hard_negative_loss_weight: float,
    label_smoothing: float = 0.05,
):
    """Combine supervised CE with uniform-target Outlier Exposure."""
    torch = require_torch()
    if hard_negative_loss_weight < 0.0:
        raise ValueError("hard-negative loss weight must be non-negative")
    positive = labels >= 0
    negative = ~positive
    components = []
    positive_loss = None
    negative_loss = None
    if positive.any():
        positive_loss = torch.nn.functional.cross_entropy(
            logits[positive], labels[positive], label_smoothing=label_smoothing
        )
        components.append(positive_loss)
    if negative.any():
        negative_loss = -torch.nn.functional.log_softmax(logits[negative], dim=1).mean()
        components.append(hard_negative_loss_weight * negative_loss)
    if not components:
        raise RuntimeError("classifier batch has no trainable samples")
    return sum(components), positive_loss, negative_loss


def _classifier_domain_split(
    calibration_records: list[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split calibration capture groups into adaptation and risk calibration."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("checkout adaptation group fraction must be in [0, 1)")
    if fraction == 0.0:
        return [], calibration_records
    groups = sorted(
        {str(record["group_id"]) for record in calibration_records},
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    adaptation_count = int(round(len(groups) * fraction))
    if adaptation_count < 1 or adaptation_count >= len(groups):
        raise ValueError("checkout adaptation split leaves an empty group partition")
    adaptation_groups = set(groups[:adaptation_count])
    adaptation = [
        dict(
            record,
            role=(
                "checkout_adaptation" if int(record["target"]) >= 0 else "checkout_hard_negative"
            ),
        )
        for record in calibration_records
        if str(record["group_id"]) in adaptation_groups
    ]
    risk_calibration = [
        record for record in calibration_records if str(record["group_id"]) not in adaptation_groups
    ]
    return adaptation, risk_calibration


def _train_one(
    args: argparse.Namespace,
    config: dict[str, Any],
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    sample_size: int | None,
    seed: int,
) -> None:
    _reject_sealed_mutation(args.output_dir, "classifier training")
    torch = require_torch()
    from torch.utils.data import DataLoader, WeightedRandomSampler

    training = config["training"]
    experiment = config["experiment"]
    full_dataset = _is_full_dataset(config)
    if full_dataset != (sample_size is None):
        raise ValueError("full_dataset runs must not specify a per-class sample size")
    run_dir = (
        args.output_dir / "runs" / "full" / f"seed{seed}"
        if full_dataset
        else args.output_dir / "runs" / f"n{sample_size}" / f"seed{seed}"
    )
    original_calibration_records = [record for record in records if record["role"] == "calibration"]
    checkout_adaptation, calibration_records = _classifier_domain_split(
        original_calibration_records,
        fraction=float(training.get("checkout_adaptation_group_fraction", 0.0)),
        seed=int(training.get("checkout_adaptation_seed", seed)),
    )
    selection_records = [record for record in records if record["role"] == "selection"]
    if full_dataset:
        train_records = [
            record for record in records if record["role"] in {"train", "hard_negative"}
        ] + checkout_adaptation
        expected_train = int(metadata["train_union_count"]) + len(checkout_adaptation)
    else:
        orders = metadata["orders"][str(seed)]
        train_ids = {
            sample_id
            for category_id in sorted(orders, key=int)
            for sample_id in orders[category_id][: int(sample_size)]
        }
        train_records = [record for record in records if record["sample_id"] in train_ids]
        expected_train = int(metadata["category_count"]) * int(sample_size)
    if len(train_records) != expected_train:
        raise ValueError(
            f"run mode={'full' if full_dataset else f'n={sample_size}'}, seed={seed} "
            f"has {len(train_records)} train samples"
        )
    run_fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "mode": "full_dataset" if full_dataset else "data_scale",
                "sample_size": sample_size,
                "seed": seed,
                "train_sample_ids": [
                    str(record["sample_id"])
                    for record in sorted(train_records, key=lambda row: str(row["sample_id"]))
                ],
                "risk_calibration_sample_ids": [
                    str(record["sample_id"])
                    for record in sorted(calibration_records, key=lambda row: str(row["sample_id"]))
                ],
                "source_hashes": metadata["source_hashes"],
                "training": training,
                "weights_sha256": sha256_file(args.weights),
                "warm_start_checkpoint_sha256": (
                    sha256_file(args.output_dir / str(training["warm_start_checkpoint"]))
                    if training.get("warm_start_checkpoint")
                    else None
                ),
            }
        ).encode()
    ).hexdigest()
    if args.resume and _run_complete(run_dir, len(selection_records), run_fingerprint):
        completed_run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if completed_run.get("stage_selection_policy") == CLASSIFIER_STAGE_SELECTION_POLICY:
            print(json.dumps({"skipped_complete_run": str(run_dir)}), flush=True)
            return
        print(
            json.dumps(
                {
                    "migrating_classifier_stage_selection": str(run_dir),
                    "stage_selection_policy": CLASSIFIER_STAGE_SELECTION_POLICY,
                }
            ),
            flush=True,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "run_identity.json"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "mode": "full_dataset" if full_dataset else "data_scale",
        "sample_size_per_class": sample_size,
        "seed": seed,
        "run_fingerprint": run_fingerprint,
    }
    if args.resume and identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"run identity mismatch: {identity_path}")
    else:
        _write_json(identity_path, identity)

    _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("RPC data-scale training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    cache_path = args.output_dir / "prepared" / "cache" / "images.npy"
    train_dataset = RpcCachedDataset(
        train_records,
        cache_path,
        image_size=int(training["image_size"]),
        training=True,
        crop_scale_min=float(training.get("augmentation_crop_scale_min", 0.78)),
        crop_ratio_min=float(training.get("augmentation_crop_ratio_min", 0.9)),
        crop_ratio_max=float(training.get("augmentation_crop_ratio_max", 1.1)),
    )
    calibration_dataset = RpcCachedDataset(
        calibration_records,
        cache_path,
        image_size=int(training["image_size"]),
        training=False,
        include_metadata=True,
    )
    selection_dataset = RpcCachedDataset(
        selection_records,
        cache_path,
        image_size=int(training["image_size"]),
        training=False,
        include_metadata=True,
    )
    generator = torch.Generator().manual_seed(seed)
    sampling_policy = str(training.get("sampling_policy", "shuffle"))
    if sampling_policy == "class_balanced":
        checkout_positive = [
            record
            for record in train_records
            if record["role"] == "checkout_adaptation" and int(record["target"]) >= 0
        ]
        source_positive = [
            record
            for record in train_records
            if record["role"] == "train" and int(record["target"]) >= 0
        ]
        checkout_counts = Counter(int(record["target"]) for record in checkout_positive)
        source_counts = Counter(int(record["target"]) for record in source_positive)
        negative_count = sum(int(record["target"]) < 0 for record in train_records)
        negative_fraction = float(training.get("hard_negative_batch_fraction", 0.25))
        if negative_count and not 0.0 < negative_fraction < 1.0:
            raise ValueError("hard-negative batch fraction must be between zero and one")
        checkout_fraction = float(training.get("checkout_adaptation_batch_fraction", 0.0))
        if checkout_positive and not 0.0 < checkout_fraction < 1.0 - negative_fraction:
            raise ValueError("checkout adaptation batch fraction is invalid")
        source_fraction = 1.0 - negative_fraction - checkout_fraction
        if source_fraction <= 0.0 or not source_counts:
            raise ValueError("source-positive batch fraction is invalid")
        checkout_class_mass = checkout_fraction / len(checkout_counts) if checkout_counts else 0.0
        source_class_mass = source_fraction / len(source_counts)
        negative_mass = negative_fraction / negative_count if negative_count else 0.0
        sample_weights = torch.as_tensor(
            [
                negative_mass
                if int(record["target"]) < 0
                else (
                    checkout_class_mass / checkout_counts[int(record["target"])]
                    if record["role"] == "checkout_adaptation"
                    else source_class_mass / source_counts[int(record["target"])]
                )
                for record in train_records
            ],
            dtype=torch.double,
        )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_records),
            replacement=True,
            generator=generator,
        )
        train_shuffle = False
    elif sampling_policy == "shuffle":
        train_sampler = None
        train_shuffle = True
    else:
        raise ValueError(f"unsupported classifier sampling policy: {sampling_policy}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=train_shuffle,
        sampler=train_sampler,
        generator=generator,
        num_workers=int(training["workers"]),
        pin_memory=True,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        pin_memory=True,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        pin_memory=True,
    )
    model = build_dino_classifier(
        str(training["backbone_kind"]),
        int(metadata["category_count"]),
        weights_path=args.weights,
        hub_repository=str(training["hub_repository"]),
        classifier_head_kind=str(training.get("classifier_head_kind", "linear")),
        cosine_scale=float(training.get("cosine_scale", 16.0)),
    ).to(device)
    warm_start_path = (
        args.output_dir / str(training["warm_start_checkpoint"])
        if training.get("warm_start_checkpoint")
        else None
    )
    if warm_start_path is not None:
        checkpoint = torch.load(warm_start_path, map_location=device, weights_only=False)
        if int(checkpoint.get("num_classes", -1)) != int(metadata["category_count"]):
            raise ValueError("warm-start classifier category count mismatch")
        if checkpoint.get("classifier_head_kind", "linear") != str(
            training.get("classifier_head_kind", "linear")
        ):
            raise ValueError("warm-start classifier head mismatch")
        model.load_state_dict(checkpoint["state_dict"])
    started = time.perf_counter()
    histories: dict[str, list[dict[str, Any]]] = {}
    stage_results: dict[str, dict[str, Any]] = {}

    def run_stage(
        name: str,
        epochs: int,
        learning_rate: float,
        *,
        unfreeze_blocks: int = 0,
        unfreeze_stages: int = 0,
        unfreeze_all: bool = False,
    ) -> Path:
        checkpoint_path = run_dir / f"{name}.pt"
        calibration_path = run_dir / f"{name}_calibration.json"
        predictions_path = run_dir / f"{name}_calibration_predictions.npz"
        history_path = run_dir / f"{name}_history.json"
        progress_path = run_dir / f"{name}_progress.pt"
        final_artifacts = (checkpoint_path, calibration_path, predictions_path, history_path)
        if args.resume and all(path.is_file() for path in final_artifacts):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint.get("stage") != name:
                raise ValueError(f"checkpoint stage mismatch: {checkpoint_path}")
            model.load_state_dict(checkpoint["state_dict"])
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if len(history) != epochs:
                raise ValueError(f"completed stage history has the wrong length: {history_path}")
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            histories[name] = history
            stage_results[name] = calibration
            progress_path.unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        "run": "full" if full_dataset else f"n{sample_size}",
                        "seed": seed,
                        "resumed_complete_stage": name,
                    }
                ),
                flush=True,
            )
            return checkpoint_path
        if not args.resume:
            progress_path.unlink(missing_ok=True)
        set_frozen_backbone(
            model,
            unfreeze_last_blocks=unfreeze_blocks,
            unfreeze_last_stages=unfreeze_stages,
            unfreeze_all=unfreeze_all,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        history: list[dict[str, Any]] = []
        start_epoch = 0
        if args.resume and progress_path.is_file():
            start_epoch, history = _load_stage_progress(
                progress_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                stage=name,
                total_epochs=epochs,
                sample_size=sample_size,
                seed=seed,
                run_fingerprint=run_fingerprint,
            )
            print(
                json.dumps(
                    {
                        "run": "full" if full_dataset else f"n{sample_size}",
                        "seed": seed,
                        "resumed_stage": name,
                        "completed_epochs": start_epoch,
                    }
                ),
                flush=True,
            )
        for epoch in range(start_epoch, epochs):
            model.train()
            losses: list[float] = []
            positive_losses: list[float] = []
            negative_losses: list[float] = []
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(images)
                    loss, positive_loss, negative_loss = _classifier_training_loss(
                        output,
                        labels,
                        hard_negative_loss_weight=float(
                            training.get("hard_negative_loss_weight", 0.5)
                        ),
                        label_smoothing=float(training.get("label_smoothing", 0.05)),
                    )
                    if positive_loss is not None:
                        positive_losses.append(float(positive_loss.detach().cpu()))
                    if negative_loss is not None:
                        negative_losses.append(float(negative_loss.detach().cpu()))
                losses.append(float(loss.detach().cpu()))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            row = {
                "stage": name,
                "epoch": epoch + 1,
                "training_loss": float(np.mean(losses)),
                "positive_loss": (float(np.mean(positive_losses)) if positive_losses else None),
                "hard_negative_loss": (
                    float(np.mean(negative_losses)) if negative_losses else None
                ),
            }
            history.append(row)
            print(
                json.dumps(
                    {"run": "full" if full_dataset else f"n{sample_size}", "seed": seed, **row}
                ),
                flush=True,
            )
            _save_stage_progress(
                progress_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                stage=name,
                completed_epochs=epoch + 1,
                total_epochs=epochs,
                history=history,
                sample_size=sample_size,
                seed=seed,
                run_fingerprint=run_fingerprint,
            )
        histories[name] = history
        calibration_predictions = _infer(model, calibration_loader, device)
        stage_calibration = _calibration(
            calibration_predictions["logits"], calibration_predictions["targets"]
        )
        stage_results[name] = stage_calibration
        _checkpoint(
            model,
            checkpoint_path,
            stage=name,
            metrics=stage_calibration,
            config=config,
            weights=args.weights,
            category_count=int(metadata["category_count"]),
        )
        _save_predictions(predictions_path, calibration_predictions)
        _write_json(history_path, history)
        _write_json(calibration_path, stage_calibration)
        progress_path.unlink(missing_ok=True)
        return checkpoint_path

    frozen_path = run_stage(
        "frozen",
        int(training["frozen_epochs"]),
        float(training["frozen_lr"]),
    )
    frozen_state = torch.load(frozen_path, map_location=device, weights_only=False)["state_dict"]
    partial_path = run_stage(
        "partial",
        int(training["finetune_epochs"]),
        float(training["finetune_lr"]),
        unfreeze_all=bool(training.get("finetune_unfreeze_all_backbone", False)),
        unfreeze_stages=int(training.get("finetune_unfreeze_last_stages", 0)),
        unfreeze_blocks=(
            0
            if bool(training.get("finetune_unfreeze_all_backbone", False))
            or int(training.get("finetune_unfreeze_last_stages", 0))
            else int(training.get("finetune_unfreeze_last_blocks", 2))
        ),
    )
    frozen_metrics = stage_results["frozen"]
    partial_metrics = stage_results["partial"]
    use_partial = _classifier_stage_rank(partial_metrics) > _classifier_stage_rank(frozen_metrics)
    selected_stage = "partial" if use_partial else "frozen"
    selected_path = partial_path if use_partial else frozen_path
    if not use_partial:
        model.load_state_dict(frozen_state)
    shutil.copyfile(selected_path, run_dir / "best.pt")
    shutil.copyfile(run_dir / f"{selected_stage}_calibration.json", run_dir / "calibration.json")
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    selection_predictions = _infer(model, selection_loader, device)
    _save_predictions(run_dir / "selection_predictions.npz", selection_predictions)
    selection_report = evaluate_logits(
        selection_predictions,
        calibration,
        category_count=int(metadata["category_count"]),
        bootstrap_repetitions=int(experiment["bootstrap_repetitions"]),
        bootstrap_seed=seed if sample_size is None else seed + sample_size,
    )
    detector_report = json.loads(
        (args.output_dir / "prepared" / "worker_gate_report.json").read_text(encoding="utf-8")
    )
    selection_report["all_ground_truth_box_outcomes"] = _ground_truth_worker_outcomes(
        selection_report, detector_report, role="selection"
    )
    selection_report["difficulty_worker_metrics"] = _difficulty_worker_metrics(
        selection_report, detector_report, role="selection"
    )
    selection_report["worker_taxonomy"] = evaluate_worker_taxonomy(
        selection_predictions,
        calibration,
        detector_report,
        role="selection",
    )
    _write_json(run_dir / "selection_report.json", selection_report)
    elapsed = time.perf_counter() - started
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "mode": "full_dataset" if full_dataset else "data_scale",
        "sample_size_per_class": sample_size,
        "seed": seed,
        "train_sample_count": len(train_records),
        "positive_train_sample_count": sum(int(record["target"]) >= 0 for record in train_records),
        "hard_negative_train_sample_count": sum(
            int(record["target"]) < 0 for record in train_records
        ),
        "checkout_adaptation_sample_count": len(checkout_adaptation),
        "calibration_sample_count": len(calibration_records),
        "selection_sample_count": len(selection_records),
        "selected_stage": selected_stage,
        "stage_selection_policy": CLASSIFIER_STAGE_SELECTION_POLICY,
        "elapsed_seconds": elapsed,
        "weights_sha256": sha256_file(args.weights),
        "source_hashes": metadata["source_hashes"],
        "training": training,
        "run_fingerprint": run_fingerprint,
    }
    _write_json(run_dir / "run.json", run_record)
    artifact_names = (
        "best.pt",
        "calibration.json",
        "selection_predictions.npz",
        "selection_report.json",
        "run.json",
    )
    _write_json(
        run_dir / "complete.json",
        {
            "complete": True,
            "completed_at": datetime.now(UTC).isoformat(),
            "run_fingerprint": run_fingerprint,
            "artifact_sha256": {
                filename: sha256_file(run_dir / filename) for filename in artifact_names
            },
        },
    )
    torch.cuda.empty_cache()


def train_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    _reject_sealed_mutation(args.output_dir, "train")
    metadata_path = args.output_dir / "prepared" / "experiment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("prepare phase has not completed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = _read_jsonl(args.output_dir / "prepared" / "cache" / "records.jsonl")
    if _is_full_dataset(config):
        seed = int(config["experiment"]["seeds"][0])
        _train_one(args, config, metadata, records, None, seed)
        return
    for sample_size in config["experiment"]["sample_sizes"]:
        for seed in config["experiment"]["seeds"]:
            _train_one(args, config, metadata, records, int(sample_size), int(seed))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _mean_std_optional(values: list[float | None]) -> dict[str, float] | None:
    present = [float(value) for value in values if value is not None]
    return _mean_std(present) if present else None


def _condition_bootstrap_top1(
    prediction_sets: list[dict[str, np.ndarray]], repetitions: int, seed: int
) -> list[float]:
    grouped_runs: list[tuple[np.ndarray, np.ndarray]] = []
    for predictions in prediction_sets:
        logits = predictions["logits"]
        targets = predictions["targets"].astype(np.int64)
        groups = predictions["groups"].astype(str)
        matched = targets >= 0
        targets = targets[matched]
        groups = groups[matched]
        correct = logits[matched].argmax(axis=1) == targets
        _, inverse = np.unique(groups, return_inverse=True)
        grouped_runs.append(
            (
                np.bincount(inverse, weights=correct.astype(np.float64)),
                np.bincount(inverse),
            )
        )
    rng = np.random.default_rng(seed)
    rates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled_run_indices = rng.integers(0, len(grouped_runs), size=len(grouped_runs))
        run_rates: list[float] = []
        for run_index in sampled_run_indices:
            successes, counts = grouped_runs[int(run_index)]
            sampled_groups = rng.integers(0, len(counts), size=len(counts))
            run_rates.append(float(successes[sampled_groups].sum() / counts[sampled_groups].sum()))
        rates[repetition] = float(np.mean(run_rates))
    return [float(value) for value in np.quantile(rates, [0.025, 0.975])]


def _operational_gate(calibration: dict[str, Any], report: dict[str, Any]) -> bool:
    unknown_top3 = report["unknown_top3_accuracy"]
    return (
        bool(calibration["risk_control_satisfied"])
        and float(report["approved_precision"]) >= 0.995
        and unknown_top3 is not None
        and float(unknown_top3) >= 0.95
    )


def _test_operational_gate(calibration: dict[str, Any], report: dict[str, Any]) -> bool:
    """Require both decision KPIs to have evaluable final-test samples."""
    return (
        _operational_gate(calibration, report)
        and int(report.get("approved_count", 0)) > 0
        and int(report.get("unknown_matched_count", 0)) > 0
    )


def _post_test_locked_full_summary(
    args: argparse.Namespace,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return the immutable selection after exact post-test seal verification."""
    lock_path = args.output_dir / "model_lock.json"
    summary_path = args.output_dir / "reports" / "selection_summary.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        lock.get("mode") != "full_dataset"
        or lock.get("status") != "validation_passed"
        or lock.get("operational_gate") is not True
        or summary.get("status") != "validation_passed"
        or summary.get("operational_gate") is not True
        or lock.get("selection_summary_sha256") != sha256_file(summary_path)
        or metadata.get("test_access_model_lock_sha256") != sha256_file(lock_path)
    ):
        raise ValueError("post-test classifier selection/model lock seal is invalid")
    model_run = lock.get("model_run")
    if not isinstance(model_run, str) or not model_run:
        raise ValueError("post-test model lock has no classifier run")
    run_dir = args.output_dir / model_run
    for filename, field in (
        ("best.pt", "checkpoint_sha256"),
        ("calibration.json", "calibration_sha256"),
        ("selection_report.json", "selection_report_sha256"),
    ):
        path = run_dir / filename
        if not path.is_file() or lock.get(field) != sha256_file(path):
            raise ValueError(f"post-test model lock checksum mismatch: {filename}")
    _checkpoint, final_complete = _final_detector_artifacts(args, config)
    final_complete_path = args.output_dir / "detector" / "final" / "complete.json"
    expected_detector_lock = {
        "rpc_config_sha256": final_complete["config_sha256"],
        "active_detector_complete_sha256": final_complete["active_detector_complete_sha256"],
        "active_detector_threshold_sha256": final_complete["active_threshold_sha256"],
        "detector_train_gate_complete_sha256": final_complete["train_gate_complete_sha256"],
        "final_detector_complete_sha256": sha256_file(final_complete_path),
        "final_detector_checkpoint_sha256": final_complete["stage_a_checkpoint_sha256"],
        "operational_detector_role": "checkout_baseline_val_all_operational",
        "train_gate_role": "offline_roi_train_gate_only",
    }

    if (
        any(lock.get(key) != value for key, value in expected_detector_lock.items())
        or metadata.get("test_access_final_detector_complete_sha256")
        != expected_detector_lock["final_detector_complete_sha256"]
        or metadata.get("test_access_final_detector_checkpoint_sha256")
        != expected_detector_lock["final_detector_checkpoint_sha256"]
    ):
        raise ValueError("post-test detector/model lock seal is invalid")
    return summary


def _summarize_full_dataset(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    seed = int(experiment["seeds"][0])
    metadata = json.loads(
        (args.output_dir / "prepared" / "experiment.json").read_text(encoding="utf-8")
    )
    if metadata.get("test_accessed") is True:
        return _post_test_locked_full_summary(args, config, metadata)
    run_dir = args.output_dir / "runs" / "full" / f"seed{seed}"
    predictions_path = run_dir / "selection_predictions.npz"
    expected_selection = (
        len(np.load(predictions_path)["targets"]) if predictions_path.is_file() else -1
    )
    if not _run_complete(run_dir, expected_selection):
        raise RuntimeError(f"incomplete run: {run_dir}")
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "selection_report.json").read_text(encoding="utf-8"))
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    worker_gate = json.loads(
        (args.output_dir / "prepared" / "worker_gate_report.json").read_text(encoding="utf-8")
    )
    operational = _operational_gate(calibration, report)
    validation_failures: list[str] = []
    if not bool(calibration["risk_control_satisfied"]):
        validation_failures.append("CALIBRATION_APPROVAL_RISK_CONTROL")
    if float(report["approved_precision"]) < 0.995:
        validation_failures.append("SELECTION_APPROVED_PRECISION")
    if report["unknown_top3_accuracy"] is None:
        validation_failures.append("SELECTION_UNKNOWN_TOP3_NOT_EVALUABLE")
    elif float(report["unknown_top3_accuracy"]) < 0.95:
        validation_failures.append("SELECTION_UNKNOWN_TOP3_ACCURACY")
    final_detector_complete: dict[str, Any] | None = None
    final_detector_complete_path = args.output_dir / "detector" / "final" / "complete.json"
    if operational:
        # Model selection is the last training-authorized phase. Complete the
        # checkout-only final detector now so test remains strictly read-only.
        train_final_detector(args, config, resume=bool(getattr(args, "resume", False)))
        final_detector_complete = json.loads(
            final_detector_complete_path.read_text(encoding="utf-8")
        )
    metrics = {
        "overall_top1_accuracy": float(report["overall_top1_accuracy"]),
        "overall_top3_accuracy": float(report["overall_top3_accuracy"]),
        "macro_top1_accuracy": float(report["macro_top1_accuracy"]),
        "macro_top3_accuracy": float(report["macro_top3_accuracy"]),
        "class_top1_min": float(report["class_top1_min"]),
        "class_top1_p05": float(report["class_top1_p05"]),
        "approved_precision": float(report["approved_precision"]),
        "approval_coverage": float(report["approval_coverage"]),
        "unknown_top3_accuracy": report["unknown_top3_accuracy"],
        "top1_cluster_bootstrap_95ci": report["top1_cluster_bootstrap_95ci"],
        "difficulty": report["difficulty"],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "full_dataset",
        "status": "validation_passed" if operational else "validation_gate_failed",
        "model_run": f"runs/full/seed{seed}",
        "seed": seed,
        "train_sample_count": int(run_record["train_sample_count"]),
        "train_counts": metadata["train_counts"],
        "train_class_imbalance": metadata["train_class_imbalance"],
        "operational_gate": operational,
        "validation_failure_reasons": validation_failures,
        "test_accessed": False,
        "calibration_risk_control": bool(calibration["risk_control_satisfied"]),
        "calibration_metrics": {
            key: calibration[key]
            for key in (
                "temperature",
                "approval_threshold",
                "approved_count",
                "approved_precision",
                "approval_coverage",
                "approved_false_rate_upper_95",
                "risk_control_satisfied",
                "accuracy",
                "top3_accuracy",
                "matched_count",
                "unmatched_detector_count",
            )
        },
        "validation_metrics": metrics,
        "detector_worker_gate": {
            key: worker_gate[key]
            for key in (
                "score_threshold",
                "train_candidates",
                "train_rejected",
                "validation_images",
                "validation_normal_images",
                "validation_recapture_images",
                "validation_recapture_reasons",
                "validation_missed_boxes",
                "validation_unmatched_boxes",
            )
        },
    }
    if final_detector_complete is not None:
        summary["final_detector"] = {
            "contract": final_detector_complete["contract"],
            "base_epochs": final_detector_complete["base_epochs"],
            "checkpoint_sha256": final_detector_complete["stage_a_checkpoint_sha256"],
            "complete_sha256": sha256_file(final_detector_complete_path),
        }
    reports_dir = args.output_dir / "reports"
    summary_path = reports_dir / "selection_summary.json"
    _write_json(summary_path, summary)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "selection_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        row = {
            "mode": "full_dataset",
            "seed": seed,
            "train_sample_count": summary["train_sample_count"],
            "operational_gate": operational,
            **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))},
        }
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    unknown = metrics["unknown_top3_accuracy"]
    unknown_display = "N/A" if unknown is None else f"{float(unknown):.4%}"
    lines = [
        "# RPC Classifier 전체 데이터셋 검증",
        "",
        f"- 결론: `{summary['status']}`",
        f"- 실패 사유: `{', '.join(validation_failures) if validation_failures else '없음'}`",
        "- test2019 접근: `false`",
        f"- 학습 ROI 수: `{summary['train_sample_count']}`",
        f"- Seed: `{seed}`",
        f"- Top-1: `{metrics['overall_top1_accuracy']:.4%}`",
        f"- Top-3: `{metrics['overall_top3_accuracy']:.4%}`",
        f"- 승인 수 / coverage: `{calibration['approved_count']} / {calibration['approval_coverage']:.4%}`",
        f"- 승인 false-rate 단측 95% 상한: `{calibration['approved_false_rate_upper_95']:.4%}`",
        f"- calibration risk-control: `{calibration['risk_control_satisfied']}`",
        f"- UNKNOWN Top-3: `{unknown_display}`",
    ]
    (reports_dir / "selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "mode": "full_dataset",
        "status": summary["status"],
        "model_run": summary["model_run"],
        "seed": seed,
        "operational_gate": operational,
        "locked_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "calibration_sha256": sha256_file(run_dir / "calibration.json"),
        "selection_report_sha256": sha256_file(run_dir / "selection_report.json"),
        "selection_summary_sha256": sha256_file(summary_path),
    }
    if final_detector_complete is not None:
        lock.update(
            {
                "rpc_config_sha256": final_detector_complete["config_sha256"],
                "active_detector_complete_sha256": final_detector_complete[
                    "active_detector_complete_sha256"
                ],
                "active_detector_threshold_sha256": final_detector_complete[
                    "active_threshold_sha256"
                ],
                "detector_train_gate_complete_sha256": final_detector_complete[
                    "train_gate_complete_sha256"
                ],
                "final_detector_complete_sha256": sha256_file(final_detector_complete_path),
                "final_detector_checkpoint_sha256": final_detector_complete[
                    "stage_a_checkpoint_sha256"
                ],
                "operational_detector_role": "checkout_baseline_val_all_operational",
                "train_gate_role": "offline_roi_train_gate_only",
            }
        )
    _write_json(args.output_dir / "model_lock.json", lock)
    return summary


def summarize(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if _is_full_dataset(config):
        return _summarize_full_dataset(args, config)
    experiment = config["experiment"]
    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    grouped_predictions: dict[int, list[dict[str, np.ndarray]]] = defaultdict(list)
    for sample_size in experiment["sample_sizes"]:
        for seed in experiment["seeds"]:
            run_dir = args.output_dir / "runs" / f"n{sample_size}" / f"seed{seed}"
            if not _run_complete(
                run_dir,
                len(np.load(run_dir / "selection_predictions.npz")["targets"])
                if (run_dir / "selection_predictions.npz").is_file()
                else -1,
            ):
                raise RuntimeError(f"incomplete run: {run_dir}")
            calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "selection_report.json").read_text(encoding="utf-8"))
            predictions_archive = np.load(run_dir / "selection_predictions.npz")
            predictions = {key: predictions_archive[key] for key in predictions_archive.files}
            per_class_top3 = [float(value) for value in report["per_class_top3"]]
            row = {
                "sample_size": int(sample_size),
                "seed": int(seed),
                "operational_gate": _operational_gate(calibration, report),
                "calibration_risk_control": bool(calibration["risk_control_satisfied"]),
                "top1": float(report["overall_top1_accuracy"]),
                "top3": float(report["overall_top3_accuracy"]),
                "macro_top1": float(report["macro_top1_accuracy"]),
                "macro_top3": float(report["macro_top3_accuracy"]),
                "class_top1_min": float(report["class_top1_min"]),
                "class_top1_p05": float(report["class_top1_p05"]),
                "class_top3_min": float(report.get("class_top3_min", min(per_class_top3))),
                "class_top3_p05": float(
                    report.get("class_top3_p05", np.quantile(per_class_top3, 0.05))
                ),
                "easy_top1": float(report["difficulty"]["easy"]["top1_accuracy"]),
                "easy_top3": float(report["difficulty"]["easy"]["top3_accuracy"]),
                "medium_top1": float(report["difficulty"]["medium"]["top1_accuracy"]),
                "medium_top3": float(report["difficulty"]["medium"]["top3_accuracy"]),
                "hard_top1": float(report["difficulty"]["hard"]["top1_accuracy"]),
                "hard_top3": float(report["difficulty"]["hard"]["top3_accuracy"]),
                "top1_bootstrap_low": float(report["top1_cluster_bootstrap_95ci"][0]),
                "top1_bootstrap_high": float(report["top1_cluster_bootstrap_95ci"][1]),
                "approved_precision": float(report["approved_precision"]),
                "approval_coverage": float(report["approval_coverage"]),
                "unknown_top3": report["unknown_top3_accuracy"],
                "classifier_border_recapture_images": int(
                    report.get("classifier_border_recapture_images", 0)
                ),
            }
            rows.append(row)
            grouped[int(sample_size)].append(row)
            grouped_predictions[int(sample_size)].append(predictions)
    baseline_size = max(grouped)
    baseline_top1 = float(np.mean([row["top1"] for row in grouped[baseline_size]]))
    margin = float(experiment["noninferiority_margin"])
    conditions: list[dict[str, Any]] = []
    selected_n: int | None = None
    for sample_size in sorted(grouped):
        run_rows = grouped[sample_size]
        top1 = _mean_std([row["top1"] for row in run_rows])
        operational = all(bool(row["operational_gate"]) for row in run_rows)
        noninferior = top1["mean"] >= baseline_top1 - margin
        eligible = operational and noninferior
        condition = {
            "sample_size": sample_size,
            "total_train_images": sample_size * 200,
            "top1": top1,
            "top3": _mean_std([row["top3"] for row in run_rows]),
            "macro_top1": _mean_std([row["macro_top1"] for row in run_rows]),
            "macro_top3": _mean_std([row["macro_top3"] for row in run_rows]),
            "class_top1_min": _mean_std([row["class_top1_min"] for row in run_rows]),
            "class_top1_p05": _mean_std([row["class_top1_p05"] for row in run_rows]),
            "class_top3_min": _mean_std([row["class_top3_min"] for row in run_rows]),
            "class_top3_p05": _mean_std([row["class_top3_p05"] for row in run_rows]),
            "difficulty": {
                level: {
                    "top1": _mean_std([row[f"{level}_top1"] for row in run_rows]),
                    "top3": _mean_std([row[f"{level}_top3"] for row in run_rows]),
                }
                for level in LEVELS
            },
            "approved_precision": _mean_std([row["approved_precision"] for row in run_rows]),
            "approval_coverage": _mean_std([row["approval_coverage"] for row in run_rows]),
            "unknown_top3": _mean_std_optional([row["unknown_top3"] for row in run_rows]),
            "classifier_border_recapture_images": _mean_std(
                [float(row["classifier_border_recapture_images"]) for row in run_rows]
            ),
            "top1_hierarchical_bootstrap_95ci": _condition_bootstrap_top1(
                grouped_predictions[sample_size],
                int(experiment["bootstrap_repetitions"]),
                int(experiment["validation_split_seed"]) + sample_size,
            ),
            "all_seeds_operational_gate": operational,
            "within_baseline_top1_margin": noninferior,
            "eligible": eligible,
        }
        conditions.append(condition)
        if eligible and selected_n is None:
            selected_n = sample_size
    status = "selected" if selected_n is not None else f"insufficient_up_to_{baseline_size}"
    worker_gate = json.loads(
        (args.output_dir / "prepared" / "worker_gate_report.json").read_text(encoding="utf-8")
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "selected_n": selected_n,
        "baseline_n": baseline_size,
        "baseline_top1_mean": baseline_top1,
        "noninferiority_margin": margin,
        "detector_worker_gate": {
            key: worker_gate[key]
            for key in (
                "score_threshold",
                "train_candidates",
                "train_rejected",
                "validation_images",
                "validation_normal_images",
                "validation_recapture_images",
                "validation_recapture_reasons",
                "validation_missed_boxes",
                "validation_unmatched_boxes",
            )
        },
        "conditions": conditions,
        "runs": rows,
    }
    reports_dir = args.output_dir / "reports"
    _write_json(reports_dir / "selection_summary.json", summary)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "selection_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# RPC Classifier 데이터 규모 실험",
        "",
        f"- 결론: `{status}`",
        f"- 선택된 클래스당 이미지 수: `{selected_n}`",
        f"- {baseline_size}장 기준 평균 Top-1: `{baseline_top1:.6f}`",
        "- 선택 조건이 없으면 test2019는 열지 않고 최종 test를 생략합니다.",
        "",
        f"| N/class | 총 학습 이미지 | Top-1 mean±std | Top-3 mean | 승인 precision | UNKNOWN Top-3 | 운영 gate | {baseline_size}장 대비 1%p | 선택 가능 |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for condition in conditions:
        unknown_top3 = condition["unknown_top3"]
        unknown_display = "N/A" if unknown_top3 is None else f"{unknown_top3['mean']:.4%}"
        lines.append(
            f"| {condition['sample_size']} | {condition['total_train_images']} | "
            f"{condition['top1']['mean']:.4%} ± {condition['top1']['std']:.4%} | "
            f"{condition['top3']['mean']:.4%} | "
            f"{condition['approved_precision']['mean']:.4%} | {unknown_display} | "
            f"{'Y' if condition['all_seeds_operational_gate'] else 'N'} | "
            f"{'Y' if condition['within_baseline_top1_margin'] else 'N'} | "
            f"{'Y' if condition['eligible'] else 'N'} |"
        )
    (reports_dir / "selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "selected_n": selected_n,
        "status": status,
        "locked_at": datetime.now(UTC).isoformat(),
        "selection_summary_sha256": sha256_file(reports_dir / "selection_summary.json"),
    }
    _write_json(args.output_dir / "selected_n.json", lock)
    return summary


def _load_checkpoint_model(checkpoint_path: Path, config: dict[str, Any], device):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_dino_classifier(
        checkpoint["backbone_kind"],
        int(checkpoint["num_classes"]),
        hub_repository=str(config["training"]["hub_repository"]),
        classifier_head_kind=checkpoint.get("classifier_head_kind", "linear"),
        cosine_scale=float(checkpoint.get("cosine_scale", 16.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _post_test_locked_final_result(
    args: argparse.Namespace,
    config: dict[str, Any],
    metadata: dict[str, Any],
    lock_path: Path,
) -> dict[str, Any]:
    """Verify and return an already completed final test without data access."""
    if not _is_full_dataset(config):
        raise RuntimeError("sealed data-scale test requires a fresh output directory")
    _post_test_locked_full_summary(args, config, metadata)
    final_path = args.output_dir / "reports" / "final_test.json"
    detector_report_path = args.output_dir / "test" / "detector_report.json"
    if metadata.get("final_test_report_sha256") != sha256_file(final_path) or metadata.get(
        "final_test_detector_report_sha256"
    ) != sha256_file(detector_report_path):
        raise ValueError("post-test result seal is missing or invalid")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    detector_report = json.loads(detector_report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_final = {
        "model_run": lock["model_run"],
        "classifier_checkpoint_sha256": lock["checkpoint_sha256"],
        "model_lock_sha256": sha256_file(lock_path),
        "detector_report_sha256": sha256_file(detector_report_path),
        "detector_checkpoint_sha256": lock["final_detector_checkpoint_sha256"],
    }
    expected_detector = {
        "model_lock_sha256": sha256_file(lock_path),
        "detector_checkpoint_sha256": lock["final_detector_checkpoint_sha256"],
        "final_detector_complete_sha256": lock["final_detector_complete_sha256"],
        "active_detector_threshold_sha256": lock["active_detector_threshold_sha256"],
    }
    if any(final.get(key) != value for key, value in expected_final.items()) or any(
        detector_report.get(key) != value for key, value in expected_detector.items()
    ):
        raise ValueError("post-test result is not bound to the immutable model lock")
    return final


def test_selected(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    full_dataset = _is_full_dataset(config)
    lock_path = args.output_dir / ("model_lock.json" if full_dataset else "selected_n.json")
    if not lock_path.is_file():
        raise FileNotFoundError(
            "select phase has not locked the full model"
            if full_dataset
            else "select phase has not locked a sample size"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected_n = None if full_dataset else lock.get("selected_n")
    if not full_dataset and selected_n is None:
        print(json.dumps({"test_skipped": "no eligible N"}), flush=True)
        return None
    if full_dataset:
        if (
            lock.get("mode") != "full_dataset"
            or lock.get("status") != "validation_passed"
            or lock.get("operational_gate") is not True
        ):
            raise RuntimeError("full_dataset validation/model lock gate has not passed")
        locked_run_dir = args.output_dir / str(lock["model_run"])
        for filename, key in (
            ("best.pt", "checkpoint_sha256"),
            ("calibration.json", "calibration_sha256"),
            ("selection_report.json", "selection_report_sha256"),
        ):
            if sha256_file(locked_run_dir / filename) != lock[key]:
                raise ValueError(f"model lock checksum mismatch: {filename}")
        summary_path = args.output_dir / "reports" / "selection_summary.json"
        if not summary_path.is_file() or sha256_file(summary_path) != lock.get(
            "selection_summary_sha256"
        ):
            raise ValueError("model lock checksum mismatch: selection_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "validation_passed"
            or summary.get("operational_gate") is not True
        ):
            raise RuntimeError("full_dataset validation/model lock gate has not passed")
    metadata_path = args.output_dir / "prepared" / "experiment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("test_accessed") is True:
        return _post_test_locked_final_result(args, config, metadata, lock_path)

    def seal_test_access() -> None:
        if not metadata.get("test_accessed", False):
            metadata["test_accessed"] = True
            metadata["test_access_started_at"] = datetime.now(UTC).isoformat()
            metadata["test_access_model_lock_sha256"] = sha256_file(lock_path)
            metadata["test_access_final_detector_complete_sha256"] = lock.get(
                "final_detector_complete_sha256"
            )
            metadata["test_access_final_detector_checkpoint_sha256"] = lock.get(
                "final_detector_checkpoint_sha256"
            )
            _write_experiment_metadata(metadata_path, metadata, durable=True)

    test_records, detector_test_report = prepare_final_test_records(
        args,
        config,
        resume=args.resume,
        model_lock_path=lock_path,
        before_test_access=seal_test_access,
    )
    test_metadata = {
        "schema_version": SCHEMA_VERSION,
        "instances_test2019_sha256": detector_test_report["test_annotation_sha256"],
        "detector_checkpoint_sha256": detector_test_report["detector_checkpoint_sha256"],
        "selection_lock_sha256": sha256_file(lock_path),
    }
    if full_dataset:
        test_metadata["mode"] = "full_dataset"
        test_metadata["model_run"] = str(lock["model_run"])
    else:
        test_metadata["selected_n"] = int(selected_n)
    test_records = _build_cache(
        args.dataset_root,
        args.output_dir / "test_cache",
        sorted(test_records, key=lambda row: row["sample_id"]),
        test_metadata,
        config["training"],
        resume=args.resume,
    )
    metadata["test_source_hash"] = test_metadata["instances_test2019_sha256"]
    _write_experiment_metadata(metadata_path, metadata, durable=True)
    torch = require_torch()
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("RPC final test requires CUDA")
    device = torch.device("cuda")
    dataset = RpcCachedDataset(
        test_records,
        args.output_dir / "test_cache" / "images.npy",
        image_size=int(config["training"]["image_size"]),
        training=False,
        include_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
    )
    seed_reports: list[dict[str, Any]] = []
    seeds = [int(lock["seed"])] if full_dataset else config["experiment"]["seeds"]
    for seed in seeds:
        run_dir = (
            args.output_dir / str(lock["model_run"])
            if full_dataset
            else args.output_dir / "runs" / f"n{selected_n}" / f"seed{seed}"
        )
        model, _ = _load_checkpoint_model(run_dir / "best.pt", config, device)
        predictions = _infer(model, loader, device)
        test_run_dir = args.output_dir / "test" / f"seed{seed}"
        test_run_dir.mkdir(parents=True, exist_ok=True)
        _save_predictions(test_run_dir / "predictions.npz", predictions)
        calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
        report = evaluate_logits(
            predictions,
            calibration,
            category_count=int(metadata["category_count"]),
            bootstrap_repetitions=int(config["experiment"]["bootstrap_repetitions"]),
            bootstrap_seed=int(seed) + 1_000_000,
        )
        report["all_ground_truth_box_outcomes"] = _ground_truth_worker_outcomes(
            report, detector_test_report, role=None
        )
        report["seed"] = int(seed)
        report["operational_gate"] = _test_operational_gate(calibration, report)
        _write_json(test_run_dir / "report.json", report)
        seed_reports.append(report)
        del model
        torch.cuda.empty_cache()
    final = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "full_dataset" if full_dataset else "data_scale",
        "sample_count": len(test_records),
        "detector": {
            key: detector_test_report[key]
            for key in (
                "image_count",
                "normal_image_count",
                "recapture_image_count",
                "recapture_reasons",
                "ground_truth_count",
                "matched_count",
                "missed_count",
                "unmatched_count",
            )
        },
        "all_seeds_operational_gate": all(row["operational_gate"] for row in seed_reports),
        "status": (
            "test_certified"
            if all(row["operational_gate"] for row in seed_reports)
            else "test_gate_failed_no_reselection"
        ),
        "top1": _mean_std([float(row["overall_top1_accuracy"]) for row in seed_reports]),
        "top3": _mean_std([float(row["overall_top3_accuracy"]) for row in seed_reports]),
        "seed_reports": seed_reports,
    }
    if full_dataset:
        final["model_run"] = str(lock["model_run"])
        final["seed"] = int(lock["seed"])
        final["classifier_checkpoint_sha256"] = sha256_file(
            args.output_dir / str(lock["model_run"]) / "best.pt"
        )
    else:
        final["selected_n"] = int(selected_n)
    final["model_lock_sha256"] = sha256_file(lock_path)
    final["detector_report_sha256"] = sha256_file(args.output_dir / "test" / "detector_report.json")
    final["detector_checkpoint_sha256"] = detector_test_report["detector_checkpoint_sha256"]
    reports_dir = args.output_dir / "reports"
    _write_json(reports_dir / "final_test.json", final)
    lines = ["# RPC Classifier 최종 test 평가", ""]
    if full_dataset:
        lines.append(f"- 모델 실행: `{lock['model_run']}`")
    else:
        lines.append(f"- 선택된 클래스당 이미지 수: `{selected_n}`")
    lines.extend(
        [
            f"- 결론: `{final['status']}`",
            f"- 전체 표본 수: `{len(test_records)}`",
            f"- Top-1 mean±std: `{final['top1']['mean']:.4%} ± {final['top1']['std']:.4%}`",
            f"- Top-3 mean±std: `{final['top3']['mean']:.4%} ± {final['top3']['std']:.4%}`",
            "",
            "test gate 실패 시 이 결과로 모델을 재선택하지 않습니다.",
        ]
    )
    (reports_dir / "final_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata["final_test_report_sha256"] = sha256_file(reports_dir / "final_test.json")
    metadata["final_test_detector_report_sha256"] = sha256_file(
        args.output_dir / "test" / "detector_report.json"
    )
    _write_experiment_metadata(metadata_path, metadata, durable=True)
    return final


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated RPC classifier data-scale experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("detector", "adapt-detector", "prepare", "train", "select", "test", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if _sealed_test_metadata(args.output_dir) is not None:
        config = _load_config(args.config)
        if args.phase == "test":
            test_selected(args, config)
            return
        raise RuntimeError(
            f"post-test output is immutable; phase {args.phase!r} requires a fresh output directory"
        )
    args.config = resolve_config_path(args.config)
    config = _load_config(args.config)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "environment.json", _environment_snapshot(config, args.weights))
    if args.phase in ("detector", "all"):
        prepare_detector_phase(args, config)
    if args.phase in ("adapt-detector", "all"):
        prepare_detector_domain_adaptation(args, config)
    if args.phase in ("prepare", "all"):
        prepare(args, config)
    if args.phase in ("train", "all"):
        train_all(args, config)
    if args.phase in ("select", "all"):
        summarize(args, config)
    if args.phase in ("test", "all"):
        test_selected(args, config)


if __name__ == "__main__":
    main()
