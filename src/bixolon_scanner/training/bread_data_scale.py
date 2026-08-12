from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..imaging import decode_image, image_original_size
from ..inference import Detection, OrtRunner, _prepare_rgb, build_onnx_adapters
from ..package import load_model_package, sha256_file
from ..pipeline import quality_reasons
from .calibration import (
    binomial_rate_upper_bound,
    fit_temperature,
    select_approval_threshold,
    softmax,
)
from .data import read_manifest
from .evaluate import wilson_interval
from .evaluate_detector import _iou, _xywh_to_xyxy
from .models import build_dino_classifier, require_torch
from .small_data import (
    apply_layer_norm,
    build_frofa_training_set,
    fit_cosine_prototype_head,
    fit_linear_svm_head,
    fit_logistic_head,
    l2_normalize,
)
from .train_classifier import train as train_classifier


SCHEMA_VERSION = "1.0"
PHASES = ("prepare", "train", "calibrate", "export", "evaluate", "benchmark", "report")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("experiment", "sampling", "training", "evaluation")
    if any(not isinstance(config.get(key), dict) for key in required):
        raise ValueError(f"config must contain objects: {', '.join(required)}")
    experiment = config["experiment"]
    sizes = [int(value) for value in experiment.get("sample_sizes", [])]
    if sizes != [5, 10, 15, 20]:
        raise ValueError(
            "bread data-scale sample_sizes must be exactly [5, 10, 15, 20]"
        )
    if int(experiment.get("seed", -1)) != 20260810:
        raise ValueError("bread data-scale experiment requires seed 20260810")
    if int(experiment.get("fold_count", 0)) != 3:
        raise ValueError("bread data-scale experiment requires three development folds")
    training = config["training"]
    strategy = str(training.get("strategy", "partial_finetune"))
    if strategy not in {
        "partial_finetune",
        "frozen_cosine_prototype",
        "frozen_prototype_knn_hybrid",
        "frozen_frofa_logistic",
        "frozen_frofa_linear_svm",
    }:
        raise ValueError(f"unsupported bread training strategy: {strategy}")
    if strategy in {"frozen_frofa_logistic", "frozen_frofa_linear_svm"}:
        if not bool(training.get("feature_l2_normalize")):
            raise ValueError(
                "frozen FroFA linear training requires feature L2 normalization"
            )
        if int(training.get("frofa_views", -1)) < 1:
            raise ValueError("frozen FroFA linear training requires at least one view")
        magnitude = float(training.get("frofa_brightness_magnitude", -1.0))
        if not 0.0 <= magnitude <= 1.0:
            raise ValueError("FroFA brightness magnitude must be between 0 and 1")
    if strategy in {
        "frozen_cosine_prototype",
        "frozen_prototype_knn_hybrid",
    } and not bool(training.get("feature_l2_normalize")):
        raise ValueError("frozen embedding heads require feature L2 normalization")
    if strategy == "frozen_prototype_knn_hybrid":
        knn_k = int(training.get("hybrid_knn_k", 0))
        if not 1 <= knn_k <= min(sizes):
            raise ValueError("hybrid k must be between 1 and the smallest sample size")
        prototype_weight = float(training.get("hybrid_prototype_weight", -1.0))
        if not 0.0 <= prototype_weight <= 1.0:
            raise ValueError("hybrid prototype weight must be between 0 and 1")
    return config


def _records_and_counts(
    manifest_path: Path, metadata_path: Path, expected_classes: int, expected_aux: int
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    records = read_manifest(manifest_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    aux = [row for row in records if row["record_type"] == "classification"]
    detection = [row for row in records if row["record_type"] == "detection"]
    labels = sorted(metadata["labels"], key=lambda row: int(row["category_id"]))
    if [int(row["category_id"]) for row in labels] != list(
        range(1, expected_classes + 1)
    ):
        raise ValueError("bread labels must be contiguous and one-based")
    aux_counts = Counter(int(row["category_id"]) for row in aux)
    if any(
        aux_counts[category] != expected_aux
        for category in range(1, expected_classes + 1)
    ):
        raise ValueError(
            f"expected {expected_aux} auxiliary images per class: {dict(aux_counts)}"
        )
    development = [row for row in detection if row["split"] == "development"]
    test = [row for row in detection if row["split"] == "test"]
    if {int(row["fold"]) for row in development} != {0, 1, 2}:
        raise ValueError("development records must contain folds 0, 1, and 2")
    aux_hashes = {str(row["image_sha256"]) for row in aux}
    evaluation_hashes = {str(row["image_sha256"]) for row in development + test}
    overlap = aux_hashes & evaluation_hashes
    if overlap:
        raise ValueError(
            f"auxiliary/evaluation image SHA overlap: {sorted(overlap)[:3]}"
        )
    roi_counts: Counter[int] = Counter()
    for row in development:
        roi_counts.update(int(item["category_id"]) for item in row["annotations"])
    current_counts = []
    for label in labels:
        category = int(label["category_id"])
        current_counts.append(
            {
                "category_id": category,
                "class_id": label["class_id"],
                "class_name": label["class_name"],
                "auxiliary_images": aux_counts[category],
                "development_rois": roi_counts[category],
                "current_final_train_total": aux_counts[category]
                + roi_counts[category],
            }
        )
    return aux, development, metadata, current_counts


def _phash_bits(path: Path, hash_size: int) -> np.ndarray:
    from scipy.fft import dctn

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
    side = hash_size * 4
    variants = []
    for rotation in (0, 90, 180, 270):
        rotated = image.rotate(rotation, expand=True).resize(
            (side, side), Image.Resampling.LANCZOS
        )
        coefficients = dctn(np.asarray(rotated, dtype=np.float32), type=2, norm="ortho")
        low = coefficients[:hash_size, :hash_size].copy()
        flat = low.reshape(-1)
        threshold = float(np.median(flat[1:]))
        variants.append((flat > threshold).astype(np.uint8))
    return min(variants, key=lambda row: np.packbits(row).tobytes())


def _perceptual_groups(
    records: list[dict[str, Any]], dataset_root: Path, hash_size: int, threshold: int
) -> tuple[list[str], dict[str, list[str]]]:
    paths = [str(row["image_path"]) for row in records]
    hashes = [_phash_bits(dataset_root / path, hash_size) for path in paths]
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if int(np.count_nonzero(hashes[left] != hashes[right])) <= threshold:
                union(left, right)
    members: dict[int, list[str]] = defaultdict(list)
    for index, path in enumerate(paths):
        members[find(index)].append(path)
    identities: dict[int, str] = {
        root: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()[:16]
        for root, values in members.items()
    }
    groups = [identities[find(index)] for index in range(len(records))]
    audit = {
        identities[root]: sorted(values)
        for root, values in sorted(members.items())
        if len(values) > 1
    }
    return groups, audit


class _ImageDataset:
    def __init__(
        self, records: list[dict[str, Any]], dataset_root: Path, image_size: int
    ):
        import torchvision.transforms as transforms

        self.records = records
        self.dataset_root = dataset_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        with Image.open(
            self.dataset_root / self.records[index]["image_path"]
        ) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image)


def _extract_embeddings(
    records: list[dict[str, Any]], args: argparse.Namespace, config: dict[str, Any]
) -> np.ndarray:
    torch = require_torch()
    from torch.utils.data import DataLoader

    training = config["training"]
    model = build_dino_classifier(
        str(training["backbone_kind"]),
        int(training["num_classes"]),
        weights_path=args.weights,
        hub_repository=str(training["hub_repository"]),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    backbone = model.backbone.to(device).eval()
    dataset = _ImageDataset(records, args.dataset_root, int(training["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["sampling"]["embedding_batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        pin_memory=device.type == "cuda",
    )
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for images in loader:
            values = (
                backbone(images.to(device, non_blocking=True)).float().cpu().numpy()
            )
            result.append(values)
    embeddings = np.concatenate(result).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def diverse_order(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    group_ids: list[str],
    maximum: int,
) -> list[int]:
    if len(records) != len(embeddings) or len(records) != len(group_ids):
        raise ValueError("record, embedding, and perceptual-group counts differ")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        by_group[group].append(index)
    representatives: list[int] = []
    for group in sorted(by_group):
        indices = by_group[group]
        centroid = embeddings[indices].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        representatives.append(
            min(
                indices,
                key=lambda index: (
                    -float(embeddings[index] @ centroid),
                    records[index]["image_path"],
                ),
            )
        )
    centroid = embeddings[representatives].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    first = min(
        representatives,
        key=lambda index: (
            -float(embeddings[index] @ centroid),
            records[index]["image_path"],
        ),
    )
    selected = [first]
    representative_pool = set(representatives) - {first}
    while representative_pool and len(selected) < maximum:
        selected.append(
            min(
                representative_pool,
                key=lambda index: (
                    -min(
                        1.0 - float(embeddings[index] @ embeddings[chosen])
                        for chosen in selected
                    ),
                    records[index]["image_path"],
                ),
            )
        )
        representative_pool.remove(selected[-1])
    remaining = set(range(len(records))) - set(selected)
    while remaining and len(selected) < maximum:
        selected.append(
            min(
                remaining,
                key=lambda index: (
                    -min(
                        1.0 - float(embeddings[index] @ embeddings[chosen])
                        for chosen in selected
                    ),
                    records[index]["image_path"],
                ),
            )
        )
        remaining.remove(selected[-1])
    if len(selected) != maximum:
        raise ValueError(f"only {len(selected)} images remain; need {maximum}")
    return selected


def _known_source_family(record: dict[str, Any], perceptual_group: str) -> str:
    if int(record["category_id"]) != 19:
        return perceptual_group
    match = re.search(r"\((\d+)\)\.[^.]+$", Path(str(record["image_path"])).name)
    if match is None:
        raise ValueError(
            f"Bread19 filename does not encode its rotated source: {record['image_path']}"
        )
    return f"bread19-source-{int(match.group(1))}"


def validate_nested_orders(
    orders: dict[str, list[str]], *, category_count: int, sample_sizes: list[int]
) -> None:
    if sorted(int(key) for key in orders) != list(range(1, category_count + 1)):
        raise ValueError("selection orders do not cover every bread category")
    maximum = max(sample_sizes)
    for category, order in orders.items():
        if len(order) != maximum or len(set(order)) != maximum:
            raise ValueError(
                f"category {category} does not contain {maximum} unique selections"
            )
        previous: set[str] = set()
        for sample_size in sample_sizes:
            current = set(order[:sample_size])
            if len(current) != sample_size or not previous <= current:
                raise ValueError(
                    f"category {category} selection is not nested at n={sample_size}"
                )
            previous = current


def _contact_sheet(
    records: list[dict[str, Any]],
    dataset_root: Path,
    output: Path,
    title: str,
    thumb: int,
) -> None:
    columns = 5
    rows = math.ceil(len(records) / columns)
    label_height = 38
    canvas = Image.new(
        "RGB", (columns * thumb, rows * (thumb + label_height) + 34), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for index, record in enumerate(records):
        with Image.open(dataset_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((thumb - 8, thumb - 8), Image.Resampling.LANCZOS)
        left = (index % columns) * thumb + (thumb - image.width) // 2
        top = 34 + (index // columns) * (thumb + label_height)
        canvas.paste(image, (left, top + (thumb - image.height) // 2))
        draw.text(
            (index % columns * thumb + 6, top + thumb),
            f"#{index + 1} {Path(record['image_path']).name}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def _manifest_metadata(
    base: dict[str, Any], records: list[dict[str, Any]], sample_size: int
) -> dict[str, Any]:
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": f"bread-data-scale-n{sample_size}-{digest[:12]}",
        "base_dataset_version": base["dataset_version"],
        "record_count": len(records),
        "sample_size_per_class": sample_size,
        "labels": base["labels"],
        "manifest_sha256": digest,
        "test_accessed": False,
    }


def _hard_reasons(image: Image.Image, result, package) -> list[str]:
    reasons = quality_reasons(image, result, package.metadata.quality)
    if result.uncertain_candidate_count:
        reasons.append("DETECTOR_UNCERTAIN_OBJECT")
    count_metadata = package.metadata.count_verifier
    if count_metadata is not None:
        if result.verified_count is None or result.count_confidence is None:
            raise ValueError("count verifier result is missing")
        if result.count_confidence < count_metadata.confidence_threshold:
            reasons.append("DETECTOR_COUNT_UNCERTAIN")
        elif result.verified_count != len(result.detections):
            reasons.append("DETECTOR_COUNT_MISMATCH")
    return list(dict.fromkeys(reasons))


def _match_detections(
    detections: list[Detection], annotations: list[dict[str, Any]], threshold: float
) -> dict[int, tuple[int, float]]:
    boxes = [_xywh_to_xyxy(item["bbox_xywh"]) for item in annotations]
    remaining = set(range(len(boxes)))
    matches: dict[int, tuple[int, float]] = {}
    for detection_index, detection in sorted(
        enumerate(detections), key=lambda item: item[1].score, reverse=True
    ):
        box = np.asarray(
            [detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32
        )
        candidates = [(index, _iou(box, boxes[index])) for index in remaining]
        if not candidates:
            continue
        annotation_index, overlap = max(candidates, key=lambda item: item[1])
        if overlap >= threshold:
            remaining.remove(annotation_index)
            matches[detection_index] = (annotation_index, float(overlap))
    return matches


def _runtime_crop_tensor(
    image: Image.Image, detection: Detection, classifier_metadata
) -> np.ndarray:
    pixel_width, pixel_height = image.size
    image_width, image_height = image_original_size(image)
    scale_x = pixel_width / image_width
    scale_y = pixel_height / image_height
    margin_x = (detection.x2 - detection.x1) * classifier_metadata.crop_margin_ratio
    margin_y = (detection.y2 - detection.y1) * classifier_metadata.crop_margin_ratio
    x1 = max(0, int(np.floor(detection.x1 - margin_x)))
    y1 = max(0, int(np.floor(detection.y1 - margin_y)))
    x2 = min(image_width, int(np.ceil(detection.x2 + margin_x)))
    y2 = min(image_height, int(np.ceil(detection.y2 + margin_y)))
    crop = image.crop(
        (
            int(np.floor(x1 * scale_x)),
            int(np.floor(y1 * scale_y)),
            int(np.ceil(x2 * scale_x)),
            int(np.ceil(y2 * scale_y)),
        )
    )
    if crop.width == 0 or crop.height == 0:
        raise ValueError("detector crop is empty")
    return _prepare_rgb(
        crop,
        classifier_metadata.input_size,
        classifier_metadata.mean,
        classifier_metadata.std,
        reducing_gap=classifier_metadata.resize_reducing_gap,
    )


def _prepare_evaluation(
    development: list[dict[str, Any]], args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    prepared = args.output_dir / "prepared"
    records_path = prepared / "evaluation_records.jsonl"
    tensors_path = prepared / "evaluation_tensors.npy"
    report_path = prepared / "detector_report.json"
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "development": [row["image_sha256"] for row in development],
                "package_metadata": sha256_file(
                    args.production_package / "metadata.json"
                ),
                "detector": sha256_file(args.production_package / "detector.onnx"),
                "match_iou_threshold": config["evaluation"]["match_iou_threshold"],
            }
        ).encode()
    ).hexdigest()
    marker = prepared / "evaluation_complete.json"
    if (
        args.resume
        and marker.is_file()
        and records_path.is_file()
        and tensors_path.is_file()
    ):
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint:
            return json.loads(report_path.read_text(encoding="utf-8"))

    package = load_model_package(args.production_package)
    detector, _, provider = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    rows: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    outcomes: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    ground_truth_count = prediction_count = matched_count = count_correct = 0
    for number, record in enumerate(development, start=1):
        image = decode_image(
            (args.dataset_root / record["image_path"]).read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        result = detector.detect(image)
        detections = sorted(result.detections, key=lambda item: (item.y1, item.x1))
        reasons = _hard_reasons(image, result, package)
        reason_counts.update(reasons)
        matches = _match_detections(
            detections,
            record["annotations"],
            float(config["evaluation"]["match_iou_threshold"]),
        )
        ground_truth_count += len(record["annotations"])
        prediction_count += len(detections)
        matched_count += len(matches)
        count_correct += int(len(detections) == len(record["annotations"]))
        outcomes.append(
            {
                "image_id": int(record["image_id"]),
                "fold": int(record["fold"]),
                "capture_session_id": str(record["capture_session_id"]),
                "ground_truth_count": len(record["annotations"]),
                "detection_count": len(detections),
                "matched_count": len(matches),
                "missed_count": len(record["annotations"]) - len(matches),
                "unmatched_count": len(detections) - len(matches),
                "recapture_reasons": reasons,
            }
        )
        if not reasons:
            original_width, original_height = image_original_size(image)
            margin = package.metadata.quality.border_margin_ratio
            for detection_index, detection in enumerate(detections):
                match = matches.get(detection_index)
                annotation = (
                    record["annotations"][match[0]] if match is not None else None
                )
                rows.append(
                    {
                        "tensor_index": len(tensors),
                        "image_id": int(record["image_id"]),
                        "fold": int(record["fold"]),
                        "group_id": str(record["capture_session_id"]),
                        "detection_index": detection_index,
                        "target": -1
                        if annotation is None
                        else int(annotation["category_id"]) - 1,
                        "match_iou": None if match is None else match[1],
                        "touches_border": bool(
                            detection.x1 <= original_width * margin
                            or detection.y1 <= original_height * margin
                            or detection.x2 >= original_width * (1.0 - margin)
                            or detection.y2 >= original_height * (1.0 - margin)
                        ),
                    }
                )
                tensors.append(
                    _runtime_crop_tensor(image, detection, package.metadata.classifier)
                )
        if number % 25 == 0:
            print(
                json.dumps({"evaluation_images": number, "total": len(development)}),
                flush=True,
            )

    tensor_array = np.stack(tensors).astype(np.float32, copy=False)
    np.save(tensors_path, tensor_array)
    _write_jsonl(records_path, rows)
    detector_report = {
        "schema_version": SCHEMA_VERSION,
        "source_package": args.production_package.name,
        "provider": provider,
        "image_count": len(development),
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "recall": matched_count / ground_truth_count,
        "precision": matched_count / prediction_count if prediction_count else 0.0,
        "count_accuracy": count_correct / len(development),
        "recapture_image_count": sum(
            bool(row["recapture_reasons"]) for row in outcomes
        ),
        "recapture_reasons": dict(sorted(reason_counts.items())),
        "outcomes": outcomes,
    }
    _write_json(report_path, detector_report)
    _write_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "tensor_count": len(tensors),
            "test_accessed": False,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return detector_report


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    sampling = config["sampling"]
    aux, development, metadata, current_counts = _records_and_counts(
        args.manifest,
        args.manifest_metadata,
        int(experiment["expected_num_classes"]),
        int(experiment["expected_aux_images_per_class"]),
    )
    prepared = args.output_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    embeddings_path = prepared / "aux_embeddings.npy"
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "images": [(row["image_path"], row["image_sha256"]) for row in aux],
                "weights": sha256_file(args.weights),
                "hub_repository": config["training"]["hub_repository"],
                "image_size": config["training"]["image_size"],
            }
        ).encode()
    ).hexdigest()
    embedding_marker = prepared / "aux_embeddings.json"
    reuse = False
    if args.resume and embeddings_path.is_file() and embedding_marker.is_file():
        reuse = (
            json.loads(embedding_marker.read_text(encoding="utf-8")).get("fingerprint")
            == fingerprint
        )
    if reuse:
        embeddings = np.load(embeddings_path)
    else:
        embeddings = _extract_embeddings(aux, args, config)
        np.save(embeddings_path, embeddings)
        _write_json(
            embedding_marker,
            {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "shape": list(embeddings.shape),
                "weights_sha256": sha256_file(args.weights),
            },
        )

    orders: dict[str, list[str]] = {}
    duplicate_audit: dict[str, Any] = {}
    by_category: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(aux):
        by_category[int(record["category_id"])].append(index)
    maximum = max(int(value) for value in experiment["sample_sizes"])
    labels = {int(row["category_id"]): row for row in metadata["labels"]}
    for category in range(1, int(experiment["expected_num_classes"]) + 1):
        indices = by_category[category]
        category_records = [aux[index] for index in indices]
        exact = Counter(str(row["image_sha256"]) for row in category_records)
        exact_duplicates = sorted(
            digest for digest, count in exact.items() if count > 1
        )
        if exact_duplicates:
            raise ValueError(
                f"class {category} contains exact SHA duplicates: {exact_duplicates[:3]}"
            )
        perceptual_groups, near_duplicates = _perceptual_groups(
            category_records,
            args.dataset_root,
            int(sampling["perceptual_hash_size"]),
            int(sampling["perceptual_hash_hamming_threshold"]),
        )
        groups = [
            _known_source_family(record, group)
            for record, group in zip(category_records, perceptual_groups)
        ]
        family_members: dict[str, list[str]] = defaultdict(list)
        for record, group in zip(category_records, groups):
            family_members[group].append(str(record["image_path"]))
        known_families = {
            group: sorted(paths)
            for group, paths in family_members.items()
            if len(paths) > 1
        }
        local_order = diverse_order(
            category_records, embeddings[indices], groups, maximum
        )
        selected_records = [category_records[index] for index in local_order]
        orders[str(category)] = [str(row["image_path"]) for row in selected_records]
        duplicate_audit[str(category)] = {
            "perceptual": near_duplicates,
            "known_source_families": known_families,
        }
        slug = str(labels[category]["class_id"])
        thumb = int(sampling["contact_sheet_thumbnail_size"])
        first_n = int(sampling["contact_sheet_first_n"])
        _contact_sheet(
            selected_records[:first_n],
            args.dataset_root,
            prepared / "contact_sheets" / f"{slug}-first5.jpg",
            f"{slug} {labels[category]['class_name']} - first {first_n}",
            thumb,
        )
        _contact_sheet(
            selected_records,
            args.dataset_root,
            prepared / "contact_sheets" / f"{slug}-order20.jpg",
            f"{slug} {labels[category]['class_name']} - nested order",
            thumb,
        )

    aux_by_path = {str(row["image_path"]): row for row in aux}
    validate_nested_orders(
        orders,
        category_count=int(experiment["expected_num_classes"]),
        sample_sizes=[int(value) for value in experiment["sample_sizes"]],
    )
    for sample_size in experiment["sample_sizes"]:
        selected_paths = {
            path
            for category_order in orders.values()
            for path in category_order[: int(sample_size)]
        }
        selected = sorted(
            (aux_by_path[path] for path in selected_paths),
            key=lambda row: (int(row["category_id"]), str(row["image_path"])),
        )
        manifest_dir = prepared / "manifests" / f"n{sample_size}"
        _write_jsonl(manifest_dir / "manifest.jsonl", selected)
        _write_json(
            manifest_dir / "metadata.json",
            _manifest_metadata(metadata, selected, int(sample_size)),
        )

    review = {
        "schema_version": SCHEMA_VERSION,
        "status": "approved" if args.approve_selection else "requires_visual_review",
        "criteria": [
            "wrong_class",
            "corrupt_or_unreadable",
            "identity_destroying_crop",
            "rotation_or_encoding_only_duplicate_in_first_five",
        ],
        "approved_at": datetime.now(UTC).isoformat()
        if args.approve_selection
        else None,
        "orders": orders,
        "image_hashes": {
            str(row["image_path"]): str(row["image_sha256"]) for row in aux
        },
        "weights_sha256": sha256_file(args.weights),
        "manifest_sha256": sha256_file(args.manifest),
        "perceptual_duplicate_candidates": duplicate_audit,
        "test_accessed": False,
    }
    _write_json(prepared / "selection_review.json", review)
    _write_json(prepared / "current_training_counts.json", current_counts)
    detector_report = _prepare_evaluation(development, args, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "auxiliary_image_count": len(aux),
        "development_image_count": len(development),
        "sample_sizes": experiment["sample_sizes"],
        "selection_status": review["status"],
        "current_training_total": sum(
            row["current_final_train_total"] for row in current_counts
        ),
        "current_counts": current_counts,
        "detector": {
            key: detector_report[key]
            for key in ("recall", "precision", "count_accuracy")
        },
        "test_accessed": False,
    }
    _write_json(prepared / "summary.json", summary)
    return summary


def _training_namespace(
    args: argparse.Namespace, config: dict[str, Any], sample_size: int
) -> argparse.Namespace:
    training = config["training"]
    cache_dir = args.classifier_cache_dir
    return argparse.Namespace(
        manifest=args.output_dir
        / "prepared"
        / "manifests"
        / f"n{sample_size}"
        / "manifest.jsonl",
        dataset_root=args.dataset_root,
        output_dir=args.output_dir / "runs" / f"n{sample_size}",
        fold=0,
        final_training=True,
        backbone_kind=str(training["backbone_kind"]),
        weights=args.weights,
        cache_dir=cache_dir,
        hub_repository=str(training["hub_repository"]),
        pretrained_name="facebook/dinov2-small",
        num_classes=int(training["num_classes"]),
        image_size=int(training["image_size"]),
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        frozen_epochs=int(training["frozen_epochs"]),
        finetune_epochs=int(training["finetune_epochs"]),
        frozen_lr=float(training["frozen_lr"]),
        finetune_lr=float(training["finetune_lr"]),
        seed=int(config["experiment"]["seed"]),
        cpu=bool(args.cpu),
    )


def _extract_support_patch_cache(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    prepared = args.output_dir / "prepared"
    patches_path = prepared / "support_patch_features.npy"
    cache_path = prepared / "support_patch_features.json"
    norm_path = prepared / "support_patch_norm.npz"
    maximum = max(int(value) for value in config["experiment"]["sample_sizes"])
    manifest_path = prepared / "manifests" / f"n{maximum}" / "manifest.jsonl"
    expected = {
        "weights_sha256": sha256_file(args.weights),
        "manifest_sha256": sha256_file(manifest_path),
        "backbone_kind": str(config["training"]["backbone_kind"]),
        "feature_source": "x_prenorm",
    }
    if patches_path.is_file() and cache_path.is_file() and norm_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            return cached

    records = read_manifest(manifest_path)
    torch = require_torch()
    from torch.utils.data import DataLoader

    training = config["training"]
    model = build_dino_classifier(
        str(training["backbone_kind"]),
        int(training["num_classes"]),
        weights_path=args.weights,
        hub_repository=str(training["hub_repository"]),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    backbone = model.backbone.to(device).eval()
    dataset = _ImageDataset(records, args.dataset_root, int(training["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        pin_memory=device.type == "cuda",
    )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images in loader:
            outputs = backbone.forward_features(images.to(device, non_blocking=True))
            batches.append(outputs["x_prenorm"].float().cpu().numpy())
    patches = np.concatenate(batches).astype(np.float32)
    np.save(patches_path, patches)
    np.savez(
        norm_path,
        weight=backbone.norm.weight.detach().float().cpu().numpy(),
        bias=backbone.norm.bias.detach().float().cpu().numpy(),
        epsilon=np.asarray(float(backbone.norm.eps), dtype=np.float64),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        **expected,
        "record_count": len(records),
        "shape": list(patches.shape),
        "image_paths": [str(row["image_path"]) for row in records],
        "category_ids": [int(row["category_id"]) for row in records],
        "patches_sha256": sha256_file(patches_path),
        "norm_sha256": sha256_file(norm_path),
        "test_accessed": False,
    }
    _write_json(cache_path, metadata)
    return metadata


def _train_frofa_linear_head(
    args: argparse.Namespace, config: dict[str, Any], sample_size: int
) -> None:
    training = config["training"]
    cache = _extract_support_patch_cache(args, config)
    prepared = args.output_dir / "prepared"
    patches = np.load(prepared / "support_patch_features.npy", mmap_mode="r")
    norm = np.load(prepared / "support_patch_norm.npz")
    subset_manifest = prepared / "manifests" / f"n{sample_size}" / "manifest.jsonl"
    subset_records = read_manifest(subset_manifest)
    subset_paths = {str(row["image_path"]) for row in subset_records}
    selected = np.asarray(
        [path in subset_paths for path in cache["image_paths"]], dtype=bool
    )
    expected_count = int(training["num_classes"]) * sample_size
    if int(selected.sum()) != expected_count:
        raise ValueError(
            f"n={sample_size} support cache selected {int(selected.sum())}, expected {expected_count}"
        )
    labels = np.asarray(cache["category_ids"], dtype=np.int64)[selected] - 1
    features, augmented_labels = build_frofa_training_set(
        np.asarray(patches[selected]),
        labels,
        layer_norm_weight=norm["weight"],
        layer_norm_bias=norm["bias"],
        layer_norm_epsilon=float(norm["epsilon"]),
        magnitude=float(training["frofa_brightness_magnitude"]),
        views=int(training["frofa_views"]),
        seed=int(config["experiment"]["seed"]),
    )
    strategy = str(training["strategy"])
    if strategy == "frozen_frofa_logistic":
        regularization_c = float(training["logistic_regularization_c"])
        max_iterations = int(training["logistic_max_iterations"])
        head = fit_logistic_head(
            features,
            augmented_labels,
            num_classes=int(training["num_classes"]),
            regularization_c=regularization_c,
            max_iterations=max_iterations,
            seed=int(config["experiment"]["seed"]),
        )
        solver = "lbfgs"
    else:
        regularization_c = float(training["linear_svm_regularization_c"])
        max_iterations = int(training["linear_svm_max_iterations"])
        head = fit_linear_svm_head(
            features,
            augmented_labels,
            num_classes=int(training["num_classes"]),
            regularization_c=regularization_c,
            max_iterations=max_iterations,
            seed=int(config["experiment"]["seed"]),
        )
        solver = "liblinear_squared_hinge"
    torch = require_torch()
    model = build_dino_classifier(
        str(training["backbone_kind"]),
        int(training["num_classes"]),
        weights_path=args.weights,
        hub_repository=str(training["hub_repository"]),
        feature_l2_normalize=True,
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.from_numpy(head.weights))
        model.classifier.bias.copy_(torch.from_numpy(head.bias))
    output_dir = args.output_dir / "runs" / f"n{sample_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "sample_size_per_class": sample_size,
        "clean_support_count": int(selected.sum()),
        "training_feature_count": len(features),
        "feature_l2_normalize": True,
        "frofa": {
            "operation": "brightness_c2",
            "magnitude": float(training["frofa_brightness_magnitude"]),
            "views": int(training["frofa_views"]),
            "feature_source": "x_prenorm",
        },
        "linear_head": {
            "regularization_c": regularization_c,
            "solver": solver,
            "iterations": head.iterations,
            "max_iterations": max_iterations,
        },
        "seed": int(config["experiment"]["seed"]),
        "manifest_sha256": sha256_file(subset_manifest),
        "support_patch_features_sha256": cache["patches_sha256"],
        "test_accessed": False,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone_kind": str(training["backbone_kind"]),
        "pretrained_name": "facebook/dinov2-small",
        "backbone_architecture": str(training["backbone_kind"]),
        "source_revision": str(training["hub_repository"]).split(":", 1)[-1],
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": sha256_file(args.weights),
        "num_classes": int(training["num_classes"]),
        "image_size": int(training["image_size"]),
        "stage": strategy,
        "feature_l2_normalize": True,
        "training_recipe": recipe,
        "metrics": {"selection_uses_development_labels": False},
        "selection": {
            "selected": strategy,
            "automatic_sample_size_recommendation": False,
        },
    }
    torch.save(checkpoint, output_dir / "best.pt")
    _write_json(output_dir / "training_recipe.json", recipe)
    _write_json(output_dir / "selection.json", checkpoint["selection"])


def _train_cosine_prototype_head(
    args: argparse.Namespace, config: dict[str, Any], sample_size: int
) -> None:
    """Store normalized class means as an exact cosine-similarity ONNX head."""
    training = config["training"]
    cache = _extract_support_patch_cache(args, config)
    prepared = args.output_dir / "prepared"
    patches = np.load(prepared / "support_patch_features.npy", mmap_mode="r")
    norm = np.load(prepared / "support_patch_norm.npz")
    subset_manifest = prepared / "manifests" / f"n{sample_size}" / "manifest.jsonl"
    subset_records = read_manifest(subset_manifest)
    subset_paths = {str(row["image_path"]) for row in subset_records}
    selected = np.asarray(
        [path in subset_paths for path in cache["image_paths"]], dtype=bool
    )
    expected_count = int(training["num_classes"]) * sample_size
    if int(selected.sum()) != expected_count:
        raise ValueError(
            f"n={sample_size} support cache selected {int(selected.sum())}, expected {expected_count}"
        )
    labels = np.asarray(cache["category_ids"], dtype=np.int64)[selected] - 1
    pooled = apply_layer_norm(
        np.asarray(patches[selected]).mean(axis=1),
        norm["weight"],
        norm["bias"],
        epsilon=float(norm["epsilon"]),
    )
    features = l2_normalize(pooled)
    head = fit_cosine_prototype_head(
        features,
        labels,
        num_classes=int(training["num_classes"]),
    )
    if not np.all(head.counts == sample_size):
        raise ValueError(
            f"prototype support counts must all equal {sample_size}: {head.counts.tolist()}"
        )

    torch = require_torch()
    model = build_dino_classifier(
        str(training["backbone_kind"]),
        int(training["num_classes"]),
        weights_path=args.weights,
        hub_repository=str(training["hub_repository"]),
        feature_l2_normalize=True,
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.from_numpy(head.weights))
        model.classifier.bias.copy_(torch.from_numpy(head.bias))

    output_dir = args.output_dir / "runs" / f"n{sample_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_sha256 = hashlib.sha256(
        np.ascontiguousarray(head.weights).tobytes()
    ).hexdigest()
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "frozen_cosine_prototype",
        "sample_size_per_class": sample_size,
        "clean_support_count": int(selected.sum()),
        "feature_l2_normalize": True,
        "feature_source": "x_prenorm_mean_then_backbone_layer_norm",
        "similarity": "cosine",
        "aggregation": "normalized_class_mean",
        "optimizer": None,
        "prototype_counts": head.counts.tolist(),
        "prototype_sha256": prototype_sha256,
        "seed": int(config["experiment"]["seed"]),
        "manifest_sha256": sha256_file(subset_manifest),
        "support_patch_features_sha256": cache["patches_sha256"],
        "test_accessed": False,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone_kind": str(training["backbone_kind"]),
        "pretrained_name": "facebook/dinov2-small",
        "backbone_architecture": str(training["backbone_kind"]),
        "source_revision": str(training["hub_repository"]).split(":", 1)[-1],
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": sha256_file(args.weights),
        "num_classes": int(training["num_classes"]),
        "image_size": int(training["image_size"]),
        "stage": "frozen_cosine_prototype",
        "feature_l2_normalize": True,
        "training_recipe": recipe,
        "metrics": {"selection_uses_development_labels": False},
        "selection": {
            "selected": "frozen_cosine_prototype",
            "automatic_sample_size_recommendation": False,
        },
    }
    torch.save(checkpoint, output_dir / "best.pt")
    _write_json(output_dir / "training_recipe.json", recipe)
    _write_json(output_dir / "selection.json", checkpoint["selection"])


def _train_prototype_knn_hybrid_head(
    args: argparse.Namespace, config: dict[str, Any], sample_size: int
) -> None:
    training = config["training"]
    cache = _extract_support_patch_cache(args, config)
    prepared = args.output_dir / "prepared"
    patches = np.load(prepared / "support_patch_features.npy", mmap_mode="r")
    norm = np.load(prepared / "support_patch_norm.npz")
    subset_manifest = prepared / "manifests" / f"n{sample_size}" / "manifest.jsonl"
    subset_records = read_manifest(subset_manifest)
    subset_paths = {str(row["image_path"]) for row in subset_records}
    selected = np.asarray(
        [path in subset_paths for path in cache["image_paths"]], dtype=bool
    )
    expected_count = int(training["num_classes"]) * sample_size
    if int(selected.sum()) != expected_count:
        raise ValueError(
            f"n={sample_size} support cache selected {int(selected.sum())}, expected {expected_count}"
        )
    labels = np.asarray(cache["category_ids"], dtype=np.int64)[selected] - 1
    pooled = apply_layer_norm(
        np.asarray(patches[selected]).mean(axis=1),
        norm["weight"],
        norm["bias"],
        epsilon=float(norm["epsilon"]),
    )
    features = l2_normalize(pooled)
    num_classes = int(training["num_classes"])
    prototype_head = fit_cosine_prototype_head(
        features,
        labels,
        num_classes=num_classes,
    )
    exemplars = np.stack(
        [features[labels == class_index] for class_index in range(num_classes)]
    ).astype(np.float32)
    if exemplars.shape[:2] != (num_classes, sample_size):
        raise ValueError(
            f"hybrid exemplar shape must start with {(num_classes, sample_size)}, got {exemplars.shape}"
        )
    knn_k = int(training["hybrid_knn_k"])
    prototype_weight = float(training["hybrid_prototype_weight"])

    torch = require_torch()
    model = build_dino_classifier(
        str(training["backbone_kind"]),
        num_classes,
        weights_path=args.weights,
        hub_repository=str(training["hub_repository"]),
        feature_l2_normalize=True,
        classifier_head_kind="prototype_knn_hybrid",
        support_per_class=sample_size,
        hybrid_knn_k=knn_k,
        hybrid_prototype_weight=prototype_weight,
    )
    with torch.no_grad():
        model.classifier.prototypes.copy_(torch.from_numpy(prototype_head.weights))
        model.classifier.exemplars.copy_(torch.from_numpy(exemplars))

    output_dir = args.output_dir / "runs" / f"n{sample_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "frozen_prototype_knn_hybrid",
        "sample_size_per_class": sample_size,
        "clean_support_count": int(selected.sum()),
        "feature_l2_normalize": True,
        "feature_source": "x_prenorm_mean_then_backbone_layer_norm",
        "similarity": "cosine",
        "prototype": "normalized_class_mean",
        "knn": "per_class_top_k_mean",
        "knn_k": knn_k,
        "prototype_weight": prototype_weight,
        "knn_weight": 1.0 - prototype_weight,
        "optimizer": None,
        "prototype_sha256": hashlib.sha256(
            np.ascontiguousarray(prototype_head.weights).tobytes()
        ).hexdigest(),
        "exemplar_sha256": hashlib.sha256(
            np.ascontiguousarray(exemplars).tobytes()
        ).hexdigest(),
        "seed": int(config["experiment"]["seed"]),
        "manifest_sha256": sha256_file(subset_manifest),
        "support_patch_features_sha256": cache["patches_sha256"],
        "test_accessed": False,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone_kind": str(training["backbone_kind"]),
        "pretrained_name": "facebook/dinov2-small",
        "backbone_architecture": str(training["backbone_kind"]),
        "source_revision": str(training["hub_repository"]).split(":", 1)[-1],
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": sha256_file(args.weights),
        "num_classes": num_classes,
        "image_size": int(training["image_size"]),
        "stage": "frozen_prototype_knn_hybrid",
        "feature_l2_normalize": True,
        "classifier_head_kind": "prototype_knn_hybrid",
        "support_per_class": sample_size,
        "hybrid_knn_k": knn_k,
        "hybrid_prototype_weight": prototype_weight,
        "training_recipe": recipe,
        "metrics": {"selection_uses_development_labels": False},
        "selection": {
            "selected": "frozen_prototype_knn_hybrid",
            "automatic_sample_size_recommendation": False,
        },
    }
    torch.save(checkpoint, output_dir / "best.pt")
    _write_json(output_dir / "training_recipe.json", recipe)
    _write_json(output_dir / "selection.json", checkpoint["selection"])


def train_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    review_path = args.output_dir / "prepared" / "selection_review.json"
    if not review_path.is_file():
        raise FileNotFoundError("prepare phase has not completed")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["status"] != "approved":
        if not args.approve_selection:
            raise RuntimeError(
                "contact sheets require visual approval; pass --approve-selection"
            )
        review["status"] = "approved"
        review["approved_at"] = datetime.now(UTC).isoformat()
        _write_json(review_path, review)
    strategy = str(config["training"].get("strategy", "partial_finetune"))
    for sample_size in config["experiment"]["sample_sizes"]:
        sample_size = int(sample_size)
        output_dir = args.output_dir / "runs" / f"n{sample_size}"
        manifest = (
            args.output_dir
            / "prepared"
            / "manifests"
            / f"n{sample_size}"
            / "manifest.jsonl"
        )
        complete = output_dir / "complete.json"
        if args.resume and complete.is_file() and (output_dir / "best.pt").is_file():
            continue
        if strategy == "partial_finetune":
            namespace = _training_namespace(args, config, sample_size)
            train_classifier(namespace)
        elif strategy == "frozen_cosine_prototype":
            _train_cosine_prototype_head(args, config, sample_size)
        elif strategy == "frozen_prototype_knn_hybrid":
            _train_prototype_knn_hybrid_head(args, config, sample_size)
        elif strategy in {"frozen_frofa_logistic", "frozen_frofa_linear_svm"}:
            _train_frofa_linear_head(args, config, sample_size)
        else:
            raise ValueError(f"unsupported bread training strategy: {strategy}")
        _write_json(
            complete,
            {
                "schema_version": SCHEMA_VERSION,
                "sample_size": sample_size,
                "checkpoint_sha256": sha256_file(output_dir / "best.pt"),
                "manifest_sha256": sha256_file(manifest),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )


def _load_checkpoint_model(checkpoint_path: Path, config: dict[str, Any], device):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_dino_classifier(
        checkpoint["backbone_kind"],
        int(checkpoint["num_classes"]),
        pretrained_name=checkpoint.get("pretrained_name", "facebook/dinov2-small"),
        hub_repository=str(config["training"]["hub_repository"]),
        feature_l2_normalize=bool(checkpoint.get("feature_l2_normalize", False)),
        classifier_head_kind=str(checkpoint.get("classifier_head_kind", "linear")),
        support_per_class=checkpoint.get("support_per_class"),
        hybrid_knn_k=int(checkpoint.get("hybrid_knn_k", 3)),
        hybrid_prototype_weight=float(checkpoint.get("hybrid_prototype_weight", 0.5)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _infer_checkpoint(
    checkpoint_path: Path, tensors_path: Path, config: dict[str, Any], *, cpu: bool
) -> np.ndarray:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model, _ = _load_checkpoint_model(checkpoint_path, config, device)
    tensors = np.load(tensors_path, mmap_mode="r")
    batch_size = int(config["training"]["batch_size"])
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], copy=True)
            ).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _fit_calibration(
    logits: np.ndarray, targets: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    matched = targets >= 0
    if not matched.any():
        raise ValueError("calibration contains no matched detector crops")
    temperature = fit_temperature(logits[matched], targets[matched])
    probabilities = softmax(logits[matched], temperature)
    threshold = select_approval_threshold(
        probabilities,
        targets[matched],
        max_false_approval_rate=float(config["experiment"]["max_false_approval_rate"]),
        confidence_level=float(config["experiment"]["confidence_level"]),
    )
    return {
        "sample_count": int(matched.sum()),
        "temperature": temperature,
        "approval_threshold": threshold.threshold,
        "approved_count": threshold.approved_count,
        "approved_precision": threshold.approved_precision,
        "approval_coverage": threshold.coverage,
        "approved_false_rate_upper_95": threshold.false_approval_rate_upper,
        "risk_control_satisfied": threshold.risk_control_satisfied,
    }


def cross_fold_calibrations(
    logits: np.ndarray, targets: np.ndarray, folds: np.ndarray, config: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for held_out in range(int(config["experiment"]["fold_count"])):
        calibration_mask = folds != held_out
        calibration = _fit_calibration(
            logits[calibration_mask], targets[calibration_mask], config
        )
        calibration["held_out_fold"] = held_out
        calibration["calibration_folds"] = sorted(set(folds[calibration_mask].tolist()))
        result[held_out] = calibration
    return result


def _cluster_bootstrap(
    successes: np.ndarray, groups: np.ndarray, repetitions: int, seed: int
) -> list[float]:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    success_counts = np.bincount(inverse, weights=successes.astype(np.float64))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    rates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(unique), len(unique))
        rates[index] = success_counts[sampled].sum() / counts[sampled].sum()
    return [float(value) for value in np.quantile(rates, [0.025, 0.975])]


def _evaluate_crossfit(
    logits: np.ndarray,
    rows: list[dict[str, Any]],
    detector_report: dict[str, Any],
    fold_calibrations: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(logits) != len(rows):
        raise ValueError("logit and evaluation record counts differ")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row["group_id"]) for row in rows])
    touches_border = np.asarray([bool(row["touches_border"]) for row in rows])
    probabilities = np.empty_like(logits, dtype=np.float64)
    thresholds = np.empty(len(rows), dtype=np.float64)
    for fold, calibration in fold_calibrations.items():
        mask = folds == fold
        probabilities[mask] = softmax(logits[mask], float(calibration["temperature"]))
        thresholds[mask] = float(calibration["approval_threshold"])
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    top3 = np.argsort(-probabilities, axis=1)[:, :3]
    matched = targets >= 0
    correct = matched & (predicted == targets)
    top3_correct = matched & np.any(top3 == targets[:, None], axis=1)
    border_recapture_ids = set(
        image_ids[touches_border & (confidence < thresholds)].tolist()
    )
    active = np.asarray([int(value) not in border_recapture_ids for value in image_ids])
    approved = active & (confidence >= thresholds)
    unknown = active & ~approved
    approved_correct = int((approved & correct).sum())
    approved_count = int(approved.sum())
    matched_unknown = unknown & matched
    approved_precision = approved_correct / approved_count if approved_count else 1.0

    status_counts: Counter[str] = Counter()
    reason_counts = Counter(detector_report["recapture_reasons"])
    approved_images = approved_images_correct = 0
    normal_outcomes = [
        row for row in detector_report["outcomes"] if not row["recapture_reasons"]
    ]
    row_indices_by_image: dict[int, list[int]] = defaultdict(list)
    for index, image_id in enumerate(image_ids):
        row_indices_by_image[int(image_id)].append(index)
    for outcome in detector_report["outcomes"]:
        image_id = int(outcome["image_id"])
        if outcome["recapture_reasons"]:
            status_counts["RECAPTURE"] += 1
            continue
        if image_id in border_recapture_ids:
            status_counts["RECAPTURE"] += 1
            reason_counts["DETECTOR_BORDER_CLIPPED"] += 1
            continue
        indices = row_indices_by_image[image_id]
        all_approved = all(bool(approved[index]) for index in indices)
        status_counts["APPROVED" if all_approved else "UNKNOWN"] += 1
        if all_approved:
            approved_images += 1
            image_correct = (
                int(outcome["ground_truth_count"]) == len(indices)
                and int(outcome["matched_count"]) == len(indices)
                and all(bool(correct[index]) for index in indices)
            )
            approved_images_correct += int(image_correct)

    class_top1: list[float | None] = []
    class_top3: list[float | None] = []
    for target in range(int(config["experiment"]["expected_num_classes"])):
        mask = matched & (targets == target)
        class_top1.append(float(correct[mask].mean()) if mask.any() else None)
        class_top3.append(float(top3_correct[mask].mean()) if mask.any() else None)
    metric = matched
    precision_interval = wilson_interval(approved_correct, approved_count)
    false_approval_rate_upper = binomial_rate_upper_bound(
        approved_count - approved_correct,
        approved_count,
        confidence_level=float(config["experiment"]["confidence_level"]),
    )
    max_false_approval_rate = float(config["experiment"]["max_false_approval_rate"])
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_policy": "development-3fold-cross-calibrated",
        "test_accessed": False,
        "sample_count": len(rows),
        "matched_sample_count": int(metric.sum()),
        "unmatched_detector_count": int((~matched).sum()),
        "overall_top1_accuracy": float(correct[metric].mean()),
        "overall_top3_accuracy": float(top3_correct[metric].mean()),
        "top1_capture_session_bootstrap_95ci": _cluster_bootstrap(
            correct[metric],
            groups[metric],
            int(config["experiment"]["bootstrap_repetitions"]),
            int(config["experiment"]["seed"]),
        ),
        "per_class_top1": class_top1,
        "per_class_top3": class_top3,
        "approved_count": approved_count,
        "approved_correct": approved_correct,
        "approved_unmatched": int((approved & ~matched).sum()),
        "approval_coverage_of_detections": approved_count / int(active.sum())
        if active.any()
        else 0.0,
        "approved_precision": approved_precision,
        "approved_precision_95ci": list(precision_interval),
        "approved_precision_target": 1.0 - max_false_approval_rate,
        "approved_false_rate_upper_95": false_approval_rate_upper,
        "approved_point_precision_gate_satisfied": (
            approved_precision >= 1.0 - max_false_approval_rate
        ),
        "approved_precision_gate_satisfied": (
            false_approval_rate_upper <= max_false_approval_rate
        ),
        "unknown_count": int(unknown.sum()),
        "unknown_matched_count": int(matched_unknown.sum()),
        "unknown_top3_correct": int((top3_correct & matched_unknown).sum()),
        "unknown_top3_accuracy": (
            float(top3_correct[matched_unknown].mean())
            if matched_unknown.any()
            else None
        ),
        "unknown_top3_gate_satisfied": bool(
            matched_unknown.any() and top3_correct[matched_unknown].mean() >= 0.95
        ),
        "classifier_border_recapture_images": len(border_recapture_ids),
        "frame_policy": {
            "image_count": len(detector_report["outcomes"]),
            "normal_detector_image_count": len(normal_outcomes),
            "status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "approved_image_count": approved_images,
            "approved_image_correct": approved_images_correct,
            "approved_image_precision": (
                approved_images_correct / approved_images if approved_images else None
            ),
        },
        "detector": {
            key: detector_report[key]
            for key in (
                "image_count",
                "ground_truth_count",
                "prediction_count",
                "matched_count",
                "recall",
                "precision",
                "count_accuracy",
                "recapture_image_count",
                "recapture_reasons",
            )
        },
    }


def calibrate_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    records_path = args.output_dir / "prepared" / "evaluation_records.jsonl"
    tensors_path = args.output_dir / "prepared" / "evaluation_tensors.npy"
    detector_path = args.output_dir / "prepared" / "detector_report.json"
    if not records_path.is_file() or not tensors_path.is_file():
        raise FileNotFoundError("prepare phase has not completed")
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    detector_report = json.loads(detector_path.read_text(encoding="utf-8"))
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    for sample_size in config["experiment"]["sample_sizes"]:
        run_dir = args.output_dir / "runs" / f"n{sample_size}"
        checkpoint_path = run_dir / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"training has not completed for n={sample_size}")
        logits_path = run_dir / "development_logits.npz"
        calibration_path = run_dir / "calibration.json"
        evaluation_path = run_dir / "crossfit_evaluation.json"
        if (
            args.resume
            and logits_path.is_file()
            and calibration_path.is_file()
            and evaluation_path.is_file()
        ):
            continue
        logits = _infer_checkpoint(checkpoint_path, tensors_path, config, cpu=args.cpu)
        np.savez_compressed(
            logits_path,
            logits=logits,
            targets=targets,
            folds=folds,
            image_ids=np.asarray([int(row["image_id"]) for row in rows]),
            groups=np.asarray([str(row["group_id"]) for row in rows]),
        )
        fold_calibrations = cross_fold_calibrations(logits, targets, folds, config)
        final_calibration = _fit_calibration(logits, targets, config)
        report = {
            "schema_version": SCHEMA_VERSION,
            "sample_size_per_class": int(sample_size),
            "seed": int(config["experiment"]["seed"]),
            "policy": "fit-on-two-folds-evaluate-held-out; final-fit-on-all-development",
            "folds": {str(key): value for key, value in fold_calibrations.items()},
            "final": final_calibration,
            "test_accessed": False,
        }
        _write_json(calibration_path, report)
        evaluation = _evaluate_crossfit(
            logits, rows, detector_report, fold_calibrations, config
        )
        evaluation["sample_size_per_class"] = int(sample_size)
        _write_json(evaluation_path, evaluation)


def _export_classifier(
    checkpoint_path: Path, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    torch = require_torch()
    model, checkpoint = _load_checkpoint_model(
        checkpoint_path, config, torch.device("cpu")
    )
    size = int(checkpoint["image_size"])
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        output,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(output))
    return checkpoint


def export_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    source_package = load_model_package(args.production_package)
    source_metadata = json.loads(
        (args.production_package / "metadata.json").read_text(encoding="utf-8")
    )
    for sample_size in config["experiment"]["sample_sizes"]:
        run_dir = args.output_dir / "runs" / f"n{sample_size}"
        checkpoint_path = run_dir / "best.pt"
        calibration_path = run_dir / "calibration.json"
        if not checkpoint_path.is_file() or not calibration_path.is_file():
            raise FileNotFoundError(
                f"calibration has not completed for n={sample_size}"
            )
        package_dir = args.output_dir / "packages" / f"n{sample_size}"
        marker = package_dir / "complete.json"
        if args.resume and marker.is_file():
            load_model_package(package_dir)
            continue
        package_dir.mkdir(parents=True, exist_ok=True)
        for source in source_package.root.iterdir():
            if source.is_file() and source.name not in {
                "classifier.onnx",
                "metadata.json",
            }:
                shutil.copy2(source, package_dir / source.name)
        classifier_path = package_dir / "classifier.onnx"
        checkpoint = _export_classifier(checkpoint_path, classifier_path, config)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["final"]
        subset_metadata = json.loads(
            (
                args.output_dir
                / "prepared"
                / "manifests"
                / f"n{sample_size}"
                / "metadata.json"
            ).read_text(encoding="utf-8")
        )
        metadata = json.loads(json.dumps(source_metadata))
        strategy = str(config["training"].get("strategy", "partial_finetune"))
        if strategy.startswith("frozen_frofa_"):
            suffix = "-frofa"
        elif strategy == "frozen_cosine_prototype":
            suffix = "-prototype"
        elif strategy == "frozen_prototype_knn_hybrid":
            suffix = "-hybrid"
        else:
            suffix = ""
        version = f"0.1.0-breadscale{suffix}.n{sample_size}"
        metadata["package_version"] = version
        metadata["promotion_status"] = "development"
        metadata.pop("promotion", None)
        metadata["dataset_version"] = subset_metadata["dataset_version"]
        metadata["classifier"]["version"] = version
        metadata["classifier"]["approval_threshold"] = calibration["approval_threshold"]
        metadata["classifier"]["temperature"] = calibration["temperature"]
        metadata["checksums"][source_package.metadata.detector.filename] = sha256_file(
            package_dir / source_package.metadata.detector.filename
        )
        metadata["checksums"]["classifier.onnx"] = sha256_file(classifier_path)
        metadata["sources"]["classifier"] = {
            "architecture": checkpoint.get("backbone_architecture"),
            "revision": checkpoint.get("source_revision"),
            "weight_filename": checkpoint.get("source_weight_filename"),
            "weight_sha256": checkpoint.get("source_weight_sha256"),
        }
        metadata["calibration"] = {
            "sample_count": calibration["sample_count"],
            "approved_precision": calibration["approved_precision"],
            "approval_coverage": calibration["approval_coverage"],
            "false_approval_rate_upper_95": calibration["approved_false_rate_upper_95"],
            "risk_control_satisfied": calibration["risk_control_satisfied"],
        }
        _write_json(package_dir / "metadata.json", metadata)
        load_model_package(package_dir)
        _write_json(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "source_package": args.production_package.name,
                "source_detector_sha256": sha256_file(source_package.detector_path),
                "classifier_sha256": sha256_file(classifier_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "test_accessed": False,
            },
        )


def _infer_onnx(
    package_dir: Path,
    tensors_path: Path,
    provider: str,
    cuda_dll_dir: Path | None,
    batch_size: int,
) -> np.ndarray:
    if provider == "cuda":
        require_torch()
    package = load_model_package(package_dir)
    runner = OrtRunner(package.classifier_path, provider, cuda_dll_dir)
    tensors = np.load(tensors_path, mmap_mode="r")
    outputs: list[np.ndarray] = []
    for start in range(0, len(tensors), batch_size):
        batch = np.asarray(tensors[start : start + batch_size], dtype=np.float32)
        (logits,) = runner.run(
            [package.metadata.classifier.logits_output],
            package.metadata.classifier.input_name,
            batch,
        )
        outputs.append(np.asarray(logits, dtype=np.float32))
    return np.concatenate(outputs)


def _classifier_signature(
    logits: np.ndarray, rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    probabilities = softmax(logits, float(calibration["temperature"]))
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    threshold = float(calibration["approval_threshold"])
    image_ids = np.asarray([int(row["image_id"]) for row in rows])
    border = np.asarray([bool(row["touches_border"]) for row in rows])
    recapture = set(image_ids[border & (confidence < threshold)].tolist())
    item_status = [
        "IGNORED_RECAPTURE"
        if int(image_id) in recapture
        else "APPROVED"
        if confidence[index] >= threshold
        else "UNKNOWN"
        for index, image_id in enumerate(image_ids)
    ]
    return {
        "prediction": prediction.tolist(),
        "top3": np.argsort(-probabilities, axis=1)[:, :3].tolist(),
        "item_status": item_status,
        "classifier_border_recapture_image_ids": sorted(
            int(value) for value in recapture
        ),
    }


def evaluate_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    records_path = args.output_dir / "prepared" / "evaluation_records.jsonl"
    tensors_path = args.output_dir / "prepared" / "evaluation_tensors.npy"
    detector_report = json.loads(
        (args.output_dir / "prepared" / "detector_report.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for sample_size in config["experiment"]["sample_sizes"]:
        run_dir = args.output_dir / "runs" / f"n{sample_size}"
        report_path = run_dir / "onnx_evaluation.json"
        if args.resume and report_path.is_file():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing.get("checkpoint_sha256") == sha256_file(run_dir / "best.pt"):
                continue
        package_dir = args.output_dir / "packages" / f"n{sample_size}"
        calibration_report = json.loads(
            (run_dir / "calibration.json").read_text(encoding="utf-8")
        )
        fold_calibrations = {
            int(key): value for key, value in calibration_report["folds"].items()
        }
        cuda_reference = np.load(run_dir / "development_logits.npz")["logits"]
        provider_reports: dict[str, Any] = {}
        provider_logits: dict[str, np.ndarray] = {}
        for provider in ("cpu", "cuda"):
            reference = (
                _infer_checkpoint(run_dir / "best.pt", tensors_path, config, cpu=True)
                if provider == "cpu"
                else cuda_reference
            )
            candidate = _infer_onnx(
                package_dir,
                tensors_path,
                provider,
                args.cuda_dll_dir,
                int(config["training"]["batch_size"]),
            )
            difference = np.abs(reference - candidate)
            denominator = np.maximum(np.abs(reference), 1e-8)
            reference_signature = _classifier_signature(
                reference, rows, calibration_report["final"]
            )
            candidate_signature = _classifier_signature(
                candidate, rows, calibration_report["final"]
            )
            reference_top3 = np.asarray(reference_signature["top3"])
            candidate_top3 = np.asarray(candidate_signature["top3"])
            top1_mismatch = int(
                np.count_nonzero(reference_top3[:, 0] != candidate_top3[:, 0])
            )
            top3_order_mismatch = int(
                np.count_nonzero(np.any(reference_top3 != candidate_top3, axis=1))
            )
            top3_set_mismatch = int(
                np.count_nonzero(
                    np.any(
                        np.sort(reference_top3, axis=1)
                        != np.sort(candidate_top3, axis=1),
                        axis=1,
                    )
                )
            )
            parity = {
                "provider": provider,
                "sample_count": len(reference),
                "max_absolute_error": float(difference.max()),
                "max_relative_error": float((difference / denominator).max()),
                "tolerance": float(config["evaluation"]["classifier_tolerance"]),
                "within_tolerance": bool(
                    difference.max()
                    <= float(config["evaluation"]["classifier_tolerance"])
                ),
                "top1_mismatch_count": top1_mismatch,
                "top3_order_mismatch_count": top3_order_mismatch,
                "top3_set_mismatch_count": top3_set_mismatch,
                "top3_equal": top3_order_mismatch == 0,
                "state_equal": reference_signature["item_status"]
                == candidate_signature["item_status"]
                and reference_signature["classifier_border_recapture_image_ids"]
                == candidate_signature["classifier_border_recapture_image_ids"],
            }
            parity["passes"] = bool(
                parity["within_tolerance"]
                and parity["top3_equal"]
                and parity["state_equal"]
            )
            evaluation = _evaluate_crossfit(
                candidate, rows, detector_report, fold_calibrations, config
            )
            provider_reports[provider] = {"parity": parity, "evaluation": evaluation}
            provider_logits[provider] = candidate
        cross_difference = np.abs(provider_logits["cpu"] - provider_logits["cuda"])
        cpu_signature = _classifier_signature(
            provider_logits["cpu"], rows, calibration_report["final"]
        )
        cuda_signature = _classifier_signature(
            provider_logits["cuda"], rows, calibration_report["final"]
        )
        cpu_top3 = np.asarray(cpu_signature["top3"])
        cuda_top3 = np.asarray(cuda_signature["top3"])
        cross_top1_mismatch = int(np.count_nonzero(cpu_top3[:, 0] != cuda_top3[:, 0]))
        cross_top3_order_mismatch = int(
            np.count_nonzero(np.any(cpu_top3 != cuda_top3, axis=1))
        )
        cross_top3_set_mismatch = int(
            np.count_nonzero(
                np.any(np.sort(cpu_top3, axis=1) != np.sort(cuda_top3, axis=1), axis=1)
            )
        )
        cross_provider = {
            "max_absolute_error": float(cross_difference.max()),
            "tolerance": float(config["evaluation"]["cross_provider_tolerance"]),
            "within_tolerance": bool(
                cross_difference.max()
                <= float(config["evaluation"]["cross_provider_tolerance"])
            ),
            "top1_mismatch_count": cross_top1_mismatch,
            "top3_order_mismatch_count": cross_top3_order_mismatch,
            "top3_set_mismatch_count": cross_top3_set_mismatch,
            "top3_equal": cross_top3_order_mismatch == 0,
            "state_equal": cpu_signature["item_status"] == cuda_signature["item_status"]
            and cpu_signature["classifier_border_recapture_image_ids"]
            == cuda_signature["classifier_border_recapture_image_ids"],
        }
        cross_provider["passes"] = bool(
            cross_provider["within_tolerance"]
            and cross_provider["top3_equal"]
            and cross_provider["state_equal"]
        )
        _write_json(
            report_path,
            {
                "schema_version": SCHEMA_VERSION,
                "sample_size_per_class": int(sample_size),
                "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                "providers": provider_reports,
                "cross_provider": cross_provider,
                "test_accessed": False,
            },
        )


def benchmark_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.benchmark_images is None or not args.benchmark_images.is_dir():
        raise FileNotFoundError(
            "--benchmark-images is required for the benchmark phase"
        )
    targets = [("current", args.production_package)] + [
        (f"n{size}", args.output_dir / "packages" / f"n{size}")
        for size in config["experiment"]["sample_sizes"]
    ]
    for name, package_dir in targets:
        output = args.output_dir / "benchmarks" / f"{name}-cuda.json"
        if args.resume and output.is_file():
            continue
        command = [
            sys.executable,
            "-m",
            "bixolon_scanner.benchmark",
            "--package-dir",
            str(package_dir),
            "--images",
            str(args.benchmark_images),
            "--provider",
            "cuda",
            "--warmup",
            str(config["evaluation"]["benchmark_warmup"]),
            "--runs",
            str(config["evaluation"]["benchmark_runs"]),
            "--output",
            str(output),
        ]
        if args.cuda_dll_dir is not None:
            command.extend(["--cuda-dll-dir", str(args.cuda_dll_dir)])
        subprocess.run(command, check=True, cwd=Path.cwd())


def _optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_columns(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None}
    full = report.get("by_path", {}).get("full_path", report)
    return {
        "latency_p50_ms": full.get("p50_ms"),
        "latency_p95_ms": full.get("p95_ms"),
        "latency_p99_ms": full.get("p99_ms"),
    }


def _parity_columns(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "cpu_parity_pass": None,
            "cuda_parity_pass": None,
            "cross_provider_parity_pass": None,
            "cuda_top3_order_mismatch_count": None,
            "cross_provider_top3_order_mismatch_count": None,
        }
    providers = report.get("providers", {})
    cpu = providers.get("cpu", {}).get("parity", {})
    cuda = providers.get("cuda", {}).get("parity", {})
    cross = report.get("cross_provider", {})
    return {
        "cpu_parity_pass": cpu.get("passes"),
        "cuda_parity_pass": cuda.get("passes"),
        "cross_provider_parity_pass": cross.get("passes"),
        "cuda_top3_order_mismatch_count": cuda.get("top3_order_mismatch_count"),
        "cross_provider_top3_order_mismatch_count": cross.get(
            "top3_order_mismatch_count"
        ),
    }


def report_all(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    prepared = json.loads(
        (args.output_dir / "prepared" / "summary.json").read_text(encoding="utf-8")
    )
    current_counts = prepared["current_counts"]
    current_oof = _optional_json(args.current_oof_report)
    current_test = _optional_json(args.current_test_report)
    current_benchmark = _optional_json(args.current_benchmark_report)
    current_detector = prepared["detector"]
    fresh_current_benchmark = _optional_json(
        args.output_dir / "benchmarks" / "current-cuda.json"
    )
    if fresh_current_benchmark is not None:
        current_benchmark = fresh_current_benchmark

    comparison: list[dict[str, Any]] = []
    if current_oof is not None:
        comparison.append(
            {
                "condition": "current_historical_oof",
                "training_strategy": "historical_partial_finetune_with_development_roi",
                "sample_size_per_class": "119-138",
                "total_train_samples": sum(
                    int(row["current_final_train_total"]) for row in current_counts
                ),
                "evaluation_policy": "historical-current-oof-reference",
                "overall_top1_accuracy": current_oof.get("overall_top1_accuracy"),
                "overall_top3_accuracy": current_oof.get("overall_top3_accuracy"),
                "approved_precision": current_oof.get("approved_precision"),
                "approved_precision_ci_low": current_oof.get(
                    "approved_precision_95ci", [None, None]
                )[0],
                "approved_precision_ci_high": current_oof.get(
                    "approved_precision_95ci", [None, None]
                )[1],
                "approved_false_rate_upper_95": current_oof.get(
                    "approved_false_rate_upper_95"
                ),
                "approval_risk_control_satisfied": current_oof.get(
                    "risk_control_satisfied"
                ),
                "approval_coverage": current_oof.get("approval_coverage"),
                "unknown_count": current_oof.get("unknown_count"),
                "unknown_top3_accuracy": current_oof.get("unknown_top3_accuracy"),
                "approved_image_precision": None,
                "detector_recall": current_detector.get("recall"),
                "detector_precision": current_detector.get("precision"),
                "detector_count_accuracy": current_detector.get("count_accuracy"),
                **_benchmark_columns(current_benchmark),
                **_parity_columns(None),
            }
        )
    for sample_size in config["experiment"]["sample_sizes"]:
        evaluation = json.loads(
            (
                args.output_dir
                / "runs"
                / f"n{sample_size}"
                / "crossfit_evaluation.json"
            ).read_text(encoding="utf-8")
        )
        benchmark = _optional_json(
            args.output_dir / "benchmarks" / f"n{sample_size}-cuda.json"
        )
        parity = _optional_json(
            args.output_dir / "runs" / f"n{sample_size}" / "onnx_evaluation.json"
        )
        comparison.append(
            {
                "condition": f"n{sample_size}",
                "training_strategy": str(
                    config["training"].get("strategy", "partial_finetune")
                ),
                "sample_size_per_class": int(sample_size),
                "total_train_samples": int(sample_size)
                * int(config["experiment"]["expected_num_classes"]),
                "evaluation_policy": evaluation["evaluation_policy"],
                "overall_top1_accuracy": evaluation["overall_top1_accuracy"],
                "overall_top3_accuracy": evaluation["overall_top3_accuracy"],
                "approved_precision": evaluation["approved_precision"],
                "approved_precision_ci_low": evaluation["approved_precision_95ci"][0],
                "approved_precision_ci_high": evaluation["approved_precision_95ci"][1],
                "approved_false_rate_upper_95": evaluation[
                    "approved_false_rate_upper_95"
                ],
                "approval_risk_control_satisfied": evaluation[
                    "approved_precision_gate_satisfied"
                ],
                "approval_coverage": evaluation["approval_coverage_of_detections"],
                "unknown_count": evaluation["unknown_count"],
                "unknown_top3_accuracy": evaluation["unknown_top3_accuracy"],
                "approved_image_precision": evaluation["frame_policy"][
                    "approved_image_precision"
                ],
                "detector_recall": evaluation["detector"]["recall"],
                "detector_precision": evaluation["detector"]["precision"],
                "detector_count_accuracy": evaluation["detector"]["count_accuracy"],
                **_benchmark_columns(benchmark),
                **_parity_columns(parity),
            }
        )

    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    difficulty_comparison = _optional_json(
        reports_dir / "bread_project_2" / "comparison-bread-project-2-emh.json"
    )
    with (reports_dir / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (reports_dir / "current_training_counts.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(current_counts[0]))
        writer.writeheader()
        writer.writerows(current_counts)

    environment = {
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "weights_sha256": sha256_file(args.weights),
        "production_package_metadata_sha256": sha256_file(
            args.production_package / "metadata.json"
        ),
        "production_detector_sha256": sha256_file(
            args.production_package / "detector.onnx"
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "bread_dino_data_scale",
        "selection_policy": "compare_only_no_automatic_recommendation",
        "sample_sizes": config["experiment"]["sample_sizes"],
        "seed": config["experiment"]["seed"],
        "training_pool": "classification_aux_only",
        "training_strategy": str(
            config["training"].get("strategy", "partial_finetune")
        ),
        "training_config": config["training"],
        "current_training": {
            "auxiliary_total": sum(
                int(row["auxiliary_images"]) for row in current_counts
            ),
            "development_roi_total": sum(
                int(row["development_rois"]) for row in current_counts
            ),
            "final_train_total": sum(
                int(row["current_final_train_total"]) for row in current_counts
            ),
            "counts": current_counts,
        },
        "comparison": comparison,
        "historical_current_test_reference": current_test,
        "bread_project_2_overall_comparison": (
            [row for row in difficulty_comparison["rows"] if row["difficulty"] == "ALL"]
            if difficulty_comparison is not None
            else None
        ),
        "environment": environment,
        "limitations": [
            "각 N 조건은 seed 20260810 단일 실행이며 seed 간 분산을 추정하지 않는다.",
            "조건 비교에는 classifier 학습에 사용하지 않은 development 이미지를 사용한다.",
            "운영 detector는 조건 간 고정되지만 development detector 독립성 승격 근거는 아니다.",
            "최종 test 94장은 이번 비교에서 열지 않았다.",
            "CPU/CUDA Worker 상태는 같았지만 후보 순위가 달랐고 N=20은 Top-1 1건도 달라 strict parity를 통과하지 못했다.",
            "동시점 current와 모든 N 조건의 RTX 5080 full-path p95가 100ms를 초과해 성능 게이트를 통과하지 못했다.",
        ],
        "test_accessed": False,
        "selected_n": None,
        "promotion_status": "experiment_only",
    }
    _write_json(reports_dir / "summary.json", summary)
    _write_json(reports_dir / "environment.json", environment)

    lines = [
        "# 빵 DINO 학습 데이터량 실험",
        "",
        "- 목적: 종류별 단품 사진 5·10·15·20장 조건의 Worker 판정 성능과 지연 비교",
        "- 선택: 자동 추천 없음",
        "- Seed: `20260810` 단일 실행",
        f"- 학습법: `{config['training'].get('strategy', 'partial_finetune')}`",
        "- 최종 test: 접근하지 않음",
        "",
        "## 현재 운영 classifier 학습량",
        "",
        "| 종류 | 단품 | development ROI | 최종 학습 합계 |",
        "|---|---:|---:|---:|",
    ]
    for row in current_counts:
        lines.append(
            f"| {row['class_id']} {row['class_name']} | {row['auxiliary_images']} | "
            f"{row['development_rois']} | {row['current_final_train_total']} |"
        )
    lines.extend(
        [
            "",
            "## 조건 비교",
            "",
            f"고정 detector: recall {current_detector['recall']:.4%}, precision {current_detector['precision']:.4%}, count accuracy {current_detector['count_accuracy']:.4%}",
            "",
            "| 조건 | 총 학습 | Top-1 | Top-3 | 승인 precision (95% CI) | risk-control | 승인 coverage | UNKNOWN Top-3 | full-path p95 | CPU parity | CUDA parity | CPU/CUDA parity |",
            "|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|:---:|:---:|",
        ]
    )
    for row in comparison:
        unknown = row["unknown_top3_accuracy"]
        p95 = row["latency_p95_ms"]
        precision_ci = (
            "N/A"
            if row["approved_precision_ci_low"] is None
            else (
                f"{row['approved_precision_ci_low']:.2%}–"
                f"{row['approved_precision_ci_high']:.2%}"
            )
        )
        lines.append(
            f"| {row['condition']} | {row['total_train_samples']} | "
            f"{row['overall_top1_accuracy']:.4%} | {row['overall_top3_accuracy']:.4%} | "
            f"{row['approved_precision']:.4%} ({precision_ci}) | "
            f"{row['approval_risk_control_satisfied']} | "
            f"{row['approval_coverage']:.4%} | "
            f"{'N/A' if unknown is None else f'{unknown:.4%}'} | "
            f"{'N/A' if p95 is None else f'{p95:.3f} ms'} | "
            f"{row['cpu_parity_pass'] if row['cpu_parity_pass'] is not None else 'N/A'} | "
            f"{row['cuda_parity_pass'] if row['cuda_parity_pass'] is not None else 'N/A'} | "
            f"{row['cross_provider_parity_pass'] if row['cross_provider_parity_pass'] is not None else 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "현재 운영 행은 기존 OOF/test 보고서의 참고값이며, N 조건은 development 3-fold 교차 보정 결과입니다.",
            "모든 N의 PyTorch/CPU ONNX parity는 통과했습니다. CPU/CUDA Worker 상태는 같지만 후보 순위가 달랐고 N=20은 Top-1 1건도 달라 strict parity를 통과하지 못했습니다.",
            "동시점 current와 모든 N 조건의 RTX 5080 full-path p95가 100 ms를 초과해 성능 게이트를 통과하지 못했습니다.",
            "어떤 N도 자동 선택하거나 production으로 승격하지 않았습니다.",
        ]
    )
    if difficulty_comparison is not None:
        lines.extend(
            [
                "",
                "## bread_project_2 E/M/H 동일 Worker 경로 비교",
                "",
                "이 데이터는 300장 중 299장이 기존 정책 적합 세트와 중복되므로 독립 test가 아닙니다.",
                "",
                "| 조건 | Top-1 | UNKNOWN Top-3 | Candidate out | APPROVED/UNKNOWN/RECAPTURE | 평균 지연 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in difficulty_comparison["rows"]:
            if row["difficulty"] != "ALL":
                continue
            top1 = row["classifier_top1_accuracy_excluding_recapture"]
            top3 = row["unknown_top3_accuracy"]
            lines.append(
                f"| {row['condition']} | "
                f"{'N/A' if top1 is None else f'{top1:.4%}'} | "
                f"{'N/A' if top3 is None else f'{top3:.4%}'} | "
                f"{row['candidate_out']} | {row['approved_images']}/"
                f"{row['unknown_images']}/{row['recapture_images']} | "
                f"{row['mean_latency_ms']:.3f} ms |"
            )
        lines.extend(
            [
                "",
                "난이도별 E/M/H 전체 행과 오류 상세는 `bread_project_2/comparison-bread-project-2-emh.md` 및 CSV/JSON을 참고하십시오.",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _verify_outputs(args: argparse.Namespace, config: dict[str, Any]) -> None:
    production_detector = sha256_file(args.production_package / "detector.onnx")
    for sample_size in config["experiment"]["sample_sizes"]:
        package_dir = args.output_dir / "packages" / f"n{sample_size}"
        package = load_model_package(package_dir)
        if sha256_file(package.detector_path) != production_detector:
            raise RuntimeError(f"experimental detector differs for n={sample_size}")
        required = [
            args.output_dir / "runs" / f"n{sample_size}" / "best.pt",
            args.output_dir / "runs" / f"n{sample_size}" / "calibration.json",
            args.output_dir / "runs" / f"n{sample_size}" / "crossfit_evaluation.json",
            args.output_dir / "runs" / f"n{sample_size}" / "onnx_evaluation.json",
            args.output_dir / "benchmarks" / f"n{sample_size}-cuda.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"n={sample_size} missing artifacts: {missing}")
    review = json.loads(
        (args.output_dir / "prepared" / "selection_review.json").read_text(
            encoding="utf-8"
        )
    )
    if review["status"] != "approved" or review["test_accessed"]:
        raise RuntimeError("selection is not approved or test was accessed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bread DINO classifier data-scale experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--production-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier-cache-dir", type=Path)
    parser.add_argument("--benchmark-images", type=Path)
    parser.add_argument("--current-oof-report", type=Path)
    parser.add_argument("--current-test-report", type=Path)
    parser.add_argument("--current-benchmark-report", type=Path)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--approve-selection", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _torch_cuda_dll_dir() -> Path | None:
    torch = require_torch()
    candidate = Path(torch.__file__).resolve().parent / "lib"
    return candidate if (candidate / "cublasLt64_13.dll").is_file() else None


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.provider is None:
        args.provider = str(config["evaluation"]["provider"])
    if args.cuda_dll_dir is None:
        args.cuda_dll_dir = _torch_cuda_dll_dir()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    phases = PHASES if args.phase == "all" else (args.phase,)
    if "prepare" in phases:
        prepare(args, config)
    if "train" in phases:
        train_all(args, config)
    if "calibrate" in phases:
        calibrate_all(args, config)
    if "export" in phases:
        export_all(args, config)
    if "evaluate" in phases:
        evaluate_all(args, config)
    if "benchmark" in phases:
        benchmark_all(args, config)
    if "report" in phases:
        _verify_outputs(args, config)
        report_all(args, config)


if __name__ == "__main__":
    main()
