from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.catalog import load_store_catalog_package, sha256_file
from ...runtime.catalog import l2_normalize


def _load_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.asarray(np.load(stream, allow_pickle=False), dtype=np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(exponential / exponential.sum(axis=1, keepdims=True), dtype=np.float32)


def _topk_message(
    similarities: np.ndarray,
    neighbor_values: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> np.ndarray:
    if k < 1 or temperature <= 0.0 or k > similarities.shape[1]:
        raise ValueError("invalid label-propagation neighbor configuration")
    indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(similarities, indices, axis=1) / np.float32(temperature)
    weights = _softmax(selected)
    return np.sum(neighbor_values[indices] * weights[:, :, None], axis=1)


def propagate_labels(
    target_embeddings: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    adapter_weight: np.ndarray,
    adapter_bias: np.ndarray,
    *,
    class_count: int,
    neighbor_count: int = 10,
    temperature: float = 0.05,
    source_weight: float = 0.4,
    target_weight: float = 0.4,
    prior_weight: float = 0.2,
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if iterations < 1:
        raise ValueError("label propagation requires at least one iteration")
    if any(value < 0.0 for value in (source_weight, target_weight, prior_weight)) or not np.isclose(
        source_weight + target_weight + prior_weight, 1.0
    ):
        raise ValueError("label-propagation message weights must be non-negative and sum to one")
    target = l2_normalize(target_embeddings)
    support = l2_normalize(support_embeddings)
    if adapter_weight.shape != (target.shape[1], class_count):
        raise ValueError("adapter weight does not match target embeddings")
    if adapter_bias.shape != (class_count,):
        raise ValueError("adapter bias does not match the class count")
    if support_labels.shape != (len(support),) or np.any(
        (support_labels < 0) | (support_labels >= class_count)
    ):
        raise ValueError("support labels are invalid")
    prior = _softmax(target @ adapter_weight + adapter_bias)
    support_one_hot = np.eye(class_count, dtype=np.float32)[support_labels]
    source_similarities = target @ support.T
    target_similarities = target @ target.T
    np.fill_diagonal(target_similarities, -np.inf)
    source_message = _topk_message(
        source_similarities,
        support_one_hot,
        k=min(neighbor_count, len(support)),
        temperature=temperature,
    )
    probabilities = prior.copy()
    for _ in range(iterations):
        target_message = _topk_message(
            target_similarities,
            probabilities,
            k=min(neighbor_count, max(1, len(target) - 1)),
            temperature=temperature,
        )
        probabilities = (
            source_weight * source_message + target_weight * target_message + prior_weight * prior
        )
        probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    return prior, np.asarray(probabilities, dtype=np.float32)


def folded_target_prototype_probabilities(
    embeddings: np.ndarray,
    pseudo_labels: np.ndarray,
    folds: np.ndarray,
    *,
    class_count: int,
    temperature: float = 0.05,
) -> np.ndarray:
    if folds.shape != (len(embeddings),) or pseudo_labels.shape != (len(embeddings),):
        raise ValueError("target prototype inputs are not aligned")
    if temperature <= 0.0:
        raise ValueError("target prototype temperature must be positive")
    values = l2_normalize(embeddings)
    probabilities = np.empty((len(values), class_count), dtype=np.float32)
    for fold in sorted(set(folds.tolist())):
        held_out = folds == fold
        training = ~held_out
        prototypes = []
        for class_index in range(class_count):
            selected = values[training & (pseudo_labels == class_index)]
            if not len(selected):
                raise ValueError("a target prototype fold is missing a pseudo class")
            prototypes.append(selected.mean(axis=0))
        similarities = values[held_out] @ l2_normalize(np.stack(prototypes)).T
        probabilities[held_out] = _softmax(similarities / np.float32(temperature))
    return probabilities


def _targets(trace: Path, class_ids: dict[str, int]) -> np.ndarray:
    values = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["status"] != "SEGMENTATION":
            continue
        values.extend(
            class_ids[str(diagnostic["target_class_id"])]
            for diagnostic in row["matched_classifier_diagnostics"]
        )
    return np.asarray(values, dtype=np.int64)


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    return {
        "sample_count": len(targets),
        "top1_correct_count": int(np.count_nonzero(ranking[:, 0] == targets)),
        "top1_error_count": int(np.count_nonzero(ranking[:, 0] != targets)),
        "top3_miss_count": int(
            np.count_nonzero(~np.any(ranking[:, :3] == targets[:, None], axis=1))
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_store_catalog_package(args.catalog)
    with np.load(args.target_embeddings, allow_pickle=False) as payload:
        if set(payload.files) != {"raw_embeddings", "image_ids", "detection_indices"}:
            raise ValueError("unlabeled target embedding archive has unexpected arrays")
        embeddings = np.asarray(payload["raw_embeddings"], dtype=np.float32)
        image_ids = np.asarray(payload["image_ids"], dtype=np.int64)
        detection_indices = np.asarray(payload["detection_indices"], dtype=np.int64)
    supports = _load_array(catalog.supports_path)
    support_labels = np.concatenate(
        [
            np.full(label.support_count, index, dtype=np.int64)
            for index, label in enumerate(catalog.metadata.labels)
        ]
    )
    if catalog.adapter_path is None:
        raise ValueError("target label propagation requires the source-only Catalog adapter")
    with np.load(catalog.adapter_path, allow_pickle=False) as adapter:
        weight = np.asarray(adapter["weight"], dtype=np.float32)
        bias = np.asarray(adapter["bias"], dtype=np.float32)
    prior, propagated = propagate_labels(
        embeddings,
        supports,
        support_labels,
        weight,
        bias,
        class_count=len(catalog.metadata.labels),
        neighbor_count=args.neighbor_count,
        temperature=args.temperature,
        source_weight=args.source_weight,
        target_weight=args.target_weight,
        prior_weight=args.prior_weight,
        iterations=args.iterations,
    )
    manifest_folds = {
        int(row["image_id"]): int(row["fold"])
        for row in (
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    folds = np.asarray([manifest_folds[int(image_id)] for image_id in image_ids], dtype=np.int64)
    target_prototype = folded_target_prototype_probabilities(
        embeddings,
        propagated.argmax(axis=1),
        folds,
        class_count=len(catalog.metadata.labels),
        temperature=args.temperature,
    )
    fused = np.sqrt(prior * target_prototype).astype(np.float32)
    fused /= fused.sum(axis=1, keepdims=True).clip(min=1e-12)
    labels = catalog.metadata.labels
    ranking = np.argsort(-propagated, axis=1, kind="stable")
    args.pseudo_labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.pseudo_labels_output.write_text(
        "".join(
            json.dumps(
                {
                    "image_id": int(image_id),
                    "detection_index": int(detection_index),
                    "class_id": labels[int(indices[0])].class_id,
                    "confidence": float(propagated[row, indices[0]]),
                    "margin": float(propagated[row, indices[0]] - propagated[row, indices[1]]),
                },
                sort_keys=True,
            )
            + "\n"
            for row, (image_id, detection_index, indices) in enumerate(
                zip(image_ids, detection_indices, ranking, strict=True)
            )
        ),
        encoding="utf-8",
    )
    targets = _targets(args.trace, {label.class_id: i for i, label in enumerate(labels)})
    if len(targets) != len(embeddings):
        raise ValueError("diagnostic targets do not align with unlabeled embeddings")
    report = {
        "schema_version": "1.0",
        "evaluation": "fixed_unlabeled_target_label_propagation",
        "evidence_role": "full_dataset_development_diagnostic",
        "adaptation_inputs": {
            "target_product_labels": False,
            "target_embeddings_sha256": sha256_file(args.target_embeddings),
            "catalog_sha256": sha256_file(args.catalog / "catalog.json"),
        },
        "policy": {
            "neighbor_count": args.neighbor_count,
            "temperature": args.temperature,
            "source_weight": args.source_weight,
            "target_weight": args.target_weight,
            "prior_weight": args.prior_weight,
            "iterations": args.iterations,
        },
        "source_adapter": _metrics(prior, targets),
        "label_propagation": _metrics(propagated, targets),
        "folded_target_prototype": _metrics(target_prototype, targets),
        "source_target_geometric_mean": _metrics(fused, targets),
        "pseudo_labels_sha256": sha256_file(args.pseudo_labels_output),
        "limitations": [
            "target labels are loaded only after pseudo-label generation for diagnostic reporting",
            "no target label, class pair, object count, difficulty, or image exception is used",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate source labels on unlabeled target ROIs")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--target-embeddings", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo-labels-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbor-count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--source-weight", type=float, default=0.4)
    parser.add_argument("--target-weight", type=float, default=0.4)
    parser.add_argument("--prior-weight", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=10)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
