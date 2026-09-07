from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_source(value: str) -> tuple[Path, str]:
    path_text, separator, key = value.rpartition(":")
    if not separator or not path_text or not key:
        raise argparse.ArgumentTypeError("score source must be PATH:ARRAY_KEY")
    return Path(path_text), key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one global convex ensemble of classifier score matrices"
    )
    parser.add_argument("--source", type=_parse_source, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.35)
    parser.add_argument("--local-concentration", type=float, default=200.0)
    parser.add_argument("--signed-local-sigma", type=float, default=0.0)
    parser.add_argument("--initial-weight", type=float, nargs="+")
    args = parser.parse_args()

    matrices = []
    labels = None
    identity = None
    for path, key in args.source:
        payload = np.load(path)
        matrix = np.asarray(payload[key], dtype=np.float32)
        source_labels = np.asarray(payload["labels"], dtype=np.int64)
        if source_labels.size and source_labels.min() == 1 and source_labels.max() == 20:
            source_labels = source_labels - 1
        source_identity = np.stack([payload["image_ids"], payload["annotation_ids"]], axis=1)
        if matrix.shape != (len(source_labels), 20):
            raise ValueError(f"invalid score matrix shape for {path}:{key}: {matrix.shape}")
        if labels is None:
            labels = source_labels
            identity = source_identity
        elif not np.array_equal(labels, source_labels) or not np.array_equal(
            identity, source_identity
        ):
            raise ValueError("score sources disagree on labels or object ordering")
        matrices.append(matrix)
    assert labels is not None
    stacked = np.stack(matrices, axis=0)
    source_count = len(matrices)
    if args.initial_weight is not None:
        if len(args.initial_weight) != source_count:
            raise ValueError("initial weight count must match score source count")
        initial = np.asarray(args.initial_weight, dtype=np.float64)
        if np.any(initial < 0) or initial.sum() <= 0:
            raise ValueError("initial weights must be non-negative with a positive sum")
        initial /= initial.sum()
    else:
        initial = np.full(source_count, 1.0 / source_count, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    best_weights = initial
    best_scores = np.tensordot(best_weights, stacked, axes=(0, 0))
    best_correct = int((best_scores.argmax(axis=1) == labels).sum())
    evaluated = 1
    remaining = args.samples
    while remaining > 0:
        count = min(args.batch_size, remaining)
        global_count = count // 2
        global_weights = rng.dirichlet(
            np.full(source_count, args.dirichlet_alpha, dtype=np.float64),
            size=global_count,
        )
        local_count = count - global_count
        if args.signed_local_sigma:
            local_weights = best_weights[None, :] + rng.normal(
                0.0, args.signed_local_sigma, size=(local_count, source_count)
            )
            sums = local_weights.sum(axis=1, keepdims=True)
            local_weights /= np.where(np.abs(sums) < 1e-9, 1.0, sums)
        else:
            local_weights = rng.dirichlet(
                best_weights * args.local_concentration + args.dirichlet_alpha,
                size=local_count,
            )
        weights = np.concatenate([global_weights, local_weights], axis=0)
        combined = np.einsum("bs,snc->bnc", weights, stacked, optimize=True)
        correct = (combined.argmax(axis=2) == labels[None, :]).sum(axis=1)
        index = int(correct.argmax())
        if int(correct[index]) > best_correct:
            best_correct = int(correct[index])
            best_weights = weights[index]
            best_scores = combined[index]
        evaluated += count
        remaining -= count

    predictions = best_scores.argmax(axis=1)
    errors = np.flatnonzero(predictions != labels)
    report = {
        "evaluation": "development_global_classifier_score_ensemble_selection",
        "evaluation_images_used_as_training_support": False,
        "seed": args.seed,
        "evaluated_weight_count": evaluated,
        "sources": [
            {"path": path.as_posix(), "array": key, "weight": float(weight)}
            for (path, key), weight in zip(args.source, best_weights, strict=True)
        ],
        "object_count": int(len(labels)),
        "correct_count": best_correct,
        "error_count": int(len(labels) - best_correct),
        "accuracy": best_correct / len(labels),
        "errors": [
            {
                "image_id": int(identity[index, 0]),
                "annotation_id": int(identity[index, 1]),
                "expected": int(labels[index] + 1),
                "predicted": int(predictions[index] + 1),
            }
            for index in errors
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
