from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def _rgb(tensor: np.ndarray) -> np.ndarray:
    values = tensor.transpose(1, 2, 0) * STD + MEAN
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _histogram(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    values = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    values = values.reshape(-1).astype(np.float32)
    return values / max(float(values.sum()), 1.0)


def handcrafted_features(tensor: np.ndarray) -> np.ndarray:
    image = cv2.resize(_rgb(tensor), (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog = HOG.compute(gray).reshape(-1).astype(np.float32)
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    background = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]
    foreground = (np.linalg.norm(lab - background_lab, axis=2) >= 16).astype(np.uint8) * 255
    grid = []
    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            patch = image[y : y + 16, x : x + 16].reshape(-1, 3).astype(np.float32) / 255.0
            grid.extend(patch.mean(axis=0))
            grid.extend(patch.std(axis=0))
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= (neighbor >= center).astype(np.uint8) << bit
    lbp_hist = np.bincount(lbp.reshape(-1), minlength=256).astype(np.float32)
    lbp_hist /= lbp_hist.sum()
    moments = cv2.moments(foreground)
    hu = np.sign(cv2.HuMoments(moments).reshape(-1)) * np.log1p(
        np.abs(cv2.HuMoments(moments).reshape(-1))
    )
    shape = np.asarray(
        [
            float(np.count_nonzero(foreground)) / foreground.size,
            float(np.count_nonzero(foreground[0])) / 64,
            float(np.count_nonzero(foreground[-1])) / 64,
            float(np.count_nonzero(foreground[:, 0])) / 64,
            float(np.count_nonzero(foreground[:, -1])) / 64,
            *hu,
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        (
            hog / max(float(np.linalg.norm(hog)), 1e-12),
            _histogram(image, None),
            _histogram(image, foreground),
            np.asarray(grid, dtype=np.float32),
            lbp_hist,
            shape,
        )
    )
    return features / max(float(np.linalg.norm(features)), 1e-12)


def _feature_matrix(tensors: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    selected = range(len(tensors)) if indices is None else indices
    parts = []
    for offset, index in enumerate(selected, start=1):
        parts.append(handcrafted_features(tensors[int(index)]))
        if offset % 1000 == 0:
            print(json.dumps({"extracted_features": offset}), flush=True)
    return np.asarray(parts, dtype=np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=1, keepdims=True)) / values.std(axis=1, keepdims=True).clip(
        1e-6
    )


def _fused_predictions(
    dino: np.ndarray, handcrafted: np.ndarray, *, weight: float, top3_only: bool
) -> np.ndarray:
    dino_scores = _zscore(dino)
    crafted_scores = _zscore(handcrafted)
    if top3_only:
        candidates = np.argsort(-dino, axis=1, kind="stable")[:, :3]
        mask = np.ones_like(crafted_scores, dtype=bool)
        mask[np.arange(len(mask))[:, None], candidates] = False
        crafted_scores[mask] = -100.0
    return (weight * dino_scores + (1.0 - weight) * crafted_scores).argmax(axis=1)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.svm import LinearSVC

    training_tensors = np.load(args.training_tensors, mmap_mode="r")
    training_cache = np.load(args.training_cache)
    labels = training_cache["labels"].astype(np.int64)
    if len(training_tensors) != len(labels) or len(training_tensors) % 200:
        raise ValueError("training cache is not aligned to the 200 originals")
    views_per_source = len(training_tensors) // 200
    indices = np.asarray(
        [
            index
            for index in range(len(training_tensors))
            if index % views_per_source < args.views_per_source
        ],
        dtype=np.int64,
    )
    training_features = _feature_matrix(training_tensors, indices)
    evaluation_tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    evaluation_features = _feature_matrix(evaluation_tensors)
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    dino_loaded = np.load(args.dino_logits)
    dino = dino_loaded[args.dino_view].astype(np.float32)
    if dino.shape != (len(targets), 20):
        raise ValueError("DINO logits are not aligned with evaluation records")
    candidates = []
    predictions_by_name = {}
    for regularization_c in args.svm_c:
        classifier = LinearSVC(C=regularization_c, dual="auto", max_iter=20_000)
        classifier.fit(training_features, labels[indices])
        scores = classifier.decision_function(evaluation_features).astype(np.float32)
        for top3_only in (False, True):
            for weight in np.linspace(0.5, 1.0, 51):
                predictions = _fused_predictions(
                    dino, scores, weight=float(weight), top3_only=top3_only
                )
                name = f"c{regularization_c:g}:weight{weight:.2f}:top3{top3_only}"
                predictions_by_name[name] = predictions
                candidates.append(
                    {
                        "name": name,
                        "svm_c": regularization_c,
                        "dino_weight": float(weight),
                        "top3_only": top3_only,
                        "top1_accuracy": float((predictions == targets).mean()),
                    }
                )
    candidates.sort(key=lambda row: (row["top1_accuracy"], row["name"]), reverse=True)
    oof = np.empty_like(targets)
    fold_results = []
    for fold in range(3):
        selection = folds != fold
        held_out = folds == fold
        selected = max(
            candidates,
            key=lambda row: float(
                (predictions_by_name[row["name"]][selection] == targets[selection]).mean()
            ),
        )
        predictions = predictions_by_name[selected["name"]]
        oof[held_out] = predictions[held_out]
        fold_results.append(
            {
                "held_out_fold": fold,
                "selected": selected,
                "held_out_top1_accuracy": float(
                    (predictions[held_out] == targets[held_out]).mean()
                ),
            }
        )
    baseline_accuracy = float((dino.argmax(axis=1) == targets).mean())
    report = {
        "schema_version": "1.0",
        "evaluation": "handcrafted_top3_reranker_probe",
        "promotion_status": "diagnostic_only",
        "training_contract": {
            "source_original_count": 200,
            "derived_training_tensor_count": len(indices),
            "evaluation_images_used_for_training": False,
        },
        "feature_count": training_features.shape[1],
        "baseline_top1_accuracy": baseline_accuracy,
        "selected": candidates[0],
        "top_candidates": candidates[:20],
        "grouped_3fold_oof": {
            "top1_accuracy": float((oof == targets).mean()),
            "folds": fold_results,
        },
        "passes_top1_gate": candidates[0]["top1_accuracy"] >= 0.99,
        "passes_grouped_oof_top1_gate": float((oof == targets).mean()) >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a low-cost handcrafted Top-3 reranker")
    parser.add_argument("--training-tensors", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--dino-logits", type=Path, required=True)
    parser.add_argument("--dino-view", default="base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-per-source", type=int, default=32)
    parser.add_argument("--svm-c", type=float, nargs="+", default=(0.001, 0.01, 0.1, 1.0))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
