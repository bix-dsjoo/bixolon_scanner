from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ...contracts.catalog import sha256_file
from ...pipeline.ports import Detection
from ...runtime.catalog import MetricTransform, l2_normalize
from ...runtime.onnx import OrtRunner, classifier_crop_box, prepare_rgb


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _images(
    dataset_root: Path,
    classifier_records: list[dict],
    detector_records: list[dict],
) -> tuple[Iterable[tuple[Image.Image, int, int, int]], dict[int, tuple[str, str]]]:
    category_map = {
        int(row["category_id"]): (str(row["class_id"]), str(row["class_name"]))
        for row in classifier_records
    }

    def generate() -> Iterator[tuple[Image.Image, int, int, int]]:
        for row in classifier_records:
            with Image.open(dataset_root / row["image_path"]) as source:
                yield source.convert("RGB"), int(row["category_id"]), -1, -1
        for row in detector_records:
            with Image.open(dataset_root / row["image_path"]) as source:
                image = source.convert("RGB")
                for annotation in row["annotations"]:
                    x, y, width, height = annotation["bbox_xywh"]
                    box = classifier_crop_box(
                        Detection(x, y, x + width, y + height, 1.0),
                        image.width,
                        image.height,
                        margin_ratio=0.05,
                        crop_mode="square_context",
                    )
                    yield (
                        image.crop(box),
                        int(annotation["category_id"]),
                        int(row["fold"]),
                        int(row["image_id"]),
                    )

    return generate(), category_map


def _extract(
    runner: OrtRunner,
    samples: Iterable[tuple[Image.Image, int, int, int]],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_parts = []
    targets: list[int] = []
    folds: list[int] = []
    image_ids: list[int] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        if not batch:
            return
        (values,) = runner.run(
            ["embeddings"], "pixel_values", np.stack(batch).astype(np.float32, copy=False)
        )
        feature_parts.append(l2_normalize(np.asarray(values, dtype=np.float32)))
        batch.clear()

    for image, target, fold, image_id in samples:
        batch.append(
            prepare_rgb(
                image,
                (224, 224),
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                reducing_gap=3.0,
            )
        )
        targets.append(target)
        folds.append(fold)
        image_ids.append(image_id)
        if len(batch) >= batch_size:
            flush()
    flush()
    return (
        np.concatenate(feature_parts),
        np.asarray(targets, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
        np.asarray(image_ids, dtype=np.int64),
    )


def _fit_lda(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    estimator = LinearDiscriminantAnalysis(solver="svd", n_components=len(np.unique(targets)) - 1)
    estimator.fit(features, targets)
    mean = np.asarray(estimator.xbar_, dtype=np.float32)[None]
    matrix = np.asarray(estimator.scalings_[:, : estimator._max_components], dtype=np.float32)
    return mean, matrix


def _transform(
    features: np.ndarray,
    mean: np.ndarray,
    matrix: np.ndarray,
    residual_weight: float,
    projection_weight: float,
) -> np.ndarray:
    return MetricTransform(
        input_dimension=features.shape[1],
        residual_weight=residual_weight,
        projection_weight=projection_weight,
        mean=mean,
        matrix=matrix,
    ).apply(features)


def _scores(
    queries: np.ndarray,
    supports: np.ndarray,
    support_targets: np.ndarray,
    category_ids: np.ndarray,
    *,
    prototype_weight: float,
    top_k: int,
) -> np.ndarray:
    prototypes = l2_normalize(
        np.stack([supports[support_targets == category].mean(axis=0) for category in category_ids])
    )
    output = np.empty((len(queries), len(category_ids)), dtype=np.float32)
    for class_index, category in enumerate(category_ids):
        values = queries @ supports[support_targets == category].T
        count = min(top_k, values.shape[1])
        nearest = np.partition(values, values.shape[1] - count, axis=1)[:, -count:].mean(axis=1)
        output[:, class_index] = (
            prototype_weight * (queries @ prototypes[class_index])
            + (1.0 - prototype_weight) * nearest
        )
    return output


def _approval_thresholds(
    scores: np.ndarray, targets: np.ndarray, category_ids: np.ndarray
) -> tuple[float, float, int]:
    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.arange(len(scores))
    top1 = scores[rows, order[:, 0]]
    margin = top1 - scores[rows, order[:, 1]]
    correct = category_ids[order[:, 0]] == targets
    wrong_top1 = top1[~correct]
    similarity_candidates = np.unique(
        np.concatenate(
            (
                [float(top1.min())],
                np.nextafter(wrong_top1, np.float32(2.0)),
                [np.nextafter(top1.max(), np.float32(2.0))],
            )
        )
    )
    best = (0, float(similarity_candidates[-1]), float(np.nextafter(margin.max(), 2.0)))
    for similarity in similarity_candidates:
        similarity_mask = top1 >= similarity
        unsafe_margins = margin[similarity_mask & ~correct]
        minimum_margin = (
            0.0
            if not len(unsafe_margins)
            else float(np.nextafter(unsafe_margins.max(), np.float32(2.0)))
        )
        approved = similarity_mask & (margin >= minimum_margin)
        correct_count = int(np.count_nonzero(approved & correct))
        candidate = (correct_count, -float(similarity), -float(minimum_margin))
        current = (best[0], -best[1], -best[2])
        if candidate > current:
            best = (correct_count, float(similarity), float(minimum_margin))
    return best[1], best[2], best[0]


def _policy_metrics(scores: np.ndarray, targets: np.ndarray, category_ids: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = category_ids[order]
    top1_correct = ranked[:, 0] == targets
    top3_correct = np.any(ranked[:, :3] == targets[:, None], axis=1)
    rows = np.arange(len(scores))
    top1 = scores[rows, order[:, 0]]
    margin = top1 - scores[rows, order[:, 1]]
    third = scores[rows, order[:, 2]]
    similarity, minimum_margin, approved = _approval_thresholds(scores, targets, category_ids)
    candidate_out = ~top3_correct
    top3_minimum = (
        float(np.nextafter(third[candidate_out].max(), np.float32(2.0)))
        if np.any(candidate_out)
        else float(third.min())
    )
    ood_maximum = float(min(np.quantile(top1, 0.01) - 0.02, similarity - 1e-4))
    return {
        "top1_correct": int(np.count_nonzero(top1_correct)),
        "top3_correct": int(np.count_nonzero(top3_correct)),
        "sample_count": len(scores),
        "zero_error_approved_count": approved,
        "approval_minimum_similarity": similarity,
        "approval_minimum_margin": minimum_margin,
        "ood_maximum_similarity": max(-1.0, ood_maximum),
        "top3_minimum_similarity": top3_minimum,
        "top1_similarity_minimum": float(top1.min()),
        "top1_margin_minimum": float(margin.min()),
    }


def fit_catalog_metric(
    dataset_root: Path,
    classifier_manifest: Path,
    detector_manifest: Path,
    embedder_path: Path,
    output_dir: Path,
    *,
    provider: str,
    batch_size: int,
    cuda_dll_dir: Path | None = None,
) -> dict:
    classifier_records = _records(classifier_manifest)
    detector_records = _records(detector_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "raw-embeddings.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            features = cache["features"].copy()
            targets = cache["targets"].copy()
            folds = cache["folds"].copy()
            image_ids = cache["image_ids"].copy()
    else:
        runner = OrtRunner(embedder_path, provider, cuda_dll_dir)
        samples, _ = _images(dataset_root, classifier_records, detector_records)
        features, targets, folds, image_ids = _extract(runner, samples, batch_size=batch_size)
        np.savez_compressed(
            cache_path,
            features=features,
            targets=targets,
            folds=folds,
            image_ids=image_ids,
        )
    support_mask = folds == -1
    query_mask = ~support_mask
    supports = features[support_mask]
    support_targets = targets[support_mask]
    queries = features[query_mask]
    query_targets = targets[query_mask]
    query_folds = folds[query_mask]
    category_ids = np.asarray(sorted(set(support_targets)), dtype=np.int64)
    fold_lda = {}
    for fold in sorted(set(query_folds)):
        train = query_folds != fold
        fold_lda[int(fold)] = _fit_lda(
            np.concatenate((supports, queries[train])),
            np.concatenate((support_targets, query_targets[train])),
        )
    candidates = []
    feature_weights = (
        (1.0, 0.0),
        (1.0, 0.05),
        (1.0, 0.1),
        (1.0, 0.25),
        (1.0, 0.5),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    for residual_weight, projection_weight in feature_weights:
        transformed_by_fold = {}
        for fold in sorted(set(query_folds)):
            validation = query_folds == fold
            mean, matrix = fold_lda[int(fold)]
            transformed_by_fold[int(fold)] = (
                validation,
                _transform(supports, mean, matrix, residual_weight, projection_weight),
                _transform(queries[validation], mean, matrix, residual_weight, projection_weight),
            )
        for prototype_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for top_k in (1, 3, 5):
                fold_scores = np.empty((len(queries), len(category_ids)), dtype=np.float32)
                for (
                    validation,
                    transformed_supports,
                    transformed_queries,
                ) in transformed_by_fold.values():
                    fold_scores[validation] = _scores(
                        transformed_queries,
                        transformed_supports,
                        support_targets,
                        category_ids,
                        prototype_weight=prototype_weight,
                        top_k=top_k,
                    )
                metrics = _policy_metrics(fold_scores, query_targets, category_ids)
                candidates.append(
                    {
                        "residual_weight": residual_weight,
                        "projection_weight": projection_weight,
                        "prototype_weight": prototype_weight,
                        "support_top_k": top_k,
                        **metrics,
                    }
                )
    selected = max(
        candidates,
        key=lambda row: (
            row["zero_error_approved_count"],
            row["top3_correct"],
            row["top1_correct"],
            -row["residual_weight"],
        ),
    )
    final_mean, final_matrix = _fit_lda(
        np.concatenate((supports, queries)), np.concatenate((support_targets, query_targets))
    )
    projection_path = output_dir / "metric-projection.bin"
    with projection_path.open("wb") as stream:
        np.savez(stream, mean=final_mean, matrix=final_matrix)
    report = {
        "schema_version": "2.0",
        "method": "3-fold group-aware OOF policy selection; final LDA fit on 300 development images",
        "promotion_evidence": False,
        "classifier_manifest_sha256": sha256_file(classifier_manifest),
        "detector_manifest_sha256": sha256_file(detector_manifest),
        "embedder_sha256": sha256_file(embedder_path),
        "support_count": int(np.count_nonzero(support_mask)),
        "query_count": int(np.count_nonzero(query_mask)),
        "image_count": len(set(image_ids[query_mask].tolist())),
        "selected": selected,
        "metric_projection": {
            "input_dimension": features.shape[1],
            "output_dimension": final_matrix.shape[1],
            "sha256": sha256_file(projection_path),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the shared 2.0 Catalog metric policy")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--classifier-manifest", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-dll-dir", type=Path)
    args = parser.parse_args(argv)
    report = fit_catalog_metric(
        args.dataset_root,
        args.classifier_manifest,
        args.detector_manifest,
        args.embedder,
        args.output_dir,
        provider=args.provider,
        batch_size=args.batch_size,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
