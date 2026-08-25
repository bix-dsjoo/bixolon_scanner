from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.catalog import load_store_catalog_package, sha256_file
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.ports import Detection
from ...runtime.catalog import OnnxCatalogClassifier, OnnxEmbedder


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _manifest_index(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in _jsonl(path)}


def _ranking(result: Any) -> np.ndarray:
    return np.argsort(-result.ranking_logits, axis=1, kind="stable")


def _metrics(
    ranking: np.ndarray,
    targets: np.ndarray,
    *,
    approved: np.ndarray | None = None,
) -> dict[str, Any]:
    approved = np.ones(len(targets), dtype=bool) if approved is None else approved
    top1_correct = ranking[:, 0] == targets
    top3_hit = np.any(ranking[:, :3] == targets[:, None], axis=1)
    return {
        "sample_count": len(targets),
        "approved_count": int(np.count_nonzero(approved)),
        "correct_approved_count": int(np.count_nonzero(approved & top1_correct)),
        "approved_misrecognition_count": int(np.count_nonzero(approved & ~top1_correct)),
        "correct_approved_rate": float(np.count_nonzero(approved & top1_correct) / len(targets)),
        "top1_error_count": int(np.count_nonzero(~top1_correct)),
        "top3_miss_count": int(np.count_nonzero(~top3_hit)),
        "rejected_count": int(np.count_nonzero(~approved)),
        "rejected_top3_miss_count": int(np.count_nonzero(~approved & ~top3_hit)),
    }


def _detections(trace: dict[str, Any]) -> list[Detection]:
    diagnostics = trace["matched_classifier_diagnostics"]
    segmentations = trace["decision"]["segmentations"]
    if len(diagnostics) != len(segmentations):
        raise ValueError("trace segmentations and classifier diagnostics are not aligned")
    output = []
    for segmentation, diagnostic in zip(segmentations, diagnostics, strict=True):
        box = segmentation["bbox"]
        output.append(
            Detection(
                float(box["x"]),
                float(box["y"]),
                float(box["x"] + box["width"]),
                float(box["y"] + box["height"]),
                float(diagnostic["detector_score"]),
            )
        )
    return output


def _targets(trace: dict[str, Any], class_ids: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [class_ids[str(row["target_class_id"])] for row in trace["matched_classifier_diagnostics"]],
        dtype=np.int64,
    )


def _trace_top1(trace: dict[str, Any], class_ids: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [
            class_ids[str(row["classifier_top1_class_id"])]
            for row in trace["matched_classifier_diagnostics"]
        ],
        dtype=np.int64,
    )


def _consensus_masks(view_top1: np.ndarray, ensemble_top1: np.ndarray) -> dict[str, np.ndarray]:
    votes_for_ensemble = np.count_nonzero(view_top1 == ensemble_top1[None, :], axis=0)
    return {
        "all_four_views": votes_for_ensemble == 4,
        "at_least_three_views": votes_for_ensemble >= 3,
        "zero_and_180_degrees": (view_top1[0] == ensemble_top1) & (view_top1[2] == ensemble_top1),
    }


def _validate_trace_reproduction(
    mismatch_count: int, *, provider: str, allow_alternative_runtime: bool
) -> None:
    if mismatch_count and provider == "cpu" and not allow_alternative_runtime:
        raise ValueError(
            f"zero-degree runtime reproduction differs from trace for {mismatch_count} objects"
        )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(args.catalog)
    embedder = OnnxEmbedder(
        runtime,
        args.provider,
        cpu_intra_op_threads=args.cpu_intra_op_threads,
    )
    classifier = OnnxCatalogClassifier(runtime, catalog, embedder)
    class_ids = {label.class_id: index for index, label in enumerate(classifier.labels)}
    manifest = _manifest_index(args.manifest)
    traces = [row for row in _jsonl(args.trace) if row["status"] == "SEGMENTATION"]
    angles = tuple(args.angles)
    if not angles or len(set(angles)) != len(angles) or any(angle % 90 for angle in angles):
        raise ValueError("rotation angles must be unique multiples of 90")
    raw_views: list[list[np.ndarray]] = [[] for _ in angles]
    targets: list[np.ndarray] = []
    traced_top1: list[np.ndarray] = []
    sample_image_ids: list[int] = []
    sample_detection_indices: list[int] = []
    evaluated_image_count = 0
    for trace in traces:
        image_id = int(trace["image_id"])
        source_path = (args.dataset_root / str(manifest[image_id]["image_path"])).resolve()
        if args.dataset_root.resolve() not in source_path.parents:
            raise ValueError("manifest image escapes the dataset root")
        detections = _detections(trace)
        if not detections:
            continue
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            prepared = embedder.prepare_detection_tensors(image, detections)
        view_batch = np.concatenate(
            [np.rot90(prepared, angle // 90, axes=(-2, -1)) for angle in angles]
        )
        unpadded_count = len(view_batch)
        if args.pad_batch_size is not None:
            if unpadded_count > args.pad_batch_size:
                raise ValueError("prepared views exceed the fixed padded batch size")
            padding = np.zeros(
                (args.pad_batch_size - unpadded_count, *view_batch.shape[1:]),
                dtype=np.float32,
            )
            view_batch = np.concatenate((view_batch, padding))
        view_embeddings = embedder.embed_prepared_tensors_raw(view_batch)[:unpadded_count]
        for view_index, values in enumerate(np.split(view_embeddings, len(angles))):
            raw_views[view_index].append(values)
        targets.append(_targets(trace, class_ids))
        traced_top1.append(_trace_top1(trace, class_ids))
        sample_image_ids.extend([image_id] * len(detections))
        sample_detection_indices.extend(range(len(detections)))
        evaluated_image_count += 1

    target_array = np.concatenate(targets)
    traced_top1_array = np.concatenate(traced_top1)
    view_embeddings = [np.concatenate(values) for values in raw_views]
    view_rankings = [_ranking(classifier.classify_embeddings(values)) for values in view_embeddings]
    mean_embeddings = np.mean(np.stack(view_embeddings), axis=0, dtype=np.float32)
    ensemble_ranking = _ranking(classifier.classify_embeddings(mean_embeddings))
    view_top1 = np.stack([ranking[:, 0] for ranking in view_rankings])
    trace_mismatch = int(np.count_nonzero(view_top1[0] != traced_top1_array))
    _validate_trace_reproduction(
        trace_mismatch,
        provider=args.provider,
        allow_alternative_runtime=args.allow_alternative_runtime,
    )
    consensus = (
        _consensus_masks(view_top1, ensemble_ranking[:, 0]) if angles == (0, 90, 180, 270) else {}
    )
    if args.embeddings_output is not None:
        args.embeddings_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.embeddings_output,
            raw_embeddings=view_embeddings[0].astype(np.float32),
            **{
                f"raw_embeddings_angle_{angle}": values.astype(np.float32)
                for angle, values in zip(angles, view_embeddings, strict=True)
            },
            image_ids=np.asarray(sample_image_ids, dtype=np.int64),
            detection_indices=np.asarray(sample_detection_indices, dtype=np.int64),
        )
    report = {
        "schema_version": "1.0",
        "evaluation": "classifier_rotation_consistency",
        "evidence_role": "full_dataset_development_diagnostic",
        "label_usage": "labels are used only for reporting after label-agnostic views",
        "image_count": evaluated_image_count,
        "sample_count": len(target_array),
        "runtime_sha256": sha256_file(args.runtime / "metadata.json"),
        "catalog_sha256": sha256_file(args.catalog / "catalog.json"),
        "trace_sha256": sha256_file(args.trace),
        "unlabeled_embeddings_sha256": (
            None if args.embeddings_output is None else sha256_file(args.embeddings_output)
        ),
        "angles_degrees": list(angles),
        "provider": args.provider,
        "padded_batch_size": args.pad_batch_size,
        "zero_degree_trace_top1_mismatch_count": trace_mismatch,
        "alternative_runtime_comparison": args.allow_alternative_runtime,
        "views": {
            str(angle): _metrics(ranking, target_array)
            for angle, ranking in zip(angles, view_rankings, strict=True)
        },
        "mean_embedding": _metrics(ensemble_ranking, target_array),
        "consensus_policies": {
            name: _metrics(ensemble_ranking, target_array, approved=mask)
            for name, mask in consensus.items()
        },
        "constraints": {
            "classifier_source": "single_objects_3",
            "multi_object_labels_used_for_training_or_calibration": False,
            "class_or_image_specific_rules": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate generic rotation consistency")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embeddings-output", type=Path)
    parser.add_argument("--angles", type=int, nargs="+", default=(0, 90, 180, 270))
    parser.add_argument("--provider", choices=("cpu", "cuda", "openvino"), default="cuda")
    parser.add_argument("--cpu-intra-op-threads", type=int, default=0)
    parser.add_argument("--pad-batch-size", type=int)
    parser.add_argument(
        "--allow-alternative-runtime",
        action="store_true",
        help="report rather than reject expected zero-degree differences from the trace runtime",
    )
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
