from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.model_package import sha256_file
from ...runtime.onnx import OrtRunner
from ...training.data import read_manifest
from .classifier_geometry_mask import (
    apply_background_mask,
    neighbor_ownership_mask,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score_tensors(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.records)
    tensors = np.load(args.tensors, mmap_mode="r")
    manifest = {int(row["image_id"]): row for row in read_manifest(args.manifest)}
    predictions = {int(row["image_id"]): row for row in _read_jsonl(args.predictions)}
    if len(rows) != len(tensors):
        raise ValueError("classifier tensors and records are not aligned")
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    score_parts = []
    for start in range(0, len(rows), args.batch_size):
        batch = np.array(tensors[start : start + args.batch_size], dtype=np.float32, copy=True)
        batch_rows = rows[start : start + len(batch)]
        masks = np.stack(
            [
                neighbor_ownership_mask(
                    image_width=int(manifest[int(row["image_id"])]["width"]),
                    image_height=int(manifest[int(row["image_id"])]["height"]),
                    boxes=row.get(
                        "mask_boxes_xyxy",
                        predictions[int(row["image_id"])]["boxes_xyxy"],
                    ),
                    target_index=int(row.get("mask_target_index", row["detection_index"])),
                    output_size=batch.shape[-1],
                    margin_ratio=args.margin_ratio,
                    distance_bias=args.distance_bias,
                    shared_scale=False,
                )
                for row in batch_rows
            ]
        )
        masked = apply_background_mask(batch, masks).astype(np.float32, copy=False)
        (scores,) = runner.run([args.logits_output], args.input_name, masked)
        score_parts.append(np.asarray(scores, dtype=np.float32))
    scores = np.concatenate(score_parts)
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    proposal_indices = np.asarray(
        [int(row.get("proposal_index", row["detection_index"])) for row in rows],
        dtype=np.int64,
    )
    folds = np.asarray([int(row.get("fold", 0)) for row in rows], dtype=np.int64)
    targets = np.asarray([int(row.get("target", -1)) for row in rows], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        scores=scores,
        image_ids=image_ids,
        proposal_indices=proposal_indices,
        folds=folds,
        targets=targets,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_final_candidate_classifier_tensor_scores",
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "provider": args.provider,
        "sample_count": len(scores),
        "class_count": scores.shape[1],
        "geometry_only_mask": True,
        "target_labels_used_for_scoring": False,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked bread final candidate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score-tensors")
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--tensors", type=Path, required=True)
    score_parser.add_argument("--records", type=Path, required=True)
    score_parser.add_argument("--manifest", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--report", type=Path, required=True)
    score_parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    score_parser.add_argument("--cuda-dll-dir", type=Path)
    score_parser.add_argument("--input-name", default="pixel_values")
    score_parser.add_argument("--logits-output", default="logits")
    score_parser.add_argument("--margin-ratio", type=float, default=0.05)
    score_parser.add_argument("--distance-bias", type=float, default=0.0)
    score_parser.add_argument("--batch-size", type=int, default=96)
    score_tensors(parser.parse_args())


if __name__ == "__main__":
    main()
