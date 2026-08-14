from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..runtime.onnx import OrtRunner
from ..training.staged_classifier_export import view_affine


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    tensors = np.load(args.tensors, mmap_mode="r")
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    if len(tensors) != len(targets):
        raise ValueError("tensor and record counts differ")
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    results: dict[str, np.ndarray] = {}
    metrics = {}
    for name in args.views:
        parts = []
        for start in range(0, len(tensors), args.batch_size):
            batch = np.asarray(tensors[start : start + args.batch_size], dtype=np.float32)
            matrices = np.repeat(view_affine(name)[None], len(batch), axis=0)
            (logits,) = runner.run_inputs(
                [args.logits_output],
                {args.input_name: batch, args.affine_input_name: matrices},
            )
            parts.append(np.asarray(logits, dtype=np.float32))
        values = np.concatenate(parts)
        results[name] = values
        metrics[name] = {"top1_accuracy": float((values.argmax(axis=1) == targets).mean())}
    args.logits_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.logits_output_path, targets=targets, **results)
    report = {
        "schema_version": "1.0",
        "evaluation": "staged_classifier_onnx_view_logits",
        "selection_set": "multi_object_scenes",
        "provider": args.provider,
        "sample_count": len(targets),
        "views": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deployable ONNX view logits")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output-path", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--affine-input-name", default="view_affine")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--views",
        nargs="+",
        default=["base", "hflip", "vflip", "rot15", "rot-15", "rot30", "rot-30"],
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
