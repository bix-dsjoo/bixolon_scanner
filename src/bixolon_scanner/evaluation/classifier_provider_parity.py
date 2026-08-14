from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.model_package import sha256_file


def classifier_provider_parity(
    model_path: Path,
    tensor_path: Path,
    *,
    sample_count: int,
    maximum_absolute_error: float,
    cuda_dll_dir: Path | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    required = {"CPUExecutionProvider", "CUDAExecutionProvider"}
    if not required.issubset(available):
        raise RuntimeError(f"classifier parity requires {sorted(required)}")
    tensors = np.load(tensor_path, mmap_mode="r")[:sample_count].astype(np.float32)
    affine = np.repeat(
        np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
        len(tensors),
        axis=0,
    )
    feeds = {"pixel_values": tensors, "view_affine": affine}
    dll_handle = None
    if cuda_dll_dir is not None:
        cuda_dll_dir = cuda_dll_dir.resolve()
        if not cuda_dll_dir.is_dir():
            raise FileNotFoundError(cuda_dll_dir)
        os.environ["PATH"] = f"{cuda_dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            dll_handle = os.add_dll_directory(str(cuda_dll_dir))
    cpu = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    cuda = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    if cpu.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPU parity session did not select CPUExecutionProvider")
    if cuda.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA parity session did not select CUDAExecutionProvider")
    reference = cpu.run(["logits"], feeds)[0]
    candidate = cuda.run(["logits"], feeds)[0]
    difference = np.abs(reference - candidate)
    maximum = float(difference.max())
    mean = float(difference.mean())
    top1_equal = bool(np.array_equal(reference.argmax(axis=1), candidate.argmax(axis=1)))
    top3_equal = bool(
        np.array_equal(np.argsort(reference, axis=1)[:, -3:], np.argsort(candidate, axis=1)[:, -3:])
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "classifier_cpu_cuda_parity",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "tensor_source": str(tensor_path),
        "tensor_source_sha256": sha256_file(tensor_path),
        "sample_count": len(tensors),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": mean,
        "maximum_absolute_error": maximum_absolute_error,
        "top1_equal": top1_equal,
        "top3_rank_equal": top3_equal,
        "onnxruntime_version": ort.__version__,
        "passes": top1_equal and top3_equal and maximum <= maximum_absolute_error,
    }
    if dll_handle is not None:
        dll_handle.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Classifier CPU and CUDA ONNX output")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--maximum-absolute-error", type=float, default=0.001)
    parser.add_argument("--cuda-dll-dir", type=Path)
    args = parser.parse_args()
    report = classifier_provider_parity(
        args.model,
        args.tensors,
        sample_count=args.sample_count,
        maximum_absolute_error=args.maximum_absolute_error,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
