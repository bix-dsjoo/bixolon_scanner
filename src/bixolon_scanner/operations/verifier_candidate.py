"""Offline experimental verifier graph optimization; never imported by the Worker."""

import json
import shutil
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file


def optimize_verifier(source: Path, destination: Path, *, quantize: bool = False) -> dict:
    import onnxruntime as ort

    if destination.exists():
        raise ValueError("Candidate destination must be new")
    metadata = load_json_config(source / "metadata.json")
    name = metadata["classifier_verification"]["independent_embedder"]["filename"]
    shutil.copytree(source, destination)
    original, optimized = source / name, destination / name
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(optimized)
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(original), options, providers=["CPUExecutionProvider"])
    del session
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        temporary = destination / "quantized-candidate.onnx"
        quantize_dynamic(
            str(optimized),
            str(temporary),
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},
        )
        temporary.replace(optimized)
    metadata["checksums"][name] = sha256_file(optimized)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "model": name,
        "source_sha256": sha256_file(original),
        "candidate_sha256": sha256_file(optimized),
        "onnxruntime_version": ort.__version__,
        "graph_optimization": "ORT_ENABLE_BASIC",
        "weight_quantization": "MatMulConstB QInt8 per-channel reduce-range" if quantize else None,
        "classification_policy_changed": False,
        "requires_accuracy_validation": True,
    }
