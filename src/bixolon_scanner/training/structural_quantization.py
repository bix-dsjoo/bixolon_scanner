"""ONNX static INT8 with per-example evidence preservation on source diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..configuration import load_json_config


def evidence_signature(
    embeddings: np.ndarray,
    integrity: np.ndarray,
    approval_margin: float,
    verification_margin: float,
    quality_threshold: float,
) -> np.ndarray:
    order = np.argsort(-embeddings, axis=-1, kind="stable")
    values = np.take_along_axis(embeddings, order, axis=-1)
    margin = values[:, 0] - values[:, 1]
    return np.column_stack(
        [
            order[:, 0],
            margin >= approval_margin,
            margin >= verification_margin,
            integrity.reshape(-1) >= quality_threshold,
        ]
    )


def run(config_path: Path, model_path: Path, output: Path):
    import nncf
    import onnx
    import onnxruntime as ort

    config = load_json_config(config_path)
    root = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "intermediate").mkdir(exist_ok=True)
    images = np.load(root / "student-cache/images.npy", mmap_mode="r")
    # Fixed source-only indices. They are not a session-disjoint validation split.
    indices = np.random.default_rng(config["seed"]).choice(len(images), size=256, replace=False)
    values = [np.array(images[i : i + 1], dtype=np.float32) for i in indices]
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    settings = load_json_config(Path(config["runtime"]) / "metadata.json")
    approval = settings["classifier_policy"]["ridge_approval_minimum_margin"]
    verification = settings["classifier_verification"]["ambiguity_maximum_approval_score"]
    quality = settings["quality"]["multi_object_recapture_threshold"]

    def session(candidate):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return ort.InferenceSession(
            candidate.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

    reference = session(model)
    expected = []
    for value in values:
        embeddings, integrity = reference.run(None, {input_name: value})
        expected.append(evidence_signature(embeddings, integrity, approval, verification, quality))
    data = list(zip(values, expected, strict=True))
    dataset = nncf.Dataset(data, lambda item: {input_name: item[0]})
    history = []

    def validate(candidate, loader):
        runner = session(candidate)
        scores = []
        for value, target in loader:
            embeddings, integrity = runner.run(None, {input_name: value})
            actual = evidence_signature(embeddings, integrity, approval, verification, quality)
            # A gain on one example cannot compensate for a loss on another.
            scores.append(float(np.array_equal(actual, target)))
        history.append(
            {"evaluation": len(history) + 1, "retained": int(sum(scores)), "count": len(scores)}
        )
        (output / "accuracy-history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        return float(np.mean(scores)), scores

    contract = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_indices": indices.tolist(),
        "nncf": nncf.__version__,
        "onnxruntime": ort.__version__,
        "max_drop": 0.0,
        "subset_size": 128,
        "scope": "source diagnostic evidence restoration, not independent validation or deployment selection",
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    start = time.perf_counter()
    quantized = nncf.quantize_with_accuracy_control(
        model,
        calibration_dataset=dataset,
        validation_dataset=dataset,
        validation_fn=validate,
        max_drop=0.0,
        target_device=nncf.TargetDevice.CPU,
        subset_size=128,
        advanced_accuracy_restorer_parameters=nncf.AdvancedAccuracyRestorerParameters(
            max_num_iterations=200,
            ranking_subset_size=32,
            num_ranking_workers=1,
            intermediate_model_dir=str(output / "intermediate"),
        ),
    )
    score, _ = validate(quantized, data)
    if score != 1.0:
        raise ValueError("quantization did not preserve every diagnostic evidence signature")
    onnx.save(quantized, output / "model.onnx")
    (output / "report.json").write_text(
        json.dumps(
            {
                "contract": contract,
                "elapsed_seconds": time.perf_counter() - start,
                "history": history,
                "model_sha256": hashlib.sha256((output / "model.onnx").read_bytes()).hexdigest(),
                "quantize_nodes": sum(n.op_type == "QuantizeLinear" for n in quantized.graph.node),
                "dequantize_nodes": sum(
                    n.op_type == "DequantizeLinear" for n in quantized.graph.node
                ),
                "deployment_selected": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.model, args.output)


if __name__ == "__main__":
    main()
