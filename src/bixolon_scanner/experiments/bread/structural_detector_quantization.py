"""Source GT-constrained ONNX detector INT8 diagnostic; final scan regression is separate."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ...configuration import load_json_config
from ...contracts.catalog import sha256_file
from ...evaluation.three_bakery import match_boxes
from ...pipeline.ports import Detection
from ...runtime.geometry import nms
from ...training.three_bakery_data import read_jsonl, write_json


def materialize_constant_aliases(model):
    """Give constant Identity aliases their exact tensor payload for NNCF lookup."""
    result = copy.deepcopy(model)
    constants = {tensor.name: tensor for tensor in result.graph.initializer}
    retained, aliases = [], []
    for node in result.graph.node:
        if node.op_type == "Constant":
            tensors = [attribute.t for attribute in node.attribute if attribute.name == "value"]
            if len(tensors) == 1:
                constants[node.output[0]] = tensors[0]
        if node.op_type == "Identity" and node.input[0] in constants:
            tensor = copy.deepcopy(constants[node.input[0]])
            tensor.name = node.output[0]
            result.graph.initializer.append(tensor)
            constants[tensor.name] = tensor
            aliases.append(node.name)
        else:
            retained.append(node)
    del result.graph.node[:]
    result.graph.node.extend(retained)
    return result, aliases


def detection_signature(outputs, row, metadata):
    logits, boxes = outputs
    scores = 1 / (1 + np.exp(-np.clip(logits[0].reshape(-1), -80, 80)))
    detections = []
    for score, (cx, cy, width, height) in zip(scores, boxes[0], strict=True):
        if score < metadata["score_threshold"] or min(width, height) <= 0:
            continue
        if max(width / height, height / width) > metadata["max_object_aspect_ratio"]:
            continue
        detections.append(
            Detection(
                max(0.0, cx - width / 2) * row["width"],
                max(0.0, cy - height / 2) * row["height"],
                min(1.0, cx + width / 2) * row["width"],
                min(1.0, cy + height / 2) * row["height"],
                float(score),
            )
        )
    detections = nms(detections, metadata["nms_iou_threshold"])
    predicted = [[d.x1, d.y1, d.x2, d.y2] for d in detections]
    truth = [
        [
            a["bbox_xywh"][0],
            a["bbox_xywh"][1],
            a["bbox_xywh"][0] + a["bbox_xywh"][2],
            a["bbox_xywh"][1] + a["bbox_xywh"][3],
        ]
        for a in row["annotations"]
    ]
    matches = match_boxes(predicted, truth, 0.5)
    return set(matches.values()), len(predicted) - len(matches)


def run(config_path: Path, model_path: Path, output: Path):
    import nncf
    import onnx
    import onnxruntime as ort

    config = load_json_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "intermediate").mkdir(exist_ok=True)
    metadata = load_json_config(Path(config["runtime"]) / "metadata.json")["detector"]
    originals = read_jsonl(Path(config["source_work"]) / "prepared/original_detection.jsonl")
    indices = sorted(
        np.random.default_rng(config["seed"]).choice(len(originals), 64, replace=False).tolist()
    )
    records = [originals[index] for index in indices]
    data = []
    for row in records:
        if row["split"] != "train" or sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError("detector quantization source provenance mismatch")
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB").resize(
                tuple(metadata["input_size"]), Image.Resampling.BILINEAR
            )
        values = np.asarray(image).astype(np.float32).transpose(2, 0, 1)[None] / 255
        data.append((values, row))
    original_model = onnx.load(model_path)
    model, constant_aliases = materialize_constant_aliases(original_model)
    onnx.checker.check_model(model)
    onnx.save(model, output / "prepared-fp32.onnx")
    input_name = metadata["input_name"]
    # D-FINE integral projection has a batched constant operand. ORT CPU's
    # QLinearMatMul rejects its per-channel zero-point shape; keep this decoder
    # operation floating point rather than substituting a different output.
    unsupported = [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul" and "/integral" in node.name
    ]

    def session(candidate):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return ort.InferenceSession(
            candidate.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

    runner = session(model)
    original_runner = session(original_model)
    parity_max_error = 0.0
    for value, _ in data:
        before = original_runner.run(None, {input_name: value})
        after = runner.run(None, {input_name: value})
        for left, right in zip(before, after, strict=True):
            np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
            parity_max_error = max(parity_max_error, float(np.max(np.abs(left - right))))
    del original_runner
    expected = {
        r["image_sha256"]: detection_signature(runner.run(None, {input_name: v}), r, metadata)
        for v, r in data
    }
    dataset = nncf.Dataset(data, lambda item: {input_name: item[0]})
    history = []

    def validate(candidate, loader):
        runner = session(candidate)
        scores = []
        for values, row in loader:
            matched, extra = detection_signature(
                runner.run(None, {input_name: values}), row, metadata
            )
            old_matched, old_extra = expected[row["image_sha256"]]
            scores.append(float(old_matched <= matched and extra <= old_extra))
        history.append(
            {"evaluation": len(history) + 1, "preserved": int(sum(scores)), "count": len(scores)}
        )
        write_json(output / "accuracy-history.json", history)
        return float(np.mean(scores)), scores

    contract = {
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(model_path),
        "source_indices": indices,
        "nncf": nncf.__version__,
        "ort": ort.__version__,
        "cpu_kernel_exclusions": unsupported,
        "materialized_constant_aliases": constant_aliases,
        "preparation_parity_max_error": parity_max_error,
        "scope": "64 source frames, GT coverage and extra detection preservation; not independent validation or final scan accuracy",
    }
    write_json(output / "contract.json", contract)
    start = time.perf_counter()
    quantized = nncf.quantize_with_accuracy_control(
        model,
        calibration_dataset=dataset,
        validation_dataset=dataset,
        validation_fn=validate,
        max_drop=0.0,
        subset_size=64,
        target_device=nncf.TargetDevice.CPU,
        ignored_scope=nncf.IgnoredScope(names=unsupported),
        advanced_accuracy_restorer_parameters=nncf.AdvancedAccuracyRestorerParameters(
            max_num_iterations=200,
            ranking_subset_size=16,
            num_ranking_workers=1,
            intermediate_model_dir=str(output / "intermediate"),
        ),
    )
    score, _ = validate(quantized, data)
    if score != 1.0:
        raise ValueError("detector quantization lost source GT evidence")
    onnx.save(quantized, output / "detector.onnx")
    write_json(
        output / "report.json",
        {
            "contract": contract,
            "history": history,
            "elapsed_seconds": time.perf_counter() - start,
            "onnx_sha256": sha256_file(output / "detector.onnx"),
            "quantize_nodes": sum(n.op_type == "QuantizeLinear" for n in quantized.graph.node),
            "deployment_selected": False,
        },
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
