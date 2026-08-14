from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ....package import sha256_file
from ....training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ....training.models import require_torch


def export(args: argparse.Namespace) -> dict[str, object]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=f"facebookresearch/dinov3:{checkpoint['source_revision']}",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    model.eval()
    image_size = int(checkpoint["image_size"])
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(args.output))
    tensors = np.load(args.parity_tensors, mmap_mode="r")[: args.parity_samples].astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(tensors)).float().numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"pixel_values": tensors})[0]
    difference = np.abs(expected - actual)
    report = {
        "schema_version": "1.0",
        "evaluation": "classifier_onnx_export_probe",
        "checkpoint": args.checkpoint.name,
        "model": args.output.name,
        "model_sha256": sha256_file(args.output),
        "model_size_bytes": args.output.stat().st_size,
        "dynamic_batch": True,
        "parity": {
            "sample_count": len(tensors),
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "top1_equal": bool(np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))),
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a classifier probe to dynamic-batch ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parity-tensors", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
