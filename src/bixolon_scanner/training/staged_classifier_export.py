from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..package import sha256_file
from .fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from .models import require_torch


def view_affine(name: str) -> np.ndarray:
    if name == "base":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if name == "hflip":
        return np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if name == "vflip":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    if not name.startswith("rot"):
        raise ValueError(f"unsupported staged classifier view: {name}")
    radians = math.radians(float(name.removeprefix("rot")))
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.asarray([[cosine, sine, 0.0], [-sine, cosine, 0.0]], dtype=np.float32)


def build_staged_view_model(torch, classifier, *, input_size: int, crop_scale: float):
    crop_size = round(input_size * crop_scale)
    offset = (input_size - crop_size) // 2

    class StagedViewModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = classifier

        def forward(self, pixel_values, view_affine):
            cropped = pixel_values[
                ...,
                offset : offset + crop_size,
                offset : offset + crop_size,
            ]
            resized = torch.nn.functional.interpolate(
                cropped,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            grid = torch.nn.functional.affine_grid(
                view_affine,
                resized.shape,
                align_corners=False,
            )
            viewed = torch.nn.functional.grid_sample(
                resized,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            return self.classifier(viewed)

    return StagedViewModel()


def export(args: argparse.Namespace) -> dict[str, object]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classifier = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=f"facebookresearch/dinov3:{checkpoint['source_revision']}",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    classifier.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    image_size = int(checkpoint["image_size"])
    model = build_staged_view_model(
        torch,
        classifier,
        input_size=image_size,
        crop_scale=args.crop_scale,
    ).eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    affine = torch.from_numpy(view_affine("base"))[None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy, affine),
        args.output,
        input_names=["pixel_values", "view_affine"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "view_affine": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(args.output))
    tensors = np.load(args.parity_tensors, mmap_mode="r")[: args.parity_samples].astype(np.float32)
    matrices = np.repeat(view_affine(args.parity_view)[None], len(tensors), axis=0)
    with torch.inference_mode():
        expected = model(torch.from_numpy(tensors), torch.from_numpy(matrices)).float().numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"pixel_values": tensors, "view_affine": matrices})[0]
    difference = np.abs(expected - actual)
    report = {
        "schema_version": "1.0",
        "evaluation": "staged_classifier_onnx_export",
        "checkpoint": args.checkpoint.name,
        "model": args.output.name,
        "model_sha256": sha256_file(args.output),
        "model_size_bytes": args.output.stat().st_size,
        "dynamic_batch": True,
        "center_crop_scale": args.crop_scale,
        "parity": {
            "view": args.parity_view,
            "sample_count": len(tensors),
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "top1_equal": bool(np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a staged-view classifier ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parity-tensors", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--parity-view", default="rot-15")
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--opset", type=int, default=20)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
