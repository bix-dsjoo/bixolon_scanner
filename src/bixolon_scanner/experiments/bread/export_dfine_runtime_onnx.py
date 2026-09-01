from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ...contracts.catalog import sha256_file


def export_dfine_runtime_onnx(
    *,
    dfine_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    image_size: int,
) -> dict[str, Any]:
    """Export a D-FINE checkpoint's raw logits and normalized boxes."""

    if image_size < 32 or image_size % 32:
        raise ValueError("D-FINE image size must be a positive multiple of 32")
    for name, path in {
        "D-FINE root": dfine_root,
        "D-FINE config": config_path,
        "D-FINE checkpoint": checkpoint_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if output_path.exists():
        raise FileExistsError(output_path)

    import torch
    import torch.nn as nn

    resolved_root = dfine_root.resolve()
    resolved_config = config_path.resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    resolved_output = output_path.resolve()
    sys.path.insert(0, str(resolved_root))
    previous_directory = Path.cwd()
    try:
        os.chdir(resolved_root)
        from src.core import YAMLConfig

        config = YAMLConfig(str(resolved_config), resume=str(resolved_checkpoint))
        config.yaml_cfg["eval_spatial_size"] = [image_size, image_size]
        if "HGNetv2" in config.yaml_cfg:
            config.yaml_cfg["HGNetv2"]["pretrained"] = False
        checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        runtime_model = config.model
        runtime_state = runtime_model.state_dict()
        load_state = dict(state)
        regenerated_buffers: list[str] = []
        for name in ("decoder.anchors", "decoder.valid_mask"):
            if name in load_state and load_state[name].shape != runtime_state[name].shape:
                load_state.pop(name)
                regenerated_buffers.append(name)
        incompatible = runtime_model.load_state_dict(load_state, strict=False)
        if (
            set(incompatible.missing_keys) != set(regenerated_buffers)
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                "D-FINE checkpoint mismatch outside resolution-derived buffers: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        class RawOutputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = runtime_model.deploy()

            def forward(self, pixel_values):  # type: ignore[no-untyped-def]
                outputs = self.model(pixel_values)
                return outputs["pred_logits"], outputs["pred_boxes"]

        model = RawOutputModel().eval()
        dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
        with torch.inference_mode():
            logits, boxes = model(dummy)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            resolved_output,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    finally:
        os.chdir(previous_directory)
        try:
            sys.path.remove(str(resolved_root))
        except ValueError:
            pass

    import onnx

    exported = onnx.load(resolved_output)
    onnx.checker.check_model(exported)
    return {
        "schema_version": "1.0",
        "operation": "export_dfine_raw_runtime_onnx",
        "dfine_root": resolved_root.as_posix(),
        "config": resolved_config.as_posix(),
        "config_sha256": sha256_file(config_path),
        "checkpoint": resolved_checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output": resolved_output.as_posix(),
        "onnx_sha256": sha256_file(output_path),
        "image_size": image_size,
        "output_shapes": {
            "logits": list(logits.shape),
            "pred_boxes": list(boxes.shape),
        },
        "weights_modified": False,
        "resolution_derived_buffers_regenerated": regenerated_buffers,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export raw D-FINE Runtime ONNX")
    parser.add_argument("--dfine-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_dfine_runtime_onnx(
        dfine_root=args.dfine_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
