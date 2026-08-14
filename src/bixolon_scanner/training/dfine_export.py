from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def checkpoint_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the inference weights preferred by the official D-FINE exporter."""
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model
    raise ValueError("D-FINE checkpoint has neither EMA nor model weights")


def compatible_checkpoint_state(
    current: dict[str, Any], checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """Drop only resolution-derived D-FINE anchor buffers when shapes change."""
    allowed = {"decoder.anchors", "decoder.valid_mask"}
    result = dict(checkpoint)
    for name in allowed:
        if name in result and name in current and result[name].shape != current[name].shape:
            del result[name]
    return result


def apply_export_resolution(config: dict[str, Any], input_size: tuple[int, int]) -> None:
    """Keep D-FINE's cached positional encodings aligned with the export input."""
    height, width = input_size
    config["eval_spatial_size"] = [height, width]


def export_dfine_onnx(
    *,
    repository: Path,
    config: Path,
    checkpoint: Path,
    output: Path,
    input_size: tuple[int, int] = (640, 640),
    opset: int = 16,
) -> None:
    """Export raw D-FINE queries to the canonical one-input detector contract."""
    repository = repository.resolve()
    config = config.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not (repository / "src" / "core").is_dir():
        raise ValueError(f"not a D-FINE repository: {repository}")
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(repository))
    try:
        import onnx
        import torch
        from src.core import YAMLConfig

        cfg = YAMLConfig(str(config), resume=str(checkpoint))
        apply_export_resolution(cfg.yaml_cfg, input_size)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = compatible_checkpoint_state(cfg.model.state_dict(), checkpoint_model_state(payload))
        missing, unexpected = cfg.model.load_state_dict(state, strict=False)
        allowed_missing = {"decoder.anchors", "decoder.valid_mask"}
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                f"incompatible D-FINE checkpoint: missing={missing}, unexpected={unexpected}"
            )

        class RawDetector(torch.nn.Module):
            def __init__(self, model) -> None:
                super().__init__()
                self.model = model.deploy()

            def forward(self, pixel_values):
                result = self.model(pixel_values)
                return result["pred_logits"], result["pred_boxes"]

        model = RawDetector(cfg.model).eval()
        height, width = input_size
        dummy = torch.zeros((1, 3, height, width), dtype=torch.float32)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(output))
    finally:
        if sys.path and sys.path[0] == str(repository):
            sys.path.pop(0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export D-FINE raw queries to the Bixolon detector ONNX contract"
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--opset", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    export_dfine_onnx(
        repository=args.repository,
        config=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        input_size=(args.input_height, args.input_width),
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
