from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file


def resize_static_image_input(
    source_path: Path,
    output_path: Path,
    *,
    image_size: int,
) -> dict[str, Any]:
    """Change only a convolutional ONNX graph's public square image input shape."""
    if image_size < 32 or image_size % 32:
        raise ValueError("ONNX image size must be a positive multiple of 32")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(output_path)

    import onnx

    model = onnx.load(source_path)
    if len(model.graph.input) != 1:
        raise ValueError("ONNX graph must expose exactly one public input")
    public_input = model.graph.input[0]
    dimensions = public_input.type.tensor_type.shape.dim
    if len(dimensions) != 4:
        raise ValueError("ONNX input must use NCHW rank 4")
    channels = dimensions[1]
    if channels.dim_value != 3 or channels.dim_param:
        raise ValueError("ONNX image input must have three static channels")
    height, width = dimensions[2:]
    if (
        height.dim_param
        or width.dim_param
        or height.dim_value < 1
        or width.dim_value < 1
        or height.dim_value != width.dim_value
    ):
        raise ValueError("ONNX image input must have one static square spatial shape")
    source_image_size = int(height.dim_value)
    if source_image_size == image_size:
        raise ValueError("requested ONNX image size is unchanged")

    height.dim_value = image_size
    width.dim_value = image_size
    # Inferred intermediate shapes describe the source resolution and are not graph
    # semantics. Recompute them after changing the public input contract.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return {
        "schema_version": "1.0",
        "operation": "resize_static_square_onnx_image_input",
        "source_onnx": source_path.resolve().as_posix(),
        "source_onnx_sha256": sha256_file(source_path),
        "output_onnx": output_path.resolve().as_posix(),
        "onnx_sha256": sha256_file(output_path),
        "source_image_size": source_image_size,
        "image_size": image_size,
        "weights_modified": False,
        "shape_inference": {"strict_mode": True, "data_prop": True},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resize a static square ONNX image input")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = resize_static_image_input(
        args.source,
        args.output,
        image_size=args.image_size,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
