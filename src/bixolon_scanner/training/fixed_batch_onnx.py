from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file


def convert_to_fixed_batch(
    source_path: Path,
    output_path: Path,
    *,
    fixed_batch_size: int = 1,
) -> dict[str, Any]:
    """Make a dynamic-batch ONNX graph statically batch-shaped without changing weights."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if fixed_batch_size < 1:
        raise ValueError("fixed ONNX batch size must be positive")

    import onnx

    model = onnx.load(source_path)
    values = [*model.graph.input, *model.graph.output]
    if not values:
        raise ValueError("ONNX graph has no public inputs or outputs")
    dynamic_value_names: list[str] = []
    for value in values:
        dimensions = value.type.tensor_type.shape.dim
        if not dimensions:
            raise ValueError(f"ONNX value has no batch dimension: {value.name}")
        batch = dimensions[0]
        if batch.dim_param:
            dynamic_value_names.append(value.name)
        elif batch.dim_value not in {0, fixed_batch_size}:
            raise ValueError(f"ONNX value has an incompatible fixed batch: {value.name}")
        batch.ClearField("dim_param")
        batch.dim_value = fixed_batch_size
    if not dynamic_value_names:
        raise ValueError("ONNX graph does not declare a dynamic public batch")

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return {
        "schema_version": "1.0",
        "operation": "fix_public_onnx_batch_and_infer_shapes",
        "source_onnx": source_path.resolve().as_posix(),
        "source_onnx_sha256": sha256_file(source_path),
        "output_onnx": output_path.resolve().as_posix(),
        "onnx_sha256": sha256_file(output_path),
        "fixed_batch_size": fixed_batch_size,
        "fixed_public_values": sorted(dynamic_value_names),
        "onnx_ir_version": model.ir_version,
        "opsets": {entry.domain or "ai.onnx": entry.version for entry in model.opset_import},
        "weights_modified": False,
        "shape_inference": {"strict_mode": True, "data_prop": True},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a dynamic-batch ONNX to fixed batch")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-batch-size", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = convert_to_fixed_batch(
        args.source,
        args.output,
        fixed_batch_size=args.fixed_batch_size,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
