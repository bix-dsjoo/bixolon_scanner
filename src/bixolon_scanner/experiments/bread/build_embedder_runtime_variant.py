from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ...contracts.runtime_package_v2 import load_runtime_package_v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if not args.embedder.is_file():
        raise FileNotFoundError(args.embedder)
    source_embedder = args.source_runtime / "embedder.onnx"
    if not source_embedder.is_file():
        raise FileNotFoundError(source_embedder)
    source_embedder_sha256 = _sha256(source_embedder)
    shutil.copytree(args.source_runtime, args.output_dir)
    output = args.output_dir
    shutil.copy2(args.embedder, output / "embedder.onnx")

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_input_size = metadata["embedder"]["input_size"]
    metadata["embedder"]["input_size"] = [args.image_size, args.image_size]
    metadata["checksums"] = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_runtime_package_v2(output)
    return {
        "schema_version": "1.0",
        "operation": "build_embedder_input_runtime_variant",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": output.resolve().as_posix(),
        "derivation": "public_onnx_input_shape_resize",
        "source_embedder_input_size": source_input_size,
        "embedder_input_size": [args.image_size, args.image_size],
        "embedder_sha256": metadata["checksums"]["embedder.onnx"],
        "source_embedder_sha256": source_embedder_sha256,
        "weights_modified": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a diagnostic embedder-size runtime")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
