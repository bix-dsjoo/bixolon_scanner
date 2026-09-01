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
    if not args.quantized_embedder.is_file():
        raise FileNotFoundError(args.quantized_embedder)

    source_runtime = load_runtime_package_v2(args.source_runtime)
    embedder_filename = source_runtime.metadata.embedder.filename
    source_embedder = args.source_runtime / embedder_filename
    if not source_embedder.is_file():
        raise FileNotFoundError(source_embedder)

    source_embedder_sha256 = _sha256(source_embedder)
    quantized_embedder_sha256 = _sha256(args.quantized_embedder)
    if source_embedder_sha256 == quantized_embedder_sha256:
        raise ValueError("quantized embedder is byte-identical to the source embedder")

    shutil.copytree(args.source_runtime, args.output_dir)
    output_embedder = args.output_dir / embedder_filename
    shutil.copy2(args.quantized_embedder, output_embedder)

    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embedder_source = metadata.setdefault("sources", {}).setdefault("embedder", {})
    source_architecture = embedder_source.get("architecture", "DINOv3 embedder")
    embedder_source["architecture"] = (
        f"{source_architecture}; NNCF INT8 PTQ ({args.quantization_description})"
    )
    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    validated = load_runtime_package_v2(args.output_dir)
    if validated.metadata.embedder.filename != embedder_filename:
        raise RuntimeError("primary embedder filename changed while building the variant")
    if _sha256(output_embedder) != quantized_embedder_sha256:
        raise RuntimeError("quantized embedder was not preserved in the runtime variant")

    return {
        "schema_version": "1.0",
        "operation": "build_quantized_embedder_runtime_variant",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "embedder_filename": embedder_filename,
        "source_embedder_sha256": source_embedder_sha256,
        "quantized_embedder_sha256": quantized_embedder_sha256,
        "quantization": args.quantization_description,
        "weights_modified": True,
        "graph_change": "NNCF post-training INT8 quantization with QDQ operators",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replace only the primary embedder with a checked INT8 model"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--quantized-embedder", type=Path, required=True)
    parser.add_argument(
        "--quantization-description",
        default="MIXED preset, CPU target, Transformer mode, SmoothQuant alpha=0.5",
    )
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
