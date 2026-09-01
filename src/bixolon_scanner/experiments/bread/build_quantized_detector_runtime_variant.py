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
    if not args.quantized_detector.is_file():
        raise FileNotFoundError(args.quantized_detector)
    source_runtime = load_runtime_package_v2(args.source_runtime)
    ensemble = source_runtime.metadata.detector.ensemble
    if ensemble is None or args.member_filename not in {
        member.filename for member in ensemble.members
    }:
        raise ValueError("requested detector member is not part of the source ensemble")
    source_detector = args.source_runtime / args.member_filename
    if not source_detector.is_file():
        raise FileNotFoundError(source_detector)
    source_sha256 = _sha256(source_detector)
    candidate_sha256 = _sha256(args.quantized_detector)
    if source_sha256 == candidate_sha256:
        raise ValueError("quantized detector is byte-identical to the source detector")

    shutil.copytree(args.source_runtime, args.output_dir)
    output_detector = args.output_dir / args.member_filename
    shutil.copy2(args.quantized_detector, output_detector)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    detector_source = metadata.setdefault("sources", {}).setdefault("detector", {})
    architecture = detector_source.get("architecture", "D-FINE detector")
    detector_source["architecture"] = (
        f"{architecture}; {args.member_filename} NNCF mixed-precision INT8 "
        f"({args.quantization_description})"
    )
    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validated = load_runtime_package_v2(args.output_dir)
    validated_ensemble = validated.metadata.detector.ensemble
    if validated_ensemble is None or args.member_filename not in {
        member.filename for member in validated_ensemble.members
    }:
        raise RuntimeError("quantized detector member metadata was not preserved")
    return {
        "schema_version": "1.0",
        "operation": "build_quantized_detector_runtime_variant",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "member_filename": args.member_filename,
        "source_detector_sha256": source_sha256,
        "quantized_detector_sha256": candidate_sha256,
        "quantization": args.quantization_description,
        "weights_modified": True,
        "graph_change": "NNCF post-training mixed-precision INT8 quantization",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replace one detector ensemble member with a checked INT8 model"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--quantized-detector", type=Path, required=True)
    parser.add_argument("--member-filename", required=True)
    parser.add_argument(
        "--quantization-description",
        default="HGNet backbone and encoder INT8; decoder and heads FP32",
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
