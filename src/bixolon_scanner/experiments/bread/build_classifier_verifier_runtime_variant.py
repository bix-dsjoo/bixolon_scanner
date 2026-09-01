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
    if not args.verifier.is_file():
        raise FileNotFoundError(args.verifier)
    if args.input_size < 16 or args.input_size % 16:
        raise ValueError("DINOv3 ViT verifier size must be a positive multiple of 16")

    shutil.copytree(args.source_runtime, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verification = metadata.get("classifier_verification")
    if verification is None:
        raise ValueError("source runtime has no classifier verifier")
    embedder = verification["independent_embedder"]
    verifier_path = args.output_dir / embedder["filename"]
    source_sha256 = _sha256(verifier_path)
    shutil.copy2(args.verifier, verifier_path)
    source_input_size = embedder["input_size"]
    embedder["input_size"] = [args.input_size, args.input_size]
    verifier_source = metadata.setdefault("sources", {}).setdefault(
        "classifier_verifier",
        {},
    )
    verifier_source["architecture"] = (
        f"DINOv3 ViT-B/16 frozen independent verifier at {args.input_size}x{args.input_size}"
    )
    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    runtime = load_runtime_package_v2(args.output_dir)
    validated = runtime.metadata.classifier_verification
    if validated is None or validated.independent_embedder.input_size != (
        args.input_size,
        args.input_size,
    ):
        raise RuntimeError("classifier verifier input size was not preserved")
    return {
        "schema_version": "1.0",
        "operation": "build_classifier_verifier_runtime_variant",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "source_input_size": source_input_size,
        "input_size": [args.input_size, args.input_size],
        "source_verifier_sha256": source_sha256,
        "verifier_sha256": _sha256(verifier_path),
        "weights_modified": False,
        "graph_change": "same frozen checkpoint re-exported for the requested fixed input size",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a verifier-size Runtime variant")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--input-size", type=int, required=True)
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
