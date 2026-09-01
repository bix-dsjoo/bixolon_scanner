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
    direct_classes = sorted(set(args.direct_class_index))
    if len(direct_classes) != len(args.direct_class_index):
        raise ValueError("direct class indices must be unique")

    source = load_runtime_package_v2(args.source_runtime)
    if source.metadata.detector_class_mode != "class_aware":
        raise ValueError("detector-primary routing requires a class-aware detector")
    if any(index < 0 or index >= source.metadata.detector_class_count for index in direct_classes):
        raise ValueError("direct class index exceeds detector class count")

    shutil.copytree(args.source_runtime, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["detector_primary_classifier_routing"] = {
        "direct_approval_class_indices": direct_classes,
        "minimum_detector_score": args.minimum_detector_score,
        "require_unique_class_per_image": args.require_unique_class_per_image,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    runtime = load_runtime_package_v2(args.output_dir)
    routing = runtime.metadata.detector_primary_classifier_routing
    if routing is None or routing.direct_approval_class_indices != direct_classes:
        raise RuntimeError("detector-primary routing metadata was not preserved")

    return {
        "schema_version": "1.0",
        "operation": "build_detector_primary_runtime",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "source_metadata_sha256": _sha256(args.source_runtime / "metadata.json"),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "output_metadata_sha256": _sha256(metadata_path),
        "direct_approval_class_indices": direct_classes,
        "minimum_detector_score": args.minimum_detector_score,
        "require_unique_class_per_image": args.require_unique_class_per_image,
        "model_files_modified": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a detector-first Runtime with selective Catalog classification"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--direct-class-index",
        type=int,
        action="append",
        required=True,
    )
    parser.add_argument("--minimum-detector-score", type=float, default=0.98)
    parser.add_argument(
        "--require-unique-class-per-image",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    report = build(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
