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
    shutil.copytree(args.source_runtime, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selector = metadata["detector"]["ensemble"]["class_verified_selector"]
    selector["low_resolution_maximum_dimension"] = None
    selector["low_resolution_score_threshold"] = None
    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_runtime_package_v2(args.output_dir)
    return {
        "schema_version": "1.0",
        "operation": "build_yolo_free_n100_policy_runtime",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "model_files_modified": False,
        "policy_changes": {
            "low_resolution_maximum_dimension": None,
            "low_resolution_score_threshold": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the YOLO-free N100 policy candidate")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
