from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ...contracts.model_package import load_model_package, sha256_file


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def assemble_runtime_candidate(
    manifest_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    repository_root = manifest_path.parents[2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_contract = manifest["package"]
    metadata_template = _resolve(repository_root, package_contract["metadata_template"])
    if sha256_file(metadata_template) != package_contract["metadata_template_sha256"]:
        raise ValueError("runtime candidate metadata template checksum mismatch")
    metadata_bytes = metadata_template.read_bytes().rstrip(b"\r\n") + b"\n"
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if metadata_sha256 != package_contract["metadata_sha256"]:
        raise ValueError("runtime candidate output metadata checksum mismatch")
    metadata = json.loads(metadata_bytes)
    output = (
        output_dir.resolve()
        if output_dir is not None
        else _resolve(repository_root, package_contract["path"])
    )
    output.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for filename, source_value in manifest["package_sources"].items():
        source = _resolve(repository_root, source_value)
        expected = metadata["checksums"].get(filename)
        if expected is None or sha256_file(source) != expected:
            raise ValueError(f"runtime candidate source checksum mismatch: {filename}")
        destination = output / filename
        if destination.exists():
            if sha256_file(destination) != expected:
                raise FileExistsError(f"runtime candidate destination differs: {destination}")
            action = "reused"
        else:
            shutil.copy2(source, destination)
            action = "copied"
        copied.append({"filename": filename, "sha256": expected, "action": action})
    metadata_destination = output / "metadata.json"
    if metadata_destination.exists():
        if sha256_file(metadata_destination) != metadata_sha256:
            raise FileExistsError(
                f"runtime candidate metadata destination differs: {metadata_destination}"
            )
        metadata_action = "reused"
    else:
        metadata_destination.write_bytes(metadata_bytes)
        metadata_action = "written"
    package = load_model_package(output)
    return {
        "schema_version": "1.0",
        "operation": "assemble_bread_1_1_runtime_candidate",
        "candidate_id": manifest["candidate_id"],
        "output": output.as_posix(),
        "metadata_sha256": metadata_sha256,
        "metadata_action": metadata_action,
        "files": copied,
        "versions": {
            "worker": package.metadata.worker_version,
            "detector": package.metadata.detector.version,
            "classifier": package.metadata.classifier.version,
        },
        "promotion_status": package.metadata.promotion_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the Bread 1.1 v3 runtime candidate")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/bread-zero-error-1.1/final_candidate_v3_2026-08-18.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = assemble_runtime_candidate(args.manifest, args.output_dir)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
