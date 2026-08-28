from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import RuntimePackageV2Metadata, load_runtime_package_v2


def enable_unknown_dual_verifier_recapture(
    source_runtime_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Copy a Runtime and enable the verified UNKNOWN dual-verifier safety policy."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = load_runtime_package_v2(source_runtime_dir)
    verification = source.metadata.classifier_verification
    if verification is None:
        raise ValueError("classifier verification metadata is required")

    payload = source.metadata.model_dump(mode="json", exclude_none=True)
    payload["classifier_verification"]["unknown_recapture_on_dual_verifier_rejection"] = True
    metadata = RuntimePackageV2Metadata.model_validate(payload)

    shutil.copytree(source.root, output_dir)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = load_runtime_package_v2(output_dir)
    if not (
        candidate.metadata.classifier_verification
        and candidate.metadata.classifier_verification.unknown_recapture_on_dual_verifier_rejection
    ):
        raise ValueError("candidate Runtime did not enable the UNKNOWN safety policy")

    source_payloads = {
        path.relative_to(source.root).as_posix(): sha256_file(path)
        for path in source.root.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    }
    candidate_payloads = {
        path.relative_to(candidate.root).as_posix(): sha256_file(path)
        for path in candidate.root.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    }
    if candidate_payloads != source_payloads:
        raise ValueError("classifier safety packaging modified an immutable Runtime payload")

    return {
        "schema_version": "1.0",
        "operation": "enable_unknown_dual_verifier_recapture",
        "source_runtime": source.root.as_posix(),
        "candidate_runtime": candidate.root.as_posix(),
        "policy": {
            "classifier_verification.unknown_recapture_on_dual_verifier_rejection": True,
        },
        "model_graph_or_weight_changed": False,
        "source_manifest_sha256": directory_content_manifest(source.root)["manifest_sha256"],
        "candidate_manifest_sha256": directory_content_manifest(candidate.root)["manifest_sha256"],
        "metadata_sha256": sha256_file(metadata_path),
        "payload_sha256": dict(sorted(candidate.metadata.checksums.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Enable the verified UNKNOWN dual-verifier recapture policy"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = enable_unknown_dual_verifier_recapture(
        args.source_runtime,
        args.output_dir,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
