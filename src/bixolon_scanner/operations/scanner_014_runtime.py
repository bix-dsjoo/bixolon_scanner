from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import (
    DetectorCrowdingPolicyMetadata,
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)


def default_crowding_policy() -> DetectorCrowdingPolicyMetadata:
    return DetectorCrowdingPolicyMetadata(
        minimum_image_aspect_ratio=1.0,
        candidate_score_threshold=0.05,
        large_proposal_score_threshold=0.145,
        large_proposal_minimum_area_ratio=0.21,
        proximity_maximum_normalized_center_distance=0.48,
        query_cluster_iou_threshold=0.7,
        query_duplicate_minimum_fraction=0.93,
        rotation_recovery_degrees=[90, 180],
        rotation_recovery_minimum_selected_count=5,
        rotation_recovery_maximum_selected_count=5,
        rotation_recovery_minimum_selected_area_fraction=0.4,
        rotation_recovery_minimum_normalized_center_distance=0.63,
        rotation_recovery_minimum_count_gain=1,
        rotation_recovery_agreement_iou_threshold=0.5,
    )


def candidate_metadata(
    base: RuntimePackageV2Metadata,
    *,
    version: str = "0.1.4",
    crowding_policy: DetectorCrowdingPolicyMetadata | None = None,
) -> RuntimePackageV2Metadata:
    """Create the 0.1.4 metadata contract without modifying model payloads."""
    payload = base.model_dump(mode="json")
    payload.pop("promotion_status", None)
    payload.update(
        {
            "worker_version": version,
            "detector_policy_version": version,
            "detector_crowding": (crowding_policy or default_crowding_policy()).model_dump(
                mode="json"
            ),
        }
    )
    payload["detector"]["version"] = version
    payload["embedder"]["version"] = version
    payload["classifier_policy"]["version"] = version
    verification = payload.get("classifier_verification")
    if isinstance(verification, dict):
        verification["independent_embedder"]["version"] = version
    return RuntimePackageV2Metadata.model_validate(payload)


def add_detector_crowding_policy(
    source_runtime_dir: Path,
    output_dir: Path,
    *,
    version: str = "0.1.4",
    crowding_policy: DetectorCrowdingPolicyMetadata | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = load_runtime_package_v2(source_runtime_dir)
    metadata = candidate_metadata(
        source.metadata,
        version=version,
        crowding_policy=crowding_policy,
    )
    source_payload_hashes = {
        filename: sha256_file(source.root / filename) for filename in source.metadata.checksums
    }

    shutil.copytree(source.root, output_dir)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_runtime_package_v2(output_dir)
    output_payload_hashes = {
        filename: sha256_file(output_dir / filename) for filename in loaded.metadata.checksums
    }
    if output_payload_hashes != source_payload_hashes:
        raise ValueError("runtime conversion modified immutable payload files")
    manifest = directory_content_manifest(output_dir)
    return {
        "schema_version": "1.0",
        "operation": "add_detector_crowding_policy",
        "version": version,
        "source_runtime": source.root.as_posix(),
        "output_runtime": output_dir.resolve().as_posix(),
        "detector_crowding": loaded.metadata.detector_crowding.model_dump(mode="json"),
        "payload_sha256": dict(sorted(output_payload_hashes.items())),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add the Scanner 0.1.4 detector crowding policy")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="0.1.4")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = add_detector_crowding_policy(
        args.source_runtime,
        args.output_dir,
        version=args.version,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
