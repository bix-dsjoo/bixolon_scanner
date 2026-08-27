from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.runtime_package_v2 import DetectorCrowdingPolicyMetadata
from .scanner_014_runtime import add_detector_crowding_policy, default_crowding_policy


def corroborated_large_proposal_policy() -> DetectorCrowdingPolicyMetadata:
    """Keep the 0.1.4 crowding policy but require evidence beyond proposal size."""
    payload = default_crowding_policy().model_dump(mode="json")
    payload["large_proposal_corroboration"] = {
        "query_containment_surplus_minimum": 1,
        "selected_center_minimum": 2,
        "selected_count_maximum": 5,
    }
    return DetectorCrowdingPolicyMetadata.model_validate(payload)


def create_scanner_015_runtime(
    source_runtime_dir: Path,
    output_dir: Path,
    *,
    version: str = "0.1.5",
) -> dict[str, Any]:
    return add_detector_crowding_policy(
        source_runtime_dir,
        output_dir,
        version=version,
        crowding_policy=corroborated_large_proposal_policy(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add corroboration to the Scanner 0.1.5 large-proposal crowding gate"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="0.1.5")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = create_scanner_015_runtime(
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
