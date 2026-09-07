from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.training.store2_dataset import export_complete_classifier_support


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export source-only classifier support with border-clipped captures removed"
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = export_complete_classifier_support(
        args.prepared_root,
        args.output_manifest,
        args.output_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
