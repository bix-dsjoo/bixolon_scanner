from __future__ import annotations

import argparse
from pathlib import Path

from bixolon_scanner.training.bread_cv import write_bread_operational_evaluation_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one checksummed bread operational evaluation manifest"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        write_bread_operational_evaluation_manifest(
            args.dataset_root,
            args.collection,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
