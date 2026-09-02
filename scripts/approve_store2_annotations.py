from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.training.store2_dataset import approve_store2_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind manual QA to store-2 annotations")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--findings", required=True)
    args = parser.parse_args()
    review = approve_store2_annotations(
        args.prepared_root,
        reviewer=args.reviewer,
        reviewed_class_ids=list(range(1, 21)),
        findings=args.findings,
    )
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
