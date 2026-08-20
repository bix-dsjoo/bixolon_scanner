from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bread_runtime_parity import compare_runtime_traces


def _decisions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    decisions = []
    for row in rows:
        decision = dict(row["decision"])
        decision["image_id"] = row["image_id"]
        decisions.append(decision)
    return decisions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare Scanner 2.0 CPU/CUDA decisions")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.999)
    parser.add_argument(
        "--maximum-confidence-error",
        type=float,
        default=0.005,
        help="Normalized approval/ranking score tolerance; decisions must still match exactly",
    )
    args = parser.parse_args(argv)
    report = compare_runtime_traces(
        _decisions(args.reference),
        _decisions(args.candidate),
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_confidence_error=args.maximum_confidence_error,
    )
    report.update(
        schema_version="2.0",
        evaluation="scanner_2_0_runtime_cpu_cuda_parity",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
