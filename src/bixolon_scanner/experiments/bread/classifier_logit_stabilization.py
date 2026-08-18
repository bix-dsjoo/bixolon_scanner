from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def stabilize_logits(
    values: np.ndarray,
    *,
    logit_quantum: float | None,
    logit_phase: float,
    tie_break_bias_span: float,
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("classifier logits must be a two-dimensional array")
    if logit_quantum is not None and logit_quantum <= 0.0:
        raise ValueError("logit quantum must be positive")
    if logit_quantum is None and logit_phase != 0.0:
        raise ValueError("logit phase requires a logit quantum")
    if logit_quantum is not None and not 0.0 <= logit_phase < logit_quantum:
        raise ValueError("logit phase must be in [0, logit quantum)")
    if tie_break_bias_span < 0.0:
        raise ValueError("tie-break bias span must be non-negative")
    stabilized = np.asarray(values, dtype=np.float32)
    if logit_quantum is not None:
        quantum = np.float32(logit_quantum)
        phase = np.float32(logit_phase)
        stabilized = np.round((stabilized + phase) / quantum) * quantum - phase
    if tie_break_bias_span:
        stabilized = stabilized + np.linspace(
            0.0,
            -tie_break_bias_span,
            stabilized.shape[1],
            dtype=np.float32,
        )
    return np.asarray(stabilized, dtype=np.float32)


def transform(args: argparse.Namespace) -> dict[str, Any]:
    payload = np.load(args.input)
    if "targets" not in payload.files:
        raise ValueError("input logits archive does not contain targets")
    transformed = {
        name: stabilize_logits(
            payload[name],
            logit_quantum=args.logit_quantum,
            logit_phase=args.logit_phase,
            tie_break_bias_span=args.tie_break_bias_span,
        )
        for name in payload.files
        if name in set(args.views)
    }
    transformed["targets"] = payload["targets"]
    missing = set(args.views) - set(transformed)
    if missing:
        raise ValueError(f"input logits archive is missing views: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **transformed)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_logit_stabilization",
        "input": str(args.input),
        "output": str(args.output),
        "sample_count": int(len(payload["targets"])),
        "views": args.views,
        "logit_quantum": args.logit_quantum,
        "logit_phase": args.logit_phase,
        "tie_break_bias_span": args.tie_break_bias_span,
        "uses_image_or_target_specific_rules": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a global classifier logit policy")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--logit-quantum", type=float)
    parser.add_argument("--logit-phase", type=float, default=0.0)
    parser.add_argument("--tie-break-bias-span", type=float, default=0.0)
    transform(parser.parse_args())


if __name__ == "__main__":
    main()
