from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bixolon_scanner.training.models import require_torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an FCOS score-logit calibration")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classification-bias", type=float, required=True)
    parser.add_argument("--centerness-bias", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch = require_torch()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classification_key = "head.classification_head.cls_logits.bias"
    centerness_key = "head.regression_head.bbox_ctrness.bias"
    state[classification_key] = state[classification_key] + args.classification_bias
    state[centerness_key] = state[centerness_key] + args.centerness_bias
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    report = {
        "schema_version": "1.0",
        "mode": "development_score_logit_calibration",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "classification_bias": args.classification_bias,
        "centerness_bias": args.centerness_bias,
        "output_checkpoint": str(args.output),
        "output_checkpoint_sha256": _sha256(args.output),
        "policy_score_threshold_changed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
