from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ...contracts.runtime_package_v2 import load_runtime_package_v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approval_rule(value: str) -> tuple[int, float]:
    try:
        count_text, score_text = value.split(":", maxsplit=1)
        count = int(count_text)
        score = float(score_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("approval rule must be COUNT:MAXIMUM_SCORE") from exc
    if count < 1 or not 0.0 <= score <= 1.0:
        raise argparse.ArgumentTypeError("approval rule values are outside their valid ranges")
    return count, score


def _render_approval_rule(
    count: int,
    score: float,
    *,
    review_without_detector_disagreement: bool,
) -> dict:
    rule = {
        "minimum_detection_count": count,
        "maximum_approval_score": score,
    }
    if review_without_detector_disagreement:
        rule.update(
            maximum_detection_count=count,
            require_detector_disagreement=False,
        )
    return rule


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if not args.primary_embedder.is_file():
        raise FileNotFoundError(args.primary_embedder)
    if args.primary_input_size < 64:
        raise ValueError("primary classifier input size must be at least 64")

    shutil.copytree(args.source_runtime, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_embedder_metadata = dict(metadata["embedder"])
    source_embedder_path = args.output_dir / source_embedder_metadata["filename"]
    fallback_embedder_path = args.output_dir / args.fallback_filename
    if fallback_embedder_path.exists():
        raise FileExistsError(f"fallback embedder already exists: {fallback_embedder_path}")

    source_embedder_sha256 = _sha256(source_embedder_path)
    shutil.copy2(source_embedder_path, fallback_embedder_path)
    shutil.copy2(args.primary_embedder, source_embedder_path)
    metadata["embedder"]["input_size"] = [args.primary_input_size, args.primary_input_size]
    fallback_embedder_metadata = dict(source_embedder_metadata)
    fallback_embedder_metadata["filename"] = args.fallback_filename
    metadata["classifier_resolution_fallback"] = {
        "embedder": fallback_embedder_metadata,
        "fallback_on_unknown": args.fallback_on_unknown,
        "fallback_on_unsafe": args.fallback_on_unsafe,
        "minimum_detector_support": args.minimum_detector_support,
        "approval_disagreement_rules": [
            _render_approval_rule(
                count,
                score,
                review_without_detector_disagreement=False,
            )
            for count, score in args.approval_rule
        ]
        + [
            _render_approval_rule(
                count,
                score,
                review_without_detector_disagreement=True,
            )
            for count, score in args.approval_review_rule
        ],
    }
    classifier_source = metadata.setdefault("sources", {}).setdefault("embedder", {})
    classifier_source["architecture"] = (
        f"DINOv3 ConvNeXt-Tiny selective {args.primary_input_size} to "
        f"{source_embedder_metadata['input_size'][0]} resolution cascade"
    )

    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    runtime = load_runtime_package_v2(args.output_dir)
    fallback = runtime.metadata.classifier_resolution_fallback
    if fallback is None or fallback.embedder.input_size != tuple(
        source_embedder_metadata["input_size"]
    ):
        raise RuntimeError("classifier fallback metadata was not preserved")

    return {
        "schema_version": "1.0",
        "operation": "build_classifier_resolution_cascade_runtime",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "primary_input_size": [args.primary_input_size, args.primary_input_size],
        "fallback_input_size": source_embedder_metadata["input_size"],
        "primary_embedder_sha256": _sha256(source_embedder_path),
        "fallback_embedder_sha256": _sha256(fallback_embedder_path),
        "source_embedder_sha256": source_embedder_sha256,
        "weights_modified": False,
        "fallback_on_unknown": args.fallback_on_unknown,
        "fallback_on_unsafe": args.fallback_on_unsafe,
        "minimum_detector_support": args.minimum_detector_support,
        "approval_disagreement_rules": [
            _render_approval_rule(
                count,
                score,
                review_without_detector_disagreement=False,
            )
            for count, score in args.approval_rule
        ]
        + [
            _render_approval_rule(
                count,
                score,
                review_without_detector_disagreement=True,
            )
            for count, score in args.approval_review_rule
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a selective classifier-size cascade")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--primary-embedder", type=Path, required=True)
    parser.add_argument("--primary-input-size", type=int, required=True)
    parser.add_argument("--fallback-filename", default="embedder-fallback-224.onnx")
    parser.add_argument(
        "--fallback-on-unknown",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fallback-on-unsafe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-detector-support", type=int, default=3)
    parser.add_argument(
        "--approval-rule",
        type=_approval_rule,
        action="append",
        default=None,
        metavar="COUNT:MAXIMUM_SCORE",
    )
    parser.add_argument(
        "--approval-review-rule",
        type=_approval_rule,
        action="append",
        default=None,
        metavar="EXACT_COUNT:MAXIMUM_SCORE",
        help="review low-score approvals even when the detector class agrees",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.approval_rule is None:
        args.approval_rule = [(1, 0.5), (5, 0.6), (6, 0.76)]
    if args.approval_review_rule is None:
        args.approval_review_rule = []
    report = build(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
