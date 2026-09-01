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


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.input_size <= 0 or args.input_size % 32:
        raise ValueError("input size must be a positive multiple of 32")

    shutil.copytree(args.source_runtime, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ensemble = metadata["detector"]["ensemble"]
    members = ensemble["members"]
    matching_members = [member for member in members if member["filename"] == args.primary_filename]
    if len(matching_members) != 1:
        raise ValueError(f"expected exactly one ensemble member named {args.primary_filename!r}")

    destination_model = args.output_dir / args.primary_filename
    fallback_model = args.output_dir / args.fallback_filename
    if fallback_model.exists():
        raise FileExistsError(f"fallback model already exists: {fallback_model}")
    source_model_sha256 = _sha256(destination_model)
    exported_model_sha256 = _sha256(args.primary_model)
    shutil.copy2(destination_model, fallback_model)
    shutil.copy2(args.primary_model, destination_model)
    fallback_member = matching_members[0]
    primary_member = dict(fallback_member)
    fallback_member["filename"] = args.fallback_filename
    fallback_member["ensemble_fallback"] = True
    primary_member["filename"] = args.primary_filename
    primary_member["input_size"] = [args.input_size, args.input_size]
    primary_member["ensemble_fallback"] = False
    members.append(primary_member)

    selector = ensemble["class_verified_selector"]
    if selector["low_resolution_primary_member_filename"] != args.primary_filename:
        raise ValueError("primary selector member does not match the requested filename")
    source_score_threshold = selector["low_resolution_score_threshold"]
    selector["low_resolution_score_threshold"] = args.score_threshold
    selector["low_resolution_primary_minimum_dimension"] = args.primary_minimum_dimension
    selector["low_resolution_small_image_primary_member_filename"] = args.fallback_filename
    selector["low_resolution_small_image_score_threshold"] = source_score_threshold
    selector["low_resolution_risky_approval_ensemble_fallback"] = True
    selector["low_resolution_risky_approval_maximum_score"] = args.risky_approval_maximum_score
    selector["low_resolution_risky_approval_minimum_aspect_ratio"] = (
        args.risky_approval_minimum_aspect_ratio
    )

    detector_source = metadata.setdefault("sources", {}).setdefault("detector", {})
    detector_source["architecture"] = (
        "D-FINE-N HGNetv2 mixed-resolution cascade: "
        f"{args.primary_filename} at {args.input_size}, 640 ensemble fallback"
    )

    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    runtime = load_runtime_package_v2(args.output_dir)
    runtime_member = next(
        member
        for member in runtime.metadata.detector.ensemble.members
        if member.filename == args.primary_filename
    )
    if runtime_member.input_size != (args.input_size, args.input_size):
        raise RuntimeError("mixed-resolution member input size was not preserved")

    return {
        "schema_version": "1.0",
        "operation": "build_yolo_free_mixed_resolution_runtime",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "primary_filename": args.primary_filename,
        "fallback_filename": args.fallback_filename,
        "source_model_sha256": source_model_sha256,
        "exported_model": args.primary_model.resolve().as_posix(),
        "exported_model_sha256": exported_model_sha256,
        "installed_model_sha256": _sha256(destination_model),
        "fallback_model_sha256": _sha256(fallback_model),
        "checkpoint_weights_changed": False,
        "graph_change": "fixed input and decoder anchors re-exported for the requested size",
        "policy_changes": {
            "primary_input_size": [args.input_size, args.input_size],
            "low_resolution_score_threshold": args.score_threshold,
            "low_resolution_primary_minimum_dimension": args.primary_minimum_dimension,
            "low_resolution_small_image_primary_member_filename": args.fallback_filename,
            "low_resolution_small_image_score_threshold": source_score_threshold,
            "low_resolution_risky_approval_ensemble_fallback": True,
            "low_resolution_risky_approval_maximum_score": (args.risky_approval_maximum_score),
            "low_resolution_risky_approval_minimum_aspect_ratio": (
                args.risky_approval_minimum_aspect_ratio
            ),
            "fallback_member_input_size": metadata["detector"]["input_size"],
            "fallback_member_count": sum(
                bool(member.get("ensemble_fallback", True)) for member in members
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a mixed-resolution YOLO-free detector runtime"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--primary-model", type=Path, required=True)
    parser.add_argument("--primary-filename", default="detector-production.onnx")
    parser.add_argument("--fallback-filename", default="detector-production-640.onnx")
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--primary-minimum-dimension", type=int, default=1000)
    parser.add_argument("--risky-approval-maximum-score", type=float, default=0.5)
    parser.add_argument("--risky-approval-minimum-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
