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
    if not 0.0 <= args.ambiguity_maximum_approval_score <= 1.0:
        raise ValueError("ambiguity maximum approval score must be in [0, 1]")

    source = load_runtime_package_v2(args.source_runtime)
    fallback = source.metadata.classifier_resolution_fallback
    verification = source.metadata.classifier_verification
    if source.metadata.detector_class_mode != "class_agnostic":
        raise ValueError("fixed 224 classifier candidate requires a class-agnostic detector")
    if fallback is None or source.classifier_fallback_embedder_path is None:
        raise ValueError("source runtime has no higher-resolution classifier fallback")
    if fallback.embedder.input_size != (224, 224):
        raise ValueError("source classifier fallback must use 224x224 input")
    if verification is None:
        raise ValueError("source runtime has no independent classifier verifier")

    shutil.copytree(args.source_runtime, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    primary_filename = str(metadata["embedder"]["filename"])
    fallback_payload = dict(metadata["classifier_resolution_fallback"]["embedder"])
    fallback_filename = str(fallback_payload["filename"])
    primary_path = args.output_dir / primary_filename
    fallback_path = args.output_dir / fallback_filename
    source_primary_sha256 = _sha256(primary_path)
    source_fallback_sha256 = _sha256(fallback_path)

    shutil.copy2(fallback_path, primary_path)
    if fallback_path != primary_path:
        fallback_path.unlink()
    fallback_payload["filename"] = primary_filename
    metadata["embedder"] = fallback_payload
    metadata["classifier_resolution_fallback"] = None
    classifier_verification = metadata["classifier_verification"]
    classifier_verification["ambiguity_maximum_approval_score"] = (
        args.ambiguity_maximum_approval_score
    )
    classifier_verification["unknown_recapture_on_dual_verifier_rejection"] = False
    classifier_verification["unknown_recapture_on_any_verifier_rejection"] = True
    metadata["sources"]["embedder"]["architecture"] = (
        "DINOv3 ConvNeXt-Tiny primary classifier at fixed 224x224 input"
    )
    metadata["checksums"] = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    candidate = load_runtime_package_v2(args.output_dir)
    candidate_verification = candidate.metadata.classifier_verification
    if candidate.metadata.embedder.input_size != (224, 224):
        raise RuntimeError("fixed 224 primary classifier input was not preserved")
    if candidate.metadata.classifier_resolution_fallback is not None:
        raise RuntimeError("classifier resolution fallback was not removed")
    if (
        candidate_verification is None
        or not candidate_verification.unknown_recapture_on_any_verifier_rejection
    ):
        raise RuntimeError("independent verifier UNKNOWN safety policy was not enabled")

    return {
        "schema_version": "1.0",
        "operation": "build_fixed224_classifier_runtime",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": args.output_dir.resolve().as_posix(),
        "detector_class_mode": candidate.metadata.detector_class_mode,
        "primary_input_size": list(candidate.metadata.embedder.input_size),
        "verifier_input_size": list(candidate_verification.independent_embedder.input_size),
        "ambiguity_maximum_approval_score": (
            candidate_verification.ambiguity_maximum_approval_score
        ),
        "unknown_recapture_on_any_verifier_rejection": True,
        "source_primary_embedder_sha256": source_primary_sha256,
        "source_fallback_embedder_sha256": source_fallback_sha256,
        "candidate_primary_embedder_sha256": _sha256(candidate.embedder_path),
        "model_graph_or_weight_changed": False,
        "routing_policy_changed": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a class-agnostic Runtime's 224 classifier fallback to the primary "
            "path and enable independent-verifier UNKNOWN safety"
        )
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ambiguity-maximum-approval-score", type=float, default=0.55)
    args = parser.parse_args(argv)
    report = build(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
