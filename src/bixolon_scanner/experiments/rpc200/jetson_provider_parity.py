from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from ...contracts.model_package import load_model_package
from ...pipeline import DecisionPipeline
from ...runtime.imaging import decode_image
from ...runtime.onnx import build_onnx_adapters

LEVELS = ("easy", "medium", "hard")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _order_key(record: dict[str, Any]) -> str:
    value = f"rpc-validation-benchmark:{record['image_id']}".encode()
    return hashlib.sha256(value).hexdigest()


def _bbox_edges(bbox: dict[str, int]) -> tuple[int, int, int, int]:
    return (
        int(bbox["x"]),
        int(bbox["y"]),
        int(bbox["x"]) + int(bbox["width"]),
        int(bbox["y"]) + int(bbox["height"]),
    )


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("status", "reason_codes"):
        if reference[field] != candidate[field]:
            errors.append(f"image {field} differs")
    left_items = reference.get("segmentations", reference.get("items", []))
    right_items = candidate.get("segmentations", candidate.get("items", []))
    if len(left_items) != len(right_items):
        return [*errors, "segmentation count differs"]
    for index, (left, right) in enumerate(zip(left_items, right_items, strict=True)):
        prefix = f"segmentation[{index}]"
        for field in ("status", "reason_codes", "prediction"):
            if left[field] != right[field]:
                errors.append(f"{prefix} {field} differs")
        if [item["class_id"] for item in left["top3"]] != [
            item["class_id"] for item in right["top3"]
        ]:
            errors.append(f"{prefix} Top-3 rank differs")
        bbox_error = max(
            abs(a - b)
            for a, b in zip(_bbox_edges(left["bbox"]), _bbox_edges(right["bbox"]), strict=True)
        )
        if bbox_error > 1:
            errors.append(f"{prefix} bbox differs by {bbox_error}px")
        if abs(float(left["confidence"]) - float(right["confidence"])) > 0.01:
            errors.append(f"{prefix} confidence differs by more than 0.01")
        if len(left["top3"]) == len(right["top3"]):
            for rank, (left_rank, right_rank) in enumerate(
                zip(left["top3"], right["top3"], strict=True), start=1
            ):
                if abs(float(left_rank["confidence"]) - float(right_rank["confidence"])) > 0.01:
                    errors.append(f"{prefix} Top-{rank} confidence differs by more than 0.01")
    return errors


def _scan_provider(
    provider: str,
    package: Any,
    records: list[dict[str, Any]],
    encoded: dict[int, bytes],
    *,
    detector_engine: Path | None = None,
    classifier_engine: Path | None = None,
) -> dict[int, dict[str, Any]]:
    if provider.startswith("tensorrt"):
        from .tensorrt_native import build_tensorrt_adapters

        detector, classifier, selected = build_tensorrt_adapters(
            package, detector_engine, classifier_engine
        )
    else:
        detector, classifier, selected = build_onnx_adapters(package, provider)
    if selected != provider:
        raise RuntimeError(
            f"provider fallback is forbidden: requested={provider}, selected={selected}"
        )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
    )
    output: dict[int, dict[str, Any]] = {}
    for record in records:
        image_id = int(record["image_id"])
        image = decode_image(
            encoded[image_id],
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        response = pipeline.scan(image, request_id=f"parity-{provider}-{image_id}")
        output[image_id] = response.model_dump(mode="json")
    del pipeline, detector, classifier
    gc.collect()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-per-level", type=int, default=30)
    parser.add_argument("--detector-engine", type=Path, required=True)
    parser.add_argument("--classifier-engine", type=Path, required=True)
    parser.add_argument(
        "--candidate-mode",
        choices=("both", "detector", "classifier"),
        default="both",
    )
    args = parser.parse_args()

    source = [row for row in _read_jsonl(args.manifest) if row["role"] == "selection"]
    records: list[dict[str, Any]] = []
    for level in LEVELS:
        selected = sorted((row for row in source if row["level"] == level), key=_order_key)[
            : args.images_per_level
        ]
        if len(selected) != args.images_per_level:
            raise ValueError(f"insufficient {level} parity images")
        records.extend(selected)
    encoded = {
        int(record["image_id"]): (args.dataset_root / record["image_path"]).read_bytes()
        for record in records
    }
    package = load_model_package(args.package_dir)
    reference = _scan_provider("cuda", package, records, encoded)
    if args.candidate_mode == "both":
        candidate_provider = "tensorrt"
        detector_engine = args.detector_engine
        classifier_engine = args.classifier_engine
    elif args.candidate_mode == "detector":
        candidate_provider = "tensorrt-detector"
        detector_engine = args.detector_engine
        classifier_engine = None
    else:
        candidate_provider = "tensorrt-classifier"
        detector_engine = None
        classifier_engine = args.classifier_engine
    candidate = _scan_provider(
        candidate_provider,
        package,
        records,
        encoded,
        detector_engine=detector_engine,
        classifier_engine=classifier_engine,
    )

    failures: list[dict[str, Any]] = []
    for record in records:
        image_id = int(record["image_id"])
        errors = _compare(reference[image_id], candidate[image_id])
        if errors:
            failures.append(
                {
                    "image_id": image_id,
                    "level": record["level"],
                    "errors": errors,
                }
            )
    report = {
        "contract": "rpc200-v18-jetson-cuda-tensorrt-parity",
        "reference_provider": "cuda",
        "candidate_provider": candidate_provider,
        "images_per_level": args.images_per_level,
        "sample_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "passes": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
