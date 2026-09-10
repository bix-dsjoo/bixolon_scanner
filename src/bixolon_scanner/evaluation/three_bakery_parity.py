"""PyTorch/ORT parity through the same pipeline, confined to development diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..runtime.imaging import decode_image
from ..runtime.onnx_session import OrtRunner
from ..runtime.preprocessing import prepare_rgb
from ..training.models import build_dino_classifier
from ..training.three_bakery_data import read_jsonl, write_json
from ..worker.runtime_factory import build_worker_runtime
from ..worker.settings import WorkerSettings
from .three_bakery import match_boxes


def matched_detector_outputs(names: list[str], actual: list, expected: list) -> dict | None:
    """Keep raw-query checks separate from class-blind, bbox-aligned D-FINE checks."""
    if "pred_boxes" not in names or "logits" not in names:
        return None
    box_index = names.index("pred_boxes")
    logit_index = names.index("logits")
    left, right = actual[box_index], expected[box_index]
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("detector parity query shape mismatch")
    batches = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):

        def corners(boxes):
            return np.concatenate(
                [boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=1
            ).tolist()

        matches = match_boxes(corners(a), corners(b), 0.5)
        i = list(matches)
        j = list(matches.values())
        pairs = [
            ("boxes", a[i], b[j]),
            ("logits", actual[logit_index][index, i], expected[logit_index][index, j]),
        ]
        batches.append(
            {
                "query_count": len(a),
                "matched_count": len(matches),
                "complete_correspondence": len(matches) == len(a),
                "reordered_query_count": sum(x != y for x, y in matches.items()),
                "outputs": {
                    name: {
                        "max_absolute_error": float(np.max(np.abs(x - y))) if len(i) else None,
                        "allclose": bool(len(i) and np.allclose(x, y, atol=1e-4, rtol=1e-3)),
                    }
                    for name, x, y in pairs
                },
            }
        )
    return {"matching": "class_blind_IoU_0.5_max_cardinality_then_total_IoU", "batches": batches}


def torch_models(work: Path, freeze: dict):
    import torch

    from ..training.ssdlite_objectness_detector import (
        build_export_wrapper,
        build_ssdlite_objectness,
    )
    from ..training.three_bakery_detector import dfine_model

    torch.set_num_threads(4)
    candidate = freeze["candidate"]
    suffix = f"{candidate['recipe']}-{freeze['seed']}"
    detector_dir = work / f"models/{candidate['architecture']}-{suffix}"
    classifier_dir = work / f"models/{candidate['method']}-{suffix}"
    detector_contract = load_json_config(detector_dir / "contract.json")
    classifier_contract = load_json_config(classifier_dir / "contract.json")
    if candidate["architecture"] == "ssdlite":
        detector = build_ssdlite_objectness(device="cpu")
        detector.load_state_dict(
            torch.load(detector_dir / "model.pt", map_location="cpu", weights_only=True)
        )
        detector = build_export_wrapper(detector.eval(), detector_class_count=1)
        detector_forward = detector
    else:
        # The D-FINE repository is an explicit experiment dependency, never a Worker dependency.
        weights = Path(detector_contract["weights"])
        detector, _criterion, _transfer = dfine_model(Path(freeze["dfine_repository"]), weights)
        detector.load_state_dict(
            torch.load(detector_dir / "model.pt", map_location="cpu", weights_only=True)
        )
        detector = detector.cpu().eval().deploy()

        def detector_forward(values):
            result = detector(values)
            return result["pred_logits"], result["pred_boxes"]

    classifier = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=Path(classifier_contract["weights"]),
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    )
    classifier.load_state_dict(
        torch.load(classifier_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    classifier.eval()
    verifier_report = load_json_config(work / "models/verifier/report.json")
    verifier_weights = Path(freeze["verifier_weights"])
    if sha256_file(verifier_weights) != verifier_report["weights_sha256"]:
        raise ValueError("verifier foundation weights differ from the exported model")
    verifier = build_dino_classifier(
        "dinov3_vitb16", 20, weights_path=verifier_weights, feature_l2_normalize=True
    ).eval()
    metadata = load_json_config(Path(freeze["candidate_path"]) / "runtime/metadata.json")

    def classify(values):
        return (torch.nn.functional.normalize(classifier(values), dim=-1),)

    primary_forward = classify
    if "roi_integrity_config" in freeze:
        from ..training.roi_integrity import build_head

        settings = load_json_config(Path(freeze["roi_integrity_config"]))
        head_path = Path(freeze["roi_integrity_head"])
        if sha256_file(head_path) != freeze["roi_integrity_head_sha256"]:
            raise ValueError("ROI integrity head changed before parity")
        head_report = load_json_config(head_path.parent / "report.json")
        head = build_head(head_report["feature_dimension"], settings)
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        head.eval()

        def primary_forward(values):
            features = classifier.extract_features(values)
            return (
                torch.nn.functional.normalize(classifier.classifier(features), dim=-1),
                torch.softmax(head(features), dim=-1)[:, 1],
            )

    return {
        metadata["detector"]["filename"]: detector_forward,
        metadata["embedder"]["filename"]: primary_forward,
        metadata["classifier_resolution_fallback"]["embedder"]["filename"]: classify,
        metadata["classifier_verification"]["independent_embedder"]["filename"]: lambda values: (
            verifier.extract_features(values),
        ),
    }


def check(work: Path) -> dict:
    import torch

    from .three_bakery_http import response_signature

    freeze = load_json_config(work / "final-candidate.json")
    candidate = Path(freeze["candidate_path"])
    metadata = load_json_config(candidate / "runtime/metadata.json")
    functions = torch_models(work, freeze)
    tensor_checks = defaultdict(list)
    matched_checks = []
    fixture_directory = work / "parity/tensors"
    fixture_directory.mkdir(parents=True, exist_ok=True)
    fixtures = []
    call_counts = defaultdict(int)
    force_fixture = False

    class TorchDiagnosticRunner:
        cuda = accelerated = False

        def __init__(self, model_path, provider, cuda_dll_dir=None, **kwargs):
            self.name = Path(model_path).name
            self.forward = functions[self.name]
            self.reference = OrtRunner(model_path, "cpu", cpu_intra_op_threads=4)

        def run(self, output_names, input_name, tensor):
            with torch.inference_mode():
                result = self.forward(torch.from_numpy(tensor.copy()))
            actual = [value.detach().numpy() for value in result]
            if (
                self.name == metadata["embedder"]["filename"]
                and metadata["embedder"].get("multi_object_output_name") is not None
            ):
                named = dict(
                    zip(
                        [
                            metadata["embedder"]["output_name"],
                            metadata["embedder"]["multi_object_output_name"],
                        ],
                        actual,
                        strict=True,
                    )
                )
                actual = [named[name] for name in output_names]
            expected = self.reference.run(output_names, input_name, tensor)
            matched = (
                matched_detector_outputs(output_names, actual, expected)
                if freeze["candidate"]["architecture"] == "dfine"
                else None
            )
            if matched is not None:
                matched_checks.append(matched)
            ordinal = call_counts[self.name]
            call_counts[self.name] += 1
            if force_fixture or (ordinal % 20 == 0 and ordinal < 500):
                fixture = fixture_directory / f"{self.name}-{ordinal:04d}.npz"
                np.savez_compressed(
                    fixture,
                    input=tensor,
                    **{f"output_{i}": value for i, value in enumerate(expected)},
                )
                fixtures.append(
                    {
                        "model": self.name,
                        "input_name": input_name,
                        "output_names": output_names,
                        "file": fixture.name,
                        "sha256": sha256_file(fixture),
                        "scope": "explicit_source_crop" if force_fixture else "source_pipeline",
                    }
                )
            for name, left, right in zip(output_names, actual, expected, strict=True):
                if left.shape != right.shape:
                    raise ValueError("PyTorch/ORT output shape mismatch")
                rank_equal = None
                if name == "embeddings" and left.shape[-1] == 20:
                    rank_equal = bool(
                        np.array_equal(
                            np.argsort(-left, axis=-1, kind="stable"),
                            np.argsort(-right, axis=-1, kind="stable"),
                        )
                    )
                tensor_checks[self.name].append(
                    {
                        "output": name,
                        "shape": list(left.shape),
                        "max_absolute_error": float(np.max(np.abs(left - right))),
                        "allclose": bool(np.allclose(left, right, atol=1e-4, rtol=1e-3)),
                        "class_rank_equal": rank_equal,
                    }
                )
            return actual

        def close(self):
            self.reference.close()

    settings = WorkerSettings(
        package_dir=candidate / "runtime",
        catalog_dir=candidate / "catalog",
        catalog_store_id="three_bakery",
        provider="cpu",
        cpu_detector_intra_op_threads=freeze.get("cpu_profile", [4, 4])[0],
        cpu_embedder_intra_op_threads=freeze.get("cpu_profile", [4, 4])[1],
    )
    records = read_jsonl(work / "prepared/original_detection.jsonl")
    expected = {}
    runtime = build_worker_runtime(settings)
    try:
        for record in records:
            with decode_image(
                Path(record["image_path"]).read_bytes(),
                max_bytes=20_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.jpeg_draft_size,
            ) as image:
                response = runtime.pipeline.scan(
                    image, request_id=f"ort-parity-{record['image_id']}"
                )
                expected[record["image_id"]] = response.model_dump(mode="json")
    finally:
        runtime.close()
    mismatches = []
    mismatch_details = []
    with (
        patch("bixolon_scanner.runtime.onnx.OrtRunner", TorchDiagnosticRunner),
        patch("bixolon_scanner.runtime.catalog.OrtRunner", TorchDiagnosticRunner),
        patch("bixolon_scanner.runtime.detector_v2.OrtRunner", TorchDiagnosticRunner),
    ):
        runtime = build_worker_runtime(settings)
        try:
            for index, record in enumerate(records):
                with decode_image(
                    Path(record["image_path"]).read_bytes(),
                    max_bytes=20_000_000,
                    max_pixels=50_000_000,
                    jpeg_draft_size=runtime.jpeg_draft_size,
                ) as image:
                    response = runtime.pipeline.scan(
                        image, request_id=f"torch-parity-{record['image_id']}"
                    )
                    if response_signature(response.model_dump(mode="json")) != response_signature(
                        expected[record["image_id"]]
                    ):
                        mismatches.append(record["image_id"])
                        mismatch_details.append(
                            {
                                "image_id": record["image_id"],
                                "onnx_response": expected[record["image_id"]],
                                "pytorch_response": response.model_dump(mode="json"),
                            }
                        )
                if (index + 1) % 50 == 0:
                    print(
                        f"PyTorch/ORT source pipeline parity {index + 1}/{len(records)}", flush=True
                    )
        finally:
            runtime.close()
    # Exercise every embedder on actual source pixels even when selective routing
    # does not reach it. This is an offline tensor check, not a production routing change.
    metadata = load_json_config(candidate / "runtime/metadata.json")
    crops = {}
    for row in read_jsonl(work / "prepared/original_crops.jsonl"):
        crops.setdefault(row["category_id"], row)
    force_fixture = True
    for embedder in (
        metadata["embedder"],
        metadata["classifier_resolution_fallback"]["embedder"],
        metadata["classifier_verification"]["independent_embedder"],
    ):
        runner = TorchDiagnosticRunner(candidate / "runtime" / embedder["filename"], "cpu")
        try:
            for category in sorted(crops):
                row = crops[category]
                path = Path(row["image_path"])
                if sha256_file(path) != row["image_sha256"]:
                    raise ValueError("source crop changed before parity measurement")
                with Image.open(path) as opened:
                    tensor = prepare_rgb(
                        opened.convert("RGB"),
                        tuple(embedder["input_size"]),
                        tuple(embedder["mean"]),
                        tuple(embedder["std"]),
                        reducing_gap=embedder["resize_reducing_gap"],
                    )[None]
                runner.run([embedder["output_name"]], embedder["input_name"], tensor)
        finally:
            runner.close()
    if set(tensor_checks) != set(functions):
        raise ValueError("parity did not exercise every exported ONNX model")
    result = {
        "source_manifest_sha256": sha256_file(work / "prepared/original_detection.jsonl"),
        "freeze_sha256": sha256_file(work / "final-candidate.json"),
        "image_count": len(records),
        "status_rank_mismatch_image_ids": mismatches,
        "status_rank_mismatch_details": mismatch_details,
        "status_rank_parity": not mismatches,
        "tensor_tolerance": {"atol": 1e-4, "rtol": 1e-3},
        "tensor_checks": dict(tensor_checks),
        "bbox_aligned_detector_checks": matched_checks,
        "model_call_counts": dict(call_counts),
        "explicit_source_crop_classes": sorted(crops),
        "source_crop_manifest_sha256": sha256_file(work / "prepared/original_crops.jsonl"),
        "all_exported_models_exercised": True,
        "all_tensors_close": all(
            v["allclose"] for checks in tensor_checks.values() for v in checks
        ),
        "all_classifier_ranks_equal": all(
            v["class_rank_equal"] is not False for checks in tensor_checks.values() for v in checks
        ),
        "scope": "same_source_diagnostic_PyTorch_CPU_vs_ORT_CPU; production_Worker_remains_ORT_only",
    }
    write_json(work / "parity/pytorch-ort.json", result)
    write_json(
        work / "parity/tensor-fixtures.json",
        {"freeze_sha256": result["freeze_sha256"], "fixtures": fixtures},
    )
    return result


def check_cuda_tensors(work: Path, cuda_dll_dir: Path) -> dict:
    freeze = load_json_config(work / "final-candidate.json")
    fixture_report = load_json_config(work / "parity/tensor-fixtures.json")
    if fixture_report["freeze_sha256"] != sha256_file(work / "final-candidate.json"):
        raise ValueError("CPU/CUDA tensor fixtures refer to a different candidate")
    directory = Path(freeze["candidate_path"]) / "runtime"
    checks = []
    matched_checks = []
    for model in sorted({row["model"] for row in fixture_report["fixtures"]}):
        runner = OrtRunner(directory / model, "cuda", cuda_dll_dir)
        try:
            for row in (r for r in fixture_report["fixtures"] if r["model"] == model):
                path = work / "parity/tensors" / row["file"]
                if sha256_file(path) != row["sha256"]:
                    raise ValueError("tensor fixture checksum mismatch")
                with np.load(path, allow_pickle=False) as fixture:
                    actual = runner.run(row["output_names"], row["input_name"], fixture["input"])
                    matched = (
                        matched_detector_outputs(
                            row["output_names"],
                            actual,
                            [fixture[f"output_{index}"] for index in range(len(actual))],
                        )
                        if freeze["candidate"]["architecture"] == "dfine"
                        else None
                    )
                    if matched is not None:
                        matched_checks.append({"fixture": row["file"], **matched})
                    for index, value in enumerate(actual):
                        expected = fixture[f"output_{index}"]
                        rank_equal = None
                        if row["output_names"][index] == "embeddings" and value.shape[-1] == 20:
                            rank_equal = bool(
                                np.array_equal(
                                    np.argsort(-value, axis=-1, kind="stable"),
                                    np.argsort(-expected, axis=-1, kind="stable"),
                                )
                            )
                        checks.append(
                            {
                                "model": model,
                                "fixture": row["file"],
                                "output": row["output_names"][index],
                                "max_absolute_error": float(np.max(np.abs(value - expected))),
                                "allclose": bool(
                                    np.allclose(value, expected, atol=1e-4, rtol=1e-3)
                                ),
                                "class_rank_equal": rank_equal,
                            }
                        )
        finally:
            runner.close()
    result = {
        "freeze_sha256": fixture_report["freeze_sha256"],
        "checks": checks,
        "bbox_aligned_detector_checks": matched_checks,
        "all_tensors_close": all(c["allclose"] for c in checks),
        "all_classifier_ranks_equal": all(c["class_rank_equal"] is not False for c in checks),
        "tensor_tolerance": {"atol": 1e-4, "rtol": 1e-3},
    }
    write_json(work / "parity/ort-cpu-cuda.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    args = parser.parse_args()
    if args.cuda_dll_dir:
        check_cuda_tensors(args.work.resolve(), args.cuda_dll_dir)
    else:
        check(args.work.resolve())


if __name__ == "__main__":
    main()
