"""Offline development diagnosis of the shared-feature ROI multiplicity head."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..configuration import load_json_config
from ..contracts import ItemStatus, ScanResponse
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import EmbedderMetadata
from ..pipeline.ports import Detection
from ..pipeline.segmentation import summarize_reason_codes
from ..training.input_isolation import protect_development
from ..training.models import build_dino_classifier
from ..training.roi_integrity import build_head, prepare_roi
from ..training.three_bakery_data import read_jsonl, write_json, write_jsonl
from .three_bakery import score_response, summarize


def diagnose(config_path: Path, architecture: str, method: str, recipe: str, seed: int) -> dict:
    import torch

    settings = load_json_config(config_path)
    work = Path(settings["source_work"]).resolve()
    protect_development(Path(settings["source_config"]), work)
    model_dir = work / f"models/{method}-{recipe}-{seed}"
    head_dir = work / f"roi-integrity/{method}-{recipe}-{seed}"
    candidate = work / f"candidates/{architecture}-{method}-{recipe}-{seed}"
    contract = load_json_config(model_dir / "contract.json")
    head_report = load_json_config(head_dir / "report.json")
    if sha256_file(head_dir / "head.pt") != head_report["head_sha256"]:
        raise ValueError("integrity head checksum mismatch")
    metadata = EmbedderMetadata.model_validate(
        load_json_config(candidate / "runtime/metadata.json")["embedder"]
    )
    torch.set_num_threads(4)
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=Path(contract["weights"]),
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    head = build_head(head_report["feature_dimension"], settings)
    head.load_state_dict(torch.load(head_dir / "head.pt", map_location="cpu", weights_only=True))
    head = head.cuda().eval()
    result = {
        "scope": "offline development hypothesis; public rounded boxes; latency not measured",
        "head_sha256": head_report["head_sha256"],
        "threshold": settings["multi_object_probability_threshold"],
    }
    source_config = load_json_config(Path(settings["source_config"]))
    diagnostic_seed = source_config["synthetic"]["diagnostic"]["seed"]
    for group, manifest in (
        ("real", work / "prepared/original_detection.jsonl"),
        ("stress", work / f"prepared/diagnostic-{diagnostic_seed}/manifest.jsonl"),
    ):
        records = {r["image_id"]: r for r in read_jsonl(manifest)}
        rows = []
        for row in read_jsonl(candidate / f"diagnostic-{group}/responses.jsonl"):
            record = records[row["image_id"]]
            response = ScanResponse.model_validate(row["response"])
            probabilities = []
            if response.segmentations:
                detections = [
                    Detection(
                        s.bbox.x, s.bbox.y, s.bbox.x + s.bbox.width, s.bbox.y + s.bbox.height, 1.0
                    )
                    for s in response.segmentations
                ]
                path = Path(record["image_path"])
                if sha256_file(path) != record["image_sha256"]:
                    raise ValueError("development diagnostic image changed")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    tensors = [
                        prepare_roi(
                            image,
                            [d] + [other for j, other in enumerate(detections) if i != j],
                            metadata,
                        )
                        for i, d in enumerate(detections)
                    ]
                with torch.inference_mode():
                    features = model.extract_features(torch.from_numpy(np.stack(tensors)).cuda())
                    probabilities = torch.softmax(head(features), dim=-1)[:, 1].cpu().tolist()
                for item, probability in zip(response.segmentations, probabilities, strict=True):
                    if probability >= settings["multi_object_probability_threshold"]:
                        item.status = ItemStatus.SEGMENT_RECAPTURE
                        item.reason_codes = ["SEGMENT_RECAPTURE_REQUIRED"]
                        item.prediction = None
                        item.top3 = []
                        item.confidence = 0.0
                # Restore validated enums after creating the counterfactual response.
                response = ScanResponse.model_validate(
                    {
                        **response.model_dump(mode="json"),
                        "reason_codes": summarize_reason_codes(
                            [
                                type(s).model_validate(s.model_dump(mode="json"))
                                for s in response.segmentations
                            ]
                        ),
                    }
                )
            rows.append(
                {
                    "image_id": row["image_id"],
                    "multi_object_probabilities": probabilities,
                    "response": response.model_dump(mode="json"),
                    "elapsed_ms": 0.0,
                    "before": row["metrics"],
                    "metrics": score_response(response, record["annotations"]),
                }
            )
        summary = summarize(rows)
        summary.pop("latency")
        summary["previously_correct_approvals_blocked"] = sum(
            max(0, r["before"]["correct_approved_count"] - r["metrics"]["correct_approved_count"])
            for r in rows
        )
        summary["previous_wrong_approvals_blocked"] = sum(
            r["before"]["wrong_approved_count"] - r["metrics"]["wrong_approved_count"] for r in rows
        )
        if group == "real":
            multi = summarize([r for r in rows if records[r["image_id"]].get("kind") == "multi"])
            multi.pop("latency")
            summary["multi"] = multi
        result[group] = summary
        write_jsonl(
            head_dir / f"diagnostic-{architecture}-{group}.jsonl",
            [{key: value for key, value in row.items() if key != "elapsed_ms"} for row in rows],
        )
        print(
            group,
            "complete",
            summary["complete_images"],
            "wrong",
            summary["wrong_approved_count"],
            "correct approvals blocked",
            summary["previously_correct_approvals_blocked"],
            flush=True,
        )
    write_json(head_dir / f"diagnostic-{architecture}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bread/three_bakery_roi_integrity.json"),
    )
    parser.add_argument("--architecture", choices=["ssdlite", "dfine"], required=True)
    parser.add_argument("--method", choices=["frozen", "finetune", "margin"], required=True)
    parser.add_argument("--recipe", choices=["basic", "dense"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    diagnose(args.config, args.architecture, args.method, args.recipe, args.seed)


if __name__ == "__main__":
    main()
