"""Learn ROI multiplicity from source GT groups using a frozen current-run backbone."""

from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import EmbedderMetadata
from ..pipeline.ports import Detection
from ..runtime.geometry import box_iou_matrix
from ..runtime.onnx import (
    apply_classifier_background_masks,
    classifier_neighbor_ownership_mask,
)
from ..runtime.preprocessing import prepare_rgb
from .models import build_dino_classifier
from .three_bakery_data import read_jsonl, write_json, write_jsonl


def xyxy(annotation: dict) -> list[float]:
    x, y, width, height = annotation["bbox_xywh"]
    return [x, y, x + width, y + height]


def union_box(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def grouped_detections(boxes: list[list[float]], group: list[int]) -> list[Detection]:
    """The group becomes one ROI; other objects retain their neighbor-mask ownership."""
    if not group or len(set(group)) != len(group) or any(i < 0 or i >= len(boxes) for i in group):
        raise ValueError("ROI group must contain distinct valid annotation indices")
    merged = union_box([boxes[i] for i in group])
    return [Detection(*merged, 1.0)] + [
        Detection(*b, 1.0) for i, b in enumerate(boxes) if i not in group
    ]


def prepare_roi(
    image: Image.Image, detections: list[Detection], metadata: EmbedderMetadata
) -> np.ndarray:
    if metadata.crop_mode != "box_resize" or metadata.crop_margin_ratio != 0.0:
        raise ValueError("ROI integrity training requires the current box-resize contract")
    roi = detections[0]
    box = (int(np.floor(roi.x1)), int(np.floor(roi.y1)), int(np.ceil(roi.x2)), int(np.ceil(roi.y2)))
    tensor = prepare_rgb(
        image.crop(box),
        metadata.input_size,
        metadata.mean,
        metadata.std,
        reducing_gap=metadata.resize_reducing_gap,
    )[None]
    if metadata.neighbor_mask:
        mask = classifier_neighbor_ownership_mask(
            detections,
            0,
            image_width=image.width,
            image_height=image.height,
            output_size=metadata.input_size[0],
            margin_ratio=metadata.crop_margin_ratio,
            distance_bias=metadata.neighbor_distance_bias,
            shared_scale=metadata.neighbor_shared_scale,
            crop_mode=metadata.crop_mode,
        )[None]
        tensor = apply_classifier_background_masks(tensor, mask)
    return tensor[0]


def build_samples(work: Path, recipe: str, seed: int, settings: dict) -> list[dict]:
    originals = read_jsonl(work / "prepared/original_detection.jsonl")
    scenes = read_jsonl(work / f"prepared/{recipe}-{seed}/manifest.jsonl")
    allowed = {r["image_sha256"] for r in read_jsonl(work / "sources/originals.jsonl")}
    original_single, synthetic_single, original_pairs, synthetic_pairs = [], [], [], []
    for original, records in ((True, originals), (False, scenes)):
        for row in records:
            if row["split"] != "train" or not set(row["parent_sha256"]).issubset(allowed):
                raise ValueError("integrity head input violates source-only training")
            boxes = [xyxy(a) for a in row["annotations"]]
            common = {
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "parent_sha256": row["parent_sha256"],
                "source_image_id": row["image_id"],
                "source_group": row["source_group"],
                "split": "train",
                "boxes": boxes,
                "original": original,
            }
            singles = original_single if original else synthetic_single
            pairs = original_pairs if original else synthetic_pairs
            for index in range(len(boxes)):
                singles.append({**common, "group": [index], "label": 0})
            for left, right in itertools.combinations(range(len(boxes)), 2):
                a, b = boxes[left], boxes[right]
                overlap = float(box_iou_matrix(np.array([a]), np.array([b]))[0, 0])
                areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in (a, b)]
                box = union_box([a, b])
                width, height = box[2] - box[0], box[3] - box[1]
                if (
                    overlap < settings["minimum_pair_iou"]
                    or min(areas) <= 0
                    or max(areas) / min(areas) > settings["maximum_pair_area_ratio"]
                    or max(width / height, height / width) > settings["maximum_union_aspect_ratio"]
                ):
                    continue
                pairs.append({**common, "group": [left, right], "label": 1})
    rng = random.Random(settings["seed"])
    count = settings["samples_per_label"]

    def fill(required: list[dict], pool: list[dict]) -> list[dict]:
        if len(required) > count or not pool:
            raise ValueError("insufficient balanced integrity training inputs")
        remaining = count - len(required)
        selected = rng.sample(pool, min(remaining, len(pool)))
        selected += rng.choices(pool, k=remaining - len(selected))
        return required + selected

    samples = fill(original_single, synthetic_single) + fill(original_pairs, synthetic_pairs)
    rng.shuffle(samples)
    return [{**row, "sample_id": index} for index, row in enumerate(samples)]


def build_head(dimension: int, settings: dict):
    import torch

    return torch.nn.Sequential(
        torch.nn.LayerNorm(dimension),
        torch.nn.Linear(dimension, settings["hidden_dimension"]),
        torch.nn.GELU(),
        torch.nn.Dropout(settings["dropout"]),
        torch.nn.Linear(settings["hidden_dimension"], 2),
    )


def train(
    config_path: Path,
    method: str,
    recipe: str,
    seed: int,
    *,
    embedder_metadata: dict | None = None,
) -> dict:
    import torch

    from ..experiments.input_isolation import protect_development
    from .three_bakery_detector import seed_everything

    settings = load_json_config(config_path)
    source_config = Path(settings["source_config"])
    work = Path(settings["source_work"]).resolve()
    protect_development(source_config, work)
    if (work / "final/benchmark-access.json").exists():
        raise ValueError("ROI policy cannot be trained after final benchmark access")
    seed_everything(settings["seed"])
    torch.set_num_threads(4)
    model_dir = work / f"models/{method}-{recipe}-{seed}"
    candidate = work / f"candidates/ssdlite-{method}-{recipe}-{seed}"
    if not candidate.exists():
        candidate = work / f"candidates/dfine-{method}-{recipe}-{seed}"
    metadata = EmbedderMetadata.model_validate(
        embedder_metadata or load_json_config(candidate / "runtime/metadata.json")["embedder"]
    )
    contract = load_json_config(model_dir / "contract.json")
    report = load_json_config(model_dir / "report.json")
    if report["output_sha256"]["model.pt"] != sha256_file(model_dir / "model.pt"):
        raise ValueError("current-run classifier checkpoint checksum mismatch")
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = work / f"roi-integrity/{method}-{recipe}-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "settings_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(work / "sources/originals.jsonl"),
        "annotations_sha256": sha256_file(work / "sources/annotations.jsonl"),
        "model_sha256": sha256_file(model_dir / "model.pt"),
        "code_sha256": sha256_file(Path(__file__)),
        "preprocessing": metadata.model_dump(mode="json"),
        "synthetic_manifest_sha256": sha256_file(work / f"prepared/{recipe}-{seed}/manifest.jsonl"),
    }
    if (output / "inputs.json").exists() and load_json_config(output / "inputs.json") != inputs:
        raise ValueError("integrity head resume input mismatch")
    if (output / "report.json").exists():
        existing = load_json_config(output / "report.json")
        if (
            existing["inputs_sha256"] != sha256_file(output / "inputs.json")
            or existing["head_sha256"] != sha256_file(output / "head.pt")
            or existing["samples_sha256"] != sha256_file(output / "samples.jsonl")
            or existing["onnx_sha256"] != sha256_file(output / "primary.onnx")
        ):
            raise ValueError("integrity head resume output checksum mismatch")
        return existing
    write_json(output / "inputs.json", inputs)
    samples = build_samples(work, recipe, seed, settings)
    write_jsonl(output / "samples.jsonl", samples)
    started = time.time()
    features = []
    batch = []
    for index, row in enumerate(samples):
        path = Path(row["image_path"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("integrity source image changed")
        with Image.open(path) as opened:
            batch.append(
                prepare_roi(
                    opened.convert("RGB"), grouped_detections(row["boxes"], row["group"]), metadata
                )
            )
        if len(batch) == settings["feature_batch_size"] or index == len(samples) - 1:
            with torch.inference_mode():
                features.append(
                    model.extract_features(torch.from_numpy(np.stack(batch)).cuda()).cpu()
                )
            batch = []
        if (index + 1) % 1000 == 0:
            print(f"integrity features {index + 1}/{len(samples)}", flush=True)
    features = torch.cat(features).cuda()
    labels = torch.tensor([r["label"] for r in samples], device="cuda")
    head = build_head(features.shape[1], settings).cuda()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    history = []
    for epoch in range(settings["epochs"]):
        head.train()
        losses = []
        for indices in torch.randperm(len(features), device="cuda").split(
            settings["head_batch_size"]
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    head.eval()
    torch.save(head.cpu().state_dict(), output / "head.pt")
    np.save(output / "training-features.npy", features.cpu().numpy())

    class SharedFeatureHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.cpu()
            self.head = head

        def forward(self, pixel_values):
            features = self.model.extract_features(pixel_values)
            embeddings = torch.nn.functional.normalize(self.model.classifier(features), dim=-1)
            multiplicity = torch.softmax(self.head(features), dim=-1)[:, 1]
            return embeddings, multiplicity

    torch.onnx.export(
        SharedFeatureHead().eval(),
        torch.zeros(1, 3, *metadata.input_size),
        output / "primary.onnx",
        input_names=[metadata.input_name],
        output_names=[metadata.output_name, "multi_object_probabilities"],
        dynamic_axes={
            metadata.input_name: {0: "batch"},
            metadata.output_name: {0: "batch"},
            "multi_object_probabilities": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    result = {
        "inputs_sha256": sha256_file(output / "inputs.json"),
        "samples_sha256": sha256_file(output / "samples.jsonl"),
        "head_sha256": sha256_file(output / "head.pt"),
        "onnx_sha256": sha256_file(output / "primary.onnx"),
        "feature_dimension": features.shape[1],
        "sample_count": len(samples),
        "unique_original_single_objects": len(
            {
                (r["source_image_id"], tuple(r["group"]))
                for r in samples
                if r["original"] and r["label"] == 0
            }
        ),
        "epochs": settings["epochs"],
        "epoch_history": history,
        "elapsed_seconds": time.time() - started,
        "current_run_classifier_checkpoint": str(model_dir / "model.pt"),
        "foundation_backbone_updated": False,
        "scope": "same-source development; not independent validation; no deployment yet",
    }
    write_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bread/three_bakery_roi_integrity.json"),
    )
    parser.add_argument("--method", choices=["frozen", "finetune", "margin"], required=True)
    parser.add_argument("--recipe", choices=["basic", "dense"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train(args.config, args.method, args.recipe, args.seed)


if __name__ == "__main__":
    main()
