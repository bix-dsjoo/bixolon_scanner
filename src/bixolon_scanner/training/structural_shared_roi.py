"""Frozen detector spatial feature probe with GT locations and crop-teacher supervision."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .roi_integrity import union_box
from .structural_student import cpu_session, evidence_loss
from .three_bakery_data import read_jsonl, write_json
from .three_bakery_detector import seed_everything


def expose_detector_features(source: Path, output: Path) -> dict:
    import onnx

    original = onnx.load(source)
    model = copy.deepcopy(original)
    # This is the high-resolution FPN tensor already consumed by the decoder.
    # Resolve it through the consumer node rather than adding a new encoder.
    consumer = next(node for node in model.graph.node if node.name == "/model/decoder/Shape")
    feature = consumer.input[0]
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(feature, onnx.TensorProto.FLOAT, [1, 256, 80, 80])
    )
    onnx.checker.check_model(model)
    if [n.SerializeToString() for n in original.graph.node] != [
        n.SerializeToString() for n in model.graph.node
    ]:
        raise ValueError("exposing a feature must not change detector operators")
    if [v.SerializeToString() for v in original.graph.initializer] != [
        v.SerializeToString() for v in model.graph.initializer
    ]:
        raise ValueError("exposing a feature must not change detector weights")
    onnx.save(model, output)
    values = np.random.default_rng(910).random((1, 3, 640, 640), dtype=np.float32)
    baseline = cpu_session(source).run(None, {"pixel_values": values})
    observed = cpu_session(output).run(None, {"pixel_values": values})
    differences = []
    for left, right in zip(baseline, observed[:2], strict=True):
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
        differences.append(float(np.max(np.abs(left - right))))
    if observed[2].shape != (1, 256, 80, 80):
        raise ValueError("observed spatial feature shape differs from the experiment")
    return {
        "source_sha256": sha256_file(source),
        "onnx_sha256": sha256_file(output),
        "feature_name": feature,
        "shape": list(observed[2].shape),
        "detector_output_errors": differences,
        "weights_identical": True,
        "operators_identical": True,
    }


def pool_rois(
    features: np.ndarray, boxes: list, image_size: tuple[int, int], pool_size: int
) -> np.ndarray:
    import torch
    from torchvision.ops import roi_align

    width, height = image_size
    regions = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    regions *= np.array([features.shape[3] / width, features.shape[2] / height] * 2, np.float32)
    rois = torch.from_numpy(np.column_stack([np.zeros(len(regions), np.float32), regions]))
    with torch.inference_mode():
        result = roi_align(
            torch.from_numpy(features),
            rois,
            output_size=pool_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
    return result.numpy()


def build_head(channels: int, pool_size: int, hidden: int, classes: int):
    import torch
    from torch.nn import functional as F

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            dimension = channels * pool_size * pool_size
            self.trunk = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.LayerNorm(dimension),
                torch.nn.Linear(dimension, hidden),
                torch.nn.GELU(),
            )
            self.identity = torch.nn.Linear(hidden, classes, bias=False)
            self.integrity = torch.nn.Linear(hidden, 2)

        def evidence(self, values):
            features = self.trunk(values)
            scores = F.linear(
                F.normalize(features, dim=-1), F.normalize(self.identity.weight, dim=-1)
            )
            return F.normalize(scores, dim=-1), self.integrity(features)

        def forward(self, roi_features):
            embeddings, logits = self.evidence(roi_features)
            return embeddings, logits.softmax(-1)[:, 1]

    return Head()


def prepare(config: dict) -> dict:
    root = Path(config["output"])
    output = root / "shared-roi"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "cache.json").exists():
        result = load_json_config(output / "cache.json")
        for name, digest in result["sha256"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("shared ROI cache changed")
        return result
    exposed = expose_detector_features(
        Path(config["runtime"]) / "detector.onnx", output / "detector-features.onnx"
    )
    write_json(output / "detector-feature-export.json", exposed)
    samples = read_jsonl(root / "student-cache/samples.jsonl")
    selected = [i for i, row in enumerate(samples) if row["quarter_turns"] == 0]
    by_image = {}
    for index, source_index in enumerate(selected):
        by_image.setdefault(samples[source_index]["image_path"], []).append((index, source_index))
    size = config["shared_roi"]["pool_size"]
    pooled = np.zeros((len(selected), 256, size, size), np.float32)
    session = cpu_session(output / "detector-features.onnx")
    for index, (path, rows) in enumerate(by_image.items()):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        values = (
            np.asarray(image.resize((640, 640), Image.Resampling.BILINEAR))
            .astype(np.float32)
            .transpose(2, 0, 1)[None]
            / 255
        )
        feature = session.run([exposed["feature_name"]], {"pixel_values": values})[0]
        boxes = [union_box([samples[j]["boxes"][k] for k in samples[j]["group"]]) for _, j in rows]
        pooled[[i for i, _ in rows]] = pool_rois(feature, boxes, image.size, size)
        if index % 100 == 0:
            print(f"shared feature cache {index + 1}/{len(by_image)}", flush=True)
    teacher = np.load(root / "student-cache/targets.npz")
    np.save(output / "features.npy", pooled)
    np.savez(output / "targets.npz", **{key: teacher[key][selected] for key in teacher.files})
    np.save(output / "source-indices.npy", selected)
    result = {
        "sample_count": len(selected),
        "source_image_count": len(by_image),
        "feature_export": exposed,
        "scope": "source GT locations; frozen detector; no rotation-equivariance assumption; no new detector training",
        "sha256": {
            name: sha256_file(output / name)
            for name in ["features.npy", "targets.npz", "source-indices.npy"]
        },
    }
    write_json(output / "cache.json", result)
    return result


def train(config_path: Path, objective: str):
    import torch

    config = load_json_config(config_path)
    prepare(config)
    seed_everything(config["seed"])
    root = Path(config["output"]) / "shared-roi"
    output = root / objective
    output.mkdir(parents=True, exist_ok=True)
    features = torch.from_numpy(np.load(root / "features.npy")).cuda()
    data = np.load(root / "targets.npz")
    targets = {key: torch.from_numpy(data[key]).cuda() for key in data.files}
    settings = config["shared_roi"]
    model = build_head(
        256,
        settings["pool_size"],
        settings["hidden_dimension"],
        config["classifier"]["class_count"],
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=0.0001
    )
    history = []
    start = time.perf_counter()
    for epoch in range(settings["epochs"]):
        model.train()
        losses = []
        for indices in torch.randperm(len(features), device="cuda").split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            embeddings, integrity = model.evidence(features[indices])
            loss = evidence_loss(
                embeddings,
                integrity,
                targets["categories"][indices],
                targets["multiplicity"][indices],
                targets["embeddings"][indices],
                targets["integrity"][indices],
                config["classifier"],
                objective,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite shared ROI loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        if (epoch + 1) % 20 == 0:
            print(
                f"shared ROI {objective} epoch {epoch + 1}: loss={history[-1]['loss']:.5f}",
                flush=True,
            )
    model = model.cpu().eval()
    torch.save(model.state_dict(), output / "model.pt")
    torch.onnx.export(
        model,
        features[:1].cpu(),
        output / "head.onnx",
        input_names=["roi_features"],
        output_names=["embeddings", "multi_object_probabilities"],
        dynamic_axes={
            "roi_features": {0: "rois"},
            "embeddings": {0: "rois"},
            "multi_object_probabilities": {0: "rois"},
        },
        opset_version=18,
        dynamo=False,
    )
    write_json(
        output / "report.json",
        {
            "history": history,
            "elapsed_seconds": time.perf_counter() - start,
            "config_sha256": sha256_file(config_path),
            "head_sha256": sha256_file(output / "head.onnx"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "deployment_selected": False,
        },
    )


def evaluate_gt(config: dict, objective: str, manifest: Path):
    """An optimistic fixed-GT-box identity probe, not end-to-end scan accuracy."""
    root = Path(config["output"]) / "shared-roi"
    metadata = load_json_config(root / "detector-feature-export.json")
    detector = cpu_session(root / "detector-features.onnx")
    head = cpu_session(root / objective / "head.onnx")
    rows = []
    for row in read_jsonl(manifest):
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        boxes, labels = [], []
        for annotation in row["annotations"]:
            x, y, w, h = annotation.get("bbox_xywh", annotation.get("bbox"))
            boxes.append([x, y, x + w, y + h])
            labels.append(annotation["category_id"] - 1)
        if not boxes:
            continue
        values = (
            np.asarray(image.resize((640, 640), Image.Resampling.BILINEAR))
            .astype(np.float32)
            .transpose(2, 0, 1)[None]
            / 255
        )
        feature = detector.run([metadata["feature_name"]], {"pixel_values": values})[0]
        pooled = pool_rois(feature, boxes, image.size, config["shared_roi"]["pool_size"])
        embeddings, integrity = head.run(None, {"roi_features": pooled})
        top3 = np.argsort(-embeddings, axis=1)[:, :3]
        rows.append(
            {
                "image_id": row["image_id"],
                "labels": labels,
                "top3": top3.tolist(),
                "integrity": integrity.tolist(),
                "boxes": boxes,
                "correct_top1": int(np.sum(top3[:, 0] == labels)),
                "correct_top3": int(np.sum(np.any(top3 == np.array(labels)[:, None], axis=1))),
            }
        )
    result = {
        "scope": "GT-box optimistic identity probe; excludes detection error, Catalog and final policy; not full-path approval metrics",
        "gt_count": sum(len(r["labels"]) for r in rows),
        "correct_top1": sum(r["correct_top1"] for r in rows),
        "correct_top3": sum(r["correct_top3"] for r in rows),
        "rows": rows,
        "manifest_sha256": sha256_file(manifest),
    }
    write_json(root / objective / "gt-log-probe.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json_config(args.config)
    for objective in ["supervised", "relational"]:
        train(args.config, objective)
        evaluate_gt(
            config,
            objective,
            Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl"),
        )


if __name__ == "__main__":
    main()
