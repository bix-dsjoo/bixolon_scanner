"""Source-only cross-architecture experiments; never select on regression images."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import EmbedderMetadata
from .roi_integrity import grouped_detections, prepare_roi
from .three_bakery_data import read_jsonl, write_json, write_jsonl
from .three_bakery_detector import seed_everything


def source_samples(config: dict) -> list[dict]:
    work = Path(config["source_work"])
    originals = read_jsonl(work / "prepared/original_detection.jsonl")
    synthetic = read_jsonl(
        work / f"prepared/{config['source_recipe']}-{config['source_seed']}/manifest.jsonl"
    )
    allowed = {r["image_sha256"] for r in read_jsonl(work / "sources/originals.jsonl")}
    sources = {r["image_sha256"]: r for r in originals + synthetic}
    result = []
    verified = set()
    for row in read_jsonl(Path(config["integrity_samples"])):
        source = sources[row["image_sha256"]]
        if source["split"] != "train" or not set(source["parent_sha256"]) <= allowed:
            raise ValueError("non-training or foreign parent in student inputs")
        if row["image_sha256"] not in verified:
            if sha256_file(Path(row["image_path"])) != row["image_sha256"]:
                raise ValueError("student source checksum mismatch")
            verified.add(row["image_sha256"])
        group = row["group"]
        category = source["annotations"][group[0]]["category_id"] - 1 if len(group) == 1 else -1
        turns = range(4) if row["original"] and len(group) == 1 else [0]
        for turn in turns:
            result.append({**row, "category": category, "quarter_turns": turn})
    return result


def cpu_session(path: Path, threads: int = 4):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def prepare_cache(config_path: Path) -> dict:
    config = load_json_config(config_path)
    output = Path(config["output"]) / "student-cache"
    output.mkdir(parents=True, exist_ok=True)
    runtime = Path(config["runtime"])
    metadata = EmbedderMetadata.model_validate(
        load_json_config(runtime / "metadata.json")["embedder"]
    )
    contract = {
        "settings_sha256": sha256_file(config_path),
        "samples_source_sha256": sha256_file(Path(config["integrity_samples"])),
        "teacher_sha256": sha256_file(runtime / metadata.filename),
        "code_sha256": sha256_file(Path(__file__)),
        "preprocessing": metadata.model_dump(mode="json"),
    }
    if (output / "complete.json").exists():
        existing = load_json_config(output / "complete.json")
        if existing["contract"] != contract:
            raise ValueError("student cache contract changed")
        for name, digest in existing["sha256"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("student cache checksum mismatch")
        return existing
    rows = source_samples(config)
    write_jsonl(output / "samples.jsonl", rows)
    shape = (len(rows), 3, *metadata.input_size)
    tensors = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.float16, shape=shape
    )
    teacher = np.zeros((len(rows), metadata.embedding_dimension), dtype=np.float32)
    integrity = np.zeros(len(rows), dtype=np.float32)
    session = cpu_session(runtime / metadata.filename)
    previous_path, image = None, None
    start = time.perf_counter()
    for start_index in range(0, len(rows), 32):
        batch = []
        for row in rows[start_index : start_index + 32]:
            if row["image_path"] != previous_path:
                if image is not None:
                    image.close()
                with Image.open(row["image_path"]) as opened:
                    image = opened.convert("RGB")
                previous_path = row["image_path"]
            value = prepare_roi(image, grouped_detections(row["boxes"], row["group"]), metadata)
            batch.append(np.rot90(value, row["quarter_turns"], axes=(1, 2)).copy())
        values = np.stack(batch).astype(np.float16)
        # Teacher and student see exactly the same cached input, including rounding.
        embeddings, probabilities = session.run(
            None, {metadata.input_name: values.astype(np.float32)}
        )
        end_index = start_index + len(values)
        tensors[start_index:end_index] = values
        teacher[start_index:end_index] = embeddings
        integrity[start_index:end_index] = probabilities.reshape(-1)
        if start_index % 512 == 0:
            print(f"teacher cache {end_index}/{len(rows)}", flush=True)
    if image is not None:
        image.close()
    tensors.flush()
    del tensors
    np.savez(
        output / "targets.npz",
        embeddings=teacher,
        integrity=integrity,
        categories=np.array([r["category"] for r in rows]),
        multiplicity=np.array([r["label"] for r in rows]),
    )
    result = {
        "contract": contract,
        "sample_count": len(rows),
        "elapsed_seconds": time.perf_counter() - start,
        "sha256": {
            name: sha256_file(output / name)
            for name in ("images.npy", "targets.npz", "samples.jsonl")
        },
        "independent_validation_available": False,
    }
    write_json(output / "complete.json", result)
    return result


def build_student(name: str, settings: dict, *, pretrained: bool = True):
    import timm
    import torch
    from torch.nn import functional as F

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
            dimension = self.backbone.num_features
            self.classifier = torch.nn.Linear(dimension, settings["class_count"], bias=False)
            self.integrity = torch.nn.Sequential(
                torch.nn.LayerNorm(dimension),
                torch.nn.Linear(dimension, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 2),
            )

        def evidence(self, values):
            features = self.backbone(values)
            scores = F.linear(
                F.normalize(features, dim=-1), F.normalize(self.classifier.weight, dim=-1)
            )
            return F.normalize(scores, dim=-1), self.integrity(features)

        def forward(self, pixel_values):
            embeddings, logits = self.evidence(pixel_values)
            return embeddings, logits.softmax(-1)[:, 1]

    return Student()


def evidence_loss(
    embeddings,
    integrity_logits,
    categories,
    multiplicity,
    teacher,
    teacher_integrity,
    settings: dict,
    objective: str,
):
    import torch
    from torch.nn import functional as F

    if objective not in {"supervised", "distilled", "relational"}:
        raise ValueError("unknown student objective")
    valid = categories >= 0
    loss = F.cross_entropy(integrity_logits, multiplicity) * settings["integrity_weight"]
    if valid.any():
        scores, targets = embeddings[valid], categories[valid]
        chosen = scores.gather(1, targets[:, None]).squeeze(1)
        other = scores.masked_fill(F.one_hot(targets, scores.shape[1]).bool(), -2).max(1).values
        loss = loss + F.cross_entropy(scores * settings["cosine_scale"], targets)
        loss = (
            loss
            + (settings["margin_target"] - chosen + other).clamp_min(0).mean()
            * settings["margin_weight"]
        )
    if objective != "supervised":
        scale = settings["distillation_scale"]
        kd = F.kl_div(
            (embeddings * scale).log_softmax(-1),
            (teacher * scale).softmax(-1),
            reduction="batchmean",
        )
        kd = kd + (1 - (embeddings * teacher).sum(-1)).mean()
        soft_quality = torch.stack([1 - teacher_integrity, teacher_integrity], -1)
        kd = kd + F.kl_div(integrity_logits.log_softmax(-1), soft_quality, reduction="batchmean")
        loss = loss + kd * settings["distillation_weight"]
    if objective == "relational" and len(embeddings) > 1:
        loss = (
            loss
            + F.mse_loss(embeddings @ embeddings.T, teacher @ teacher.T)
            * settings["relation_weight"]
        )
    return loss


def export_student(model, output: Path, metadata: EmbedderMetadata, probe=None) -> dict:
    import torch
    from timm.utils import reparameterize_model

    original = copy.deepcopy(model).cpu().eval()
    deployed = copy.deepcopy(original)
    deployed.backbone = reparameterize_model(deployed.backbone)
    generator = torch.Generator().manual_seed(910)
    values = torch.randn(4, 3, *metadata.input_size, generator=generator)
    if probe is not None:
        values = torch.cat([values, torch.as_tensor(probe, dtype=torch.float32)])
    with torch.inference_mode():
        expected = original(values)
        actual = deployed(values)
    fusion_errors = [
        float((left - right).abs().max()) for left, right in zip(expected, actual, strict=True)
    ]
    fused = True
    try:
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-4)
    except AssertionError:
        # Preserve the trained graph when algebraic branch fusion is numerically unstable.
        # This is an explicit export choice, not a runtime provider fallback.
        deployed, actual, fused = original, expected, False
    torch.onnx.export(
        deployed,
        values[:1],
        output,
        input_names=[metadata.input_name],
        output_names=[metadata.output_name, metadata.multi_object_output_name],
        dynamic_axes={
            metadata.input_name: {0: "batch"},
            metadata.output_name: {0: "batch"},
            metadata.multi_object_output_name: {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
        do_constant_folding=fused,
    )
    observed = cpu_session(output).run(None, {metadata.input_name: values.numpy()})
    errors = []
    for left, right in zip(actual, observed, strict=True):
        np.testing.assert_allclose(left.numpy(), right, atol=1e-3, rtol=1e-3)
        errors.append(float(np.max(np.abs(left.numpy() - right))))
    np.testing.assert_array_equal(actual[0].argmax(-1).numpy(), observed[0].argmax(-1))
    return {
        "reparameterized": fused,
        "fusion_max_absolute_error": fusion_errors,
        "probe_count": len(values),
        "onnx_tolerance": {"atol": 1e-3, "rtol": 1e-3, "top1_equal": True},
        "onnx_max_absolute_error": errors,
        "onnx_sha256": sha256_file(output),
    }


def train_student(config_path: Path, name: str, objective: str) -> dict:
    import torch

    config = load_json_config(config_path)
    settings = config["classifier"]
    seed_everything(config["seed"])
    root = Path(config["output"])
    cache = root / "student-cache"
    completed = load_json_config(cache / "complete.json")
    output = root / "students" / f"{name.split('.')[0]}-{objective}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "config_sha256": sha256_file(config_path),
        "cache_sha256": sha256_file(cache / "complete.json"),
        "name": name,
        "objective": objective,
        "seed": config["seed"],
        "checkpoint_selection": config["checkpoint_selection"],
    }
    if (output / "report.json").exists():
        report = load_json_config(output / "report.json")
        if (
            report["contract"] != contract
            or sha256_file(output / "model.onnx") != report["export"]["onnx_sha256"]
        ):
            raise ValueError("student completed output differs from contract")
        return report
    if (output / "contract.json").exists() and load_json_config(
        output / "contract.json"
    ) != contract:
        raise ValueError("student resume contract changed")
    write_json(output / "contract.json", contract)
    model = build_student(name, settings).cuda()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": settings["backbone_lr"]},
            {
                "params": list(model.classifier.parameters()) + list(model.integrity.parameters()),
                "lr": settings["head_lr"],
            },
        ],
        weight_decay=settings["weight_decay"],
    )
    # Cache fits in host RAM; transfer only each sampled batch to the GPU.
    images = np.load(cache / "images.npy", mmap_mode="r")
    targets = np.load(cache / "targets.npz")
    categories = torch.from_numpy(targets["categories"]).cuda()
    multiplicity = torch.from_numpy(targets["multiplicity"]).cuda()
    teacher = torch.from_numpy(targets["embeddings"]).cuda()
    teacher_integrity = torch.from_numpy(targets["integrity"]).cuda()
    history, start_epoch = [], 0
    if (output / "resume.pt").exists():
        state = torch.load(output / "resume.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start_epoch = state["history"], state["epoch"]
    start = time.perf_counter()
    for epoch in range(start_epoch, settings["epochs"]):
        seed_everything(config["seed"] + epoch)
        model.train()
        losses = []
        for indices in torch.randperm(len(images)).split(settings["batch_size"]):
            index = indices.cuda()
            values = torch.from_numpy(np.array(images[indices.numpy()], dtype=np.float32)).cuda()
            optimizer.zero_grad(set_to_none=True)
            embeddings, integrity_logits = model.evidence(values)
            loss = evidence_loss(
                embeddings,
                integrity_logits,
                categories[index],
                multiplicity[index],
                teacher[index],
                teacher_integrity[index],
                settings,
                objective,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite student loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        write_json(output / "history.json", history)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
            },
            output / "resume.pt",
        )
        print(
            f"{name} {objective} epoch {epoch + 1}/{settings['epochs']}: loss={history[-1]['loss']:.5f}",
            flush=True,
        )
    model.eval()
    torch.save(model.cpu().state_dict(), output / "model.pt")
    metadata = EmbedderMetadata.model_validate(
        load_json_config(Path(config["runtime"]) / "metadata.json")["embedder"]
    )
    exported = export_student(
        model, output / "model.onnx", metadata, np.array(images[:16], dtype=np.float32)
    )
    report = {
        "contract": contract,
        "export": exported,
        "samples": completed["sample_count"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
        "independent_validation_available": False,
        "deployment_selected": False,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["cache", "train"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--objective", choices=["supervised", "distilled", "relational"])
    args = parser.parse_args()
    if args.action == "cache":
        prepare_cache(args.config)
    else:
        if not args.model or not args.objective:
            parser.error("train needs --model and --objective")
        train_student(args.config, args.model, args.objective)


if __name__ == "__main__":
    main()
