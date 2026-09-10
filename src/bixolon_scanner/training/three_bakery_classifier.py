"""Source-only cosine classifier comparison and frozen foundation verifier export."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .models import build_dino_classifier, set_frozen_backbone
from .three_bakery_data import read_jsonl, verify_prepared, verify_sources, write_json
from .three_bakery_detector import seed_everything, verify_coverage
from .three_bakery_preparation import validate_parents

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ClassifierDataset:
    def __init__(self, work: Path, recipe: str, seed: int, config: dict):
        verify_prepared(work, recipe, seed)
        originals = read_jsonl(work / "prepared/original_crops.jsonl")
        scenes = read_jsonl(work / f"prepared/{recipe}-{seed}/manifest.jsonl")
        _, sources = verify_sources(work / "sources")
        allowed = {r["image_sha256"] for r in sources}
        validate_parents(originals + scenes, allowed)
        for row in originals + scenes:
            if (
                row["split"] != "train"
                or sha256_file(Path(row["image_path"])) != row["image_sha256"]
            ):
                raise ValueError("classifier input provenance or split mismatch")
        self.required = {r["image_sha256"] for r in sources if r["kind"] != "background"}
        self.required_crops = {r["image_id"] for r in originals}
        if len(self.required_crops) != len(originals):
            raise ValueError("classifier crop IDs must be unique")
        if {r["parent_sha256"][0] for r in originals} != self.required:
            raise ValueError("classifier crops omit original foreground images")
        self.samples = []
        base_rows = []
        for row in originals:
            base_rows.append(
                {
                    "path": row["image_path"],
                    "box": None,
                    "category": row["category_id"],
                    "original": row["parent_sha256"][0],
                    "crop_id": row["image_id"],
                }
            )
        for view in range(config["training"]["classifier"]["views"]):
            self.samples.extend((index, view) for index in range(len(originals)))
        for scene in scenes:
            for annotation in scene["annotations"]:
                x, y, width, height = annotation["bbox_xywh"]
                base_rows.append(
                    {
                        "path": scene["image_path"],
                        "box": [x, y, x + width, y + height],
                        "category": annotation["category_id"],
                        "original": None,
                        "crop_id": "",
                    }
                )
                self.samples.append((len(base_rows) - 1, 1))
        self.rows = base_rows
        self.size = config["runtime_policy"]["detail_size"]
        self.primary_size = config["runtime_policy"]["primary_size"]
        cache = work / f"cache/classifier-{recipe}-{seed}"
        cache.mkdir(parents=True, exist_ok=True)
        self.path = cache / "images.npy"
        key = {
            "original_manifest_sha256": sha256_file(work / "prepared/original_crops.jsonl"),
            "synthetic_manifest_sha256": sha256_file(
                work / f"prepared/{recipe}-{seed}/manifest.jsonl"
            ),
            "size": self.size,
            "rows": len(base_rows),
        }
        if (cache / "index.json").exists():
            if load_json_config(cache / "index.json") != key:
                raise ValueError("classifier cache input mismatch")
        else:
            images = np.lib.format.open_memmap(
                self.path,
                mode="w+",
                dtype=np.uint8,
                shape=(len(base_rows), self.size, self.size, 3),
            )
            previous_path, image = None, None
            for index, row in enumerate(base_rows):
                if row["path"] != previous_path:
                    with Image.open(row["path"]) as opened:
                        image = opened.convert("RGB")
                    previous_path = row["path"]
                cropped = image if row["box"] is None else image.crop(row["box"])
                images[index] = np.asarray(
                    cropped.resize((self.size, self.size), Image.Resampling.BICUBIC)
                )
            images.flush()
            del images
            write_json(cache / "index.json", key)
        self.images = np.load(self.path, mmap_mode="r")

    def __getstate__(self):
        return {**self.__dict__, "images": None}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.images = np.load(self.path, mmap_mode="r")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        source_index, view = self.samples[index]
        row = self.rows[source_index]
        image = Image.fromarray(self.images[source_index])
        if view:
            image = image.rotate(random.randrange(4) * 90)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.25))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
        tensors = []
        for size in (self.primary_size, self.size):
            resized = image.resize((size, size), Image.Resampling.BICUBIC)
            values = (
                np.asarray(resized).astype(np.float32) / 255 - np.asarray(MEAN, dtype=np.float32)
            ) / np.asarray(STD, dtype=np.float32)
            tensors.append(torch.from_numpy(values.transpose(2, 0, 1).copy()))
        return *tensors, row["category"] - 1, row["original"] or "", row["crop_id"]


def export_classifier(model, output: Path, size: int, *, features: bool = False) -> None:
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            values = self.model.extract_features(pixel_values)
            if features:
                return values
            return torch.nn.functional.normalize(self.model.classifier(values), dim=-1)

    output.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    torch.onnx.export(
        Wrapper().eval(),
        torch.zeros(1, 3, size, size, device=device),
        output,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def train(work: Path, method: str, recipe: str, seed: int, weights: Path) -> dict:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from .margin_objective import normalized_margin_loss

    seed_everything(seed)
    report, _ = verify_sources(work / "sources")
    config = load_json_config(Path(report["config_path"]))
    settings = config["training"]["classifier"]
    output = work / f"models/{method}-{recipe}-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "method": method,
        "recipe": recipe,
        "seed": seed,
        "settings": settings,
        "config_sha256": report["config_sha256"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "annotation_sha256": sha256_file(work / "sources/annotations.jsonl"),
        "original_manifest_sha256": sha256_file(work / "prepared/original_crops.jsonl"),
        "synthetic_manifest_sha256": sha256_file(work / f"prepared/{recipe}-{seed}/manifest.jsonl"),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "code_sha256": sha256_file(Path(__file__)),
        "command": sys.argv,
        "preprocessing": {
            "mean": MEAN,
            "std": STD,
            "cosine_scale": 16.0,
            "primary_size": config["runtime_policy"]["primary_size"],
            "detail_size": config["runtime_policy"]["detail_size"],
        },
        "checkpoint_selection": "fixed_last_epoch_without_evaluation",
    }
    contract["preprocessing"]["mean"] = list(MEAN)
    contract["preprocessing"]["std"] = list(STD)
    if (output / "contract.json").exists():
        previous = load_json_config(output / "contract.json")
        if {k: v for k, v in previous.items() if k != "command"} != {
            k: v for k, v in contract.items() if k != "command"
        }:
            raise ValueError("classifier resume input mismatch")
        if (output / "report.json").exists():
            result = load_json_config(output / "report.json")
            for name, digest in result["output_sha256"].items():
                if sha256_file(output / name) != digest:
                    raise ValueError("classifier output checksum mismatch")
            return result
    else:
        write_json(output / "contract.json", contract)
    dataset = ClassifierDataset(work, recipe, seed, config)
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=config["training"]["workers"],
        pin_memory=True,
    )
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        config["class_count"],
        weights_path=weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    ).cuda()
    set_frozen_backbone(model, unfreeze_last_stages=0 if method == "frozen" else 1)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": settings["backbone_learning_rate"],
            },
            {"params": model.classifier.parameters(), "lr": settings["head_learning_rate"]},
        ],
        weight_decay=0.0001,
    )
    observed, history, start_epoch = Counter(), [], 0
    if (output / "resume.pt").exists():
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        observed, history, start_epoch = (
            Counter(state["observed"]),
            state["history"],
            state["epoch"],
        )
    started = time.time()
    for epoch in range(start_epoch, settings["epochs"]):
        seed_everything(seed + epoch)
        model.train()
        if method == "frozen":
            model.backbone.eval()
        else:
            for stage in model.backbone.stages[:3]:
                stage.eval()
        total, steps, correct, samples = 0.0, 0, 0, 0
        epoch_sources = Counter()
        epoch_crops = Counter()
        for primary, detail, labels, original, crop_ids in loader:
            primary, labels = primary.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            primary_features = model.extract_features(primary)
            logits = model.classifier(primary_features)
            loss = F.cross_entropy(logits, labels)
            if method == "margin":
                loss = loss + settings["margin_weight"] * normalized_margin_loss(
                    logits, labels, target_margin=settings["margin_target"]
                )
            if method != "frozen":
                detail_features = model.extract_features(detail.cuda(non_blocking=True))
                detail_logits = model.classifier(detail_features)
                detail_loss = F.cross_entropy(detail_logits, labels)
                if method == "margin":
                    detail_loss = detail_loss + settings["margin_weight"] * normalized_margin_loss(
                        detail_logits, labels, target_margin=settings["margin_target"]
                    )
                consistency = (1 - F.cosine_similarity(primary_features, detail_features)).mean()
                loss = (loss + detail_loss) / 2 + settings["consistency_weight"] * consistency
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite classifier loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_sources.update(v for v in original if v)
            epoch_crops.update(v for v in crop_ids if v)
            total += float(loss.detach())
            steps += 1
            correct += int((logits.argmax(-1) == labels).sum())
            samples += len(labels)
        verify_coverage(epoch_sources, dataset.required)
        verify_coverage(epoch_crops, dataset.required_crops)
        observed.update(epoch_sources)
        row = {
            "epoch": epoch + 1,
            "mean_loss": total / steps,
            "training_accuracy": correct / samples,
            "samples": samples,
            "steps": steps,
            "original_input_counts": dict(epoch_sources),
            "original_crop_input_counts": dict(epoch_crops),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"{method}-{recipe}-{seed} epoch {epoch + 1}: loss={total / steps:.5f}, train_acc={correct / samples:.4f}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "observed": dict(observed),
            },
            output / "resume.pt",
        )
        write_json(output / "history.json", history)
    verify_coverage(observed, dataset.required)
    torch.save(model.state_dict(), output / "model.pt")
    model.eval()
    export_classifier(model, output / "primary.onnx", config["runtime_policy"]["primary_size"])
    export_classifier(model, output / "detail.onnx", config["runtime_policy"]["detail_size"])
    result = {
        "contract_sha256": sha256_file(output / "contract.json"),
        "epochs": settings["epochs"],
        "observed_original_count": len(observed),
        "observed_original_crop_count": len(dataset.required_crops),
        "original_crop_input_counts": dict(
            sum((Counter(row["original_crop_input_counts"]) for row in history), Counter())
        ),
        "observed_input_counts": dict(observed),
        "elapsed_seconds": time.time() - started,
        "output_sha256": {
            name: sha256_file(output / name)
            for name in ("model.pt", "primary.onnx", "detail.onnx", "history.json")
        },
    }
    write_json(output / "report.json", result)
    return result


def export_verifier(work: Path, weights: Path) -> None:
    import torch

    torch.set_num_threads(4)
    report, _ = verify_sources(work / "sources")
    config = load_json_config(Path(report["config_path"]))
    output = work / "models/verifier"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "architecture": "dinov3_vitb16",
        "frozen": True,
        "weights_sha256": sha256_file(weights),
        "size": config["runtime_policy"]["verifier_size"],
        "code_sha256": sha256_file(Path(__file__)),
    }
    if (output / "report.json").exists():
        previous = load_json_config(output / "report.json")
        if any(previous[k] != v for k, v in contract.items()) or previous[
            "onnx_sha256"
        ] != sha256_file(output / "verifier.onnx"):
            raise ValueError("verifier export input/output changed")
        return
    model = build_dino_classifier(
        "dinov3_vitb16", config["class_count"], weights_path=weights, feature_l2_normalize=True
    ).eval()
    export_classifier(model, output / "verifier.onnx", contract["size"], features=True)
    write_json(
        output / "report.json", {**contract, "onnx_sha256": sha256_file(output / "verifier.onnx")}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--method", choices=["frozen", "finetune", "margin", "verifier"], required=True
    )
    parser.add_argument("--recipe", choices=["basic", "dense"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "verifier":
        export_verifier(args.work, args.weights)
    else:
        train(args.work, args.method, args.recipe, args.seed, args.weights)


if __name__ == "__main__":
    main()
