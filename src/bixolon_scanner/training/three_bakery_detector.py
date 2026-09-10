"""Fixed-budget class-agnostic detector training with observed source coverage."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .three_bakery_data import read_jsonl, verify_prepared, verify_sources, write_json
from .three_bakery_preparation import validate_parents


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def training_rows(work: Path, recipe: str, seed: int) -> tuple[dict, list[dict], set[str]]:
    verify_prepared(work, recipe, seed)
    report, sources = verify_sources(work / "sources")
    config = load_json_config(Path(report["config_path"]))
    originals = read_jsonl(work / "prepared/original_detection.jsonl")
    synthetic = read_jsonl(work / f"prepared/{recipe}-{seed}/manifest.jsonl")
    allowed = {r["image_sha256"] for r in sources}
    if {r["image_sha256"] for r in originals} != allowed:
        raise ValueError("detector training must include every original")
    for row in originals + synthetic:
        if row["split"] != "train":
            raise ValueError("diagnostic or final evaluation input cannot enter training")
        if sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError("training image checksum mismatch")
    validate_parents(originals + synthetic, allowed)
    return config, originals * config["training"]["real_repeat"] + synthetic, allowed


def verify_coverage(observed: Counter, required: set[str]) -> None:
    if set(observed) != required or any(observed[v] <= 0 for v in required):
        raise ValueError("actual training input coverage differs from the source contract")


class DetectorDataset:
    def __init__(self, rows: list[dict], size: int, cache: Path, architecture: str):
        self.rows, self.size, self.architecture = rows, size, architecture
        unique = {row["image_sha256"]: row for row in rows}
        cache.mkdir(parents=True, exist_ok=True)
        metadata_path = cache / "index.json"
        expected = {"image_sha256": list(unique), "size": size}
        if metadata_path.exists():
            if load_json_config(metadata_path) != expected:
                raise ValueError("detector image cache input mismatch")
        else:
            images = np.lib.format.open_memmap(
                cache / "images.npy", mode="w+", dtype=np.uint8, shape=(len(unique), size, size, 3)
            )
            for index, row in enumerate(unique.values()):
                with Image.open(row["image_path"]) as opened:
                    images[index] = np.asarray(
                        ImageOps.exif_transpose(opened)
                        .convert("RGB")
                        .resize((size, size), Image.Resampling.BILINEAR)
                    )
            images.flush()
            del images
            write_json(metadata_path, expected)
        self.index = {digest: i for i, digest in enumerate(unique)}
        self.path = cache / "images.npy"
        self.images = np.load(self.path, mmap_mode="r")

    def __getstate__(self):
        return {**self.__dict__, "images": None}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.images = np.load(self.path, mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        image = Image.fromarray(self.images[self.index[row["image_sha256"]]])
        boxes = np.asarray([a["bbox_xywh"] for a in row["annotations"]], dtype=np.float32).reshape(
            -1, 4
        )
        boxes[:, 2:] += boxes[:, :2]
        boxes /= np.asarray([row["width"], row["height"]] * 2, dtype=np.float32)
        turns = random.randrange(4)
        image = image.rotate(turns * 90)
        for _ in range(turns):
            boxes = np.stack([boxes[:, 1], 1 - boxes[:, 2], boxes[:, 3], 1 - boxes[:, 0]], axis=1)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.2))
        if self.architecture == "dfine":
            boxes = np.concatenate(
                [(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], axis=1
            )
            labels = torch.zeros(len(boxes), dtype=torch.long)
        else:
            boxes *= self.size
            labels = torch.ones(len(boxes), dtype=torch.long)
        target = {"boxes": torch.from_numpy(boxes.copy()), "labels": labels}
        values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        # Observe originals themselves, not only their descendants.
        original = row["image_sha256"] if row.get("kind") else None
        return values, target, original, row["parent_sha256"]


def collate_detector(batch):
    return tuple(zip(*batch, strict=True))


def dfine_model(repository: Path, weights: Path):
    import torch

    sys.path.insert(0, str(repository.resolve()))
    from src.core import YAMLConfig

    config = YAMLConfig(
        str(repository / "configs/dfine/dfine_hgnetv2_s_coco.yml"),
        num_classes=1,
        remap_mscoco_category=False,
    )
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    target = model.state_dict()
    transferable = {k: v for k, v in state.items() if k in target and v.shape == target[k].shape}
    missing = model.load_state_dict(transferable, strict=False)
    return (
        model.cuda(),
        config.criterion.cuda(),
        {"loaded": len(transferable), "missing": missing.missing_keys},
    )


def export_dfine(model, output: Path, size: int) -> None:
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()

        def forward(self, pixel_values):
            values = self.model(pixel_values)
            return values["pred_logits"], values["pred_boxes"]

    torch.onnx.export(
        Wrapper().eval(),
        torch.zeros(1, 3, size, size, device="cuda"),
        output,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamo=False,
    )


def train(
    work: Path,
    architecture: str,
    recipe: str,
    seed: int,
    weights: Path,
    repository: Path | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from .ssdlite_objectness_detector import (
        build_ssdlite_objectness,
        enable_empty_image_hard_negative_loss,
        export_ssdlite_onnx,
    )

    seed_everything(seed)
    config, rows, required = training_rows(work, recipe, seed)
    settings = config["training"][architecture]
    output = work / f"models/{architecture}-{recipe}-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "architecture": architecture,
        "recipe": recipe,
        "seed": seed,
        "settings": settings,
        "source_manifest_sha256": sha256_file(work / "sources/originals.jsonl"),
        "annotation_sha256": sha256_file(work / "sources/annotations.jsonl"),
        "config_sha256": sha256_file(Path(verify_sources(work / "sources")[0]["config_path"])),
        "code_sha256": sha256_file(Path(__file__)),
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "row_count": len(rows),
        "original_manifest_sha256": sha256_file(work / "prepared/original_detection.jsonl"),
        "synthetic_manifest_sha256": sha256_file(work / f"prepared/{recipe}-{seed}/manifest.jsonl"),
        "command": sys.argv,
        "checkpoint_selection": "fixed_last_epoch_without_evaluation",
    }
    contract_path = output / "contract.json"
    if contract_path.exists():
        old = load_json_config(contract_path)
        if {k: v for k, v in old.items() if k != "command"} != {
            k: v for k, v in contract.items() if k != "command"
        }:
            raise ValueError("detector training resume contract mismatch")
        if (output / "report.json").exists():
            result = load_json_config(output / "report.json")
            for name, digest in result["output_sha256"].items():
                if sha256_file(output / name) != digest:
                    raise ValueError("trained detector output changed")
            return result
    else:
        write_json(contract_path, contract)
    dataset = DetectorDataset(
        rows, settings["image_size"], work / f"cache/{architecture}-{recipe}-{seed}", architecture
    )
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=config["training"]["workers"],
        pin_memory=True,
        collate_fn=collate_detector,
        persistent_workers=False,
    )
    if architecture == "ssdlite":
        # torchvision loads this named COCO foundation file from its standard cache.
        expected = (
            Path(torch.hub.get_dir())
            / "checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
        )
        if sha256_file(expected) != sha256_file(weights):
            raise ValueError("SSDLite foundation weight cache differs from contract")
        model = build_ssdlite_objectness(pretrained_detector_transfer=True, device="cuda")
        enable_empty_image_hard_negative_loss(model, minimum_negatives=32)
        criterion = None
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"], weight_decay=0.0001
        )
        transfer = {"kind": "COCO_shape_compatible_transfer"}
    else:
        model, criterion, transfer = dfine_model(repository, weights)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in model.named_parameters() if n.startswith("backbone.")],
                    "lr": settings["backbone_learning_rate"],
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if not n.startswith("backbone.")
                    ],
                    "lr": settings["learning_rate"],
                },
            ],
            weight_decay=0.0001,
        )
    start_epoch = 0
    history = []
    observed = Counter()
    if (output / "resume.pt").exists():
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, history, observed = (
            state["epoch"],
            state["history"],
            Counter(state["observed"]),
        )
    started = time.time()
    for epoch in range(start_epoch, settings["epochs"]):
        seed_everything(seed + epoch)
        model.train()
        if criterion is not None:
            criterion.train()
        total, steps = 0.0, 0
        epoch_sources = Counter()
        for values, targets, originals, _parents in loader:
            targets = [
                {k: v.cuda(non_blocking=True) for k, v in target.items()} for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            if criterion is None:
                losses = model([v.cuda(non_blocking=True) for v in values], targets)
            else:
                predictions = model(torch.stack(values).cuda(non_blocking=True), targets=targets)
                losses = criterion(
                    predictions,
                    targets,
                    epoch=epoch,
                    step=steps,
                    global_step=epoch * len(loader) + steps,
                    epoch_step=len(loader),
                )
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite detector training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.1 if architecture == "dfine" else 10.0
            )
            optimizer.step()
            epoch_sources.update(v for v in originals if v is not None)
            total += float(loss.detach())
            steps += 1
        verify_coverage(epoch_sources, required)
        observed.update(epoch_sources)
        row = {
            "epoch": epoch + 1,
            "mean_loss": total / steps,
            "steps": steps,
            "original_input_counts": dict(epoch_sources),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"{architecture}-{recipe}-{seed} epoch {epoch + 1}: loss={total / steps:.5f}",
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
    verify_coverage(observed, required)
    torch.save(model.state_dict(), output / "model.pt")
    model.eval()
    if architecture == "ssdlite":
        export_ssdlite_onnx(model, output / "detector.onnx", detector_class_count=1)
    else:
        export_dfine(model, output / "detector.onnx", settings["image_size"])
    result = {
        "contract_sha256": sha256_file(contract_path),
        "epochs": settings["epochs"],
        "observed_original_count": len(observed),
        "observed_input_counts": dict(observed),
        "transfer": transfer,
        "elapsed_seconds": time.time() - started,
        "output_sha256": {
            name: sha256_file(output / name)
            for name in ("model.pt", "detector.onnx", "history.json")
        },
    }
    write_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--architecture", choices=["ssdlite", "dfine"], required=True)
    parser.add_argument("--recipe", choices=["basic", "dense"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args()
    train(args.work, args.architecture, args.recipe, args.seed, args.weights, args.repository)


if __name__ == "__main__":
    main()
