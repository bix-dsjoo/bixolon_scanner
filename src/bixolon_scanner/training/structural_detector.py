"""D-FINE-N GT training and GT-matched teacher distillation at unchanged resolution."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..runtime.geometry import box_iou_matrix
from .structural_student import cpu_session
from .three_bakery_data import write_json
from .three_bakery_detector import (
    DetectorDataset,
    collate_detector,
    export_dfine,
    seed_everything,
    training_rows,
)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    return np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)


def rotate_boxes(boxes: np.ndarray, turns: int) -> np.ndarray:
    result = cxcywh_to_xyxy(boxes)
    for _ in range(turns % 4):
        result = np.stack([result[:, 1], 1 - result[:, 2], result[:, 3], 1 - result[:, 0]], -1)
    return np.concatenate([(result[:, :2] + result[:, 2:]) / 2, result[:, 2:] - result[:, :2]], -1)


def match_teacher(
    gt: np.ndarray, predicted: np.ndarray, scores: np.ndarray, minimum_iou: float
) -> dict:
    from scipy.optimize import linear_sum_assignment

    boxes = gt.copy()
    confidence = np.zeros(len(gt), np.float32)
    if len(gt) and len(predicted):
        overlaps = box_iou_matrix(cxcywh_to_xyxy(gt), cxcywh_to_xyxy(predicted))
        left, right = linear_sum_assignment(-overlaps)
        accepted = overlaps[left, right] >= minimum_iou
        left, right = left[accepted], right[accepted]
        boxes[left] = predicted[right]
        confidence[left] = scores[right]
    return {"boxes": boxes.tolist(), "scores": confidence.tolist()}


def teacher_cache(config: dict, rows: list[dict], size: int) -> dict:
    root = Path(config["output"]) / "detector-cache"
    root.mkdir(parents=True, exist_ok=True)
    model = Path(config["runtime"]) / "detector.onnx"
    metadata = load_json_config(Path(config["runtime"]) / "metadata.json")["detector"]
    unique = {r["image_sha256"]: r for r in rows}
    contract = {
        "teacher_sha256": sha256_file(model),
        "images": sorted(unique),
        "size": size,
        "minimum_iou": config["detector"]["teacher_match_iou"],
    }
    destination = root / "teacher.json"
    if destination.exists():
        previous = load_json_config(destination)
        if previous["contract"] != contract:
            raise ValueError("detector teacher cache contract changed")
        return previous["rows"]
    session = cpu_session(model)
    result = {}
    for index, (digest, row) in enumerate(unique.items()):
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        values = np.asarray(image).astype(np.float32).transpose(2, 0, 1)[None] / 255
        # Detector's current package uses RGB [0,1]; reject unsupported changes.
        if metadata["mean"] != [0.0, 0.0, 0.0] or metadata["std"] != [1.0, 1.0, 1.0]:
            raise ValueError("teacher cache needs the declared RGB [0,1] detector")
        logits, predicted = session.run(
            [metadata["logits_output"], metadata["boxes_output"]], {metadata["input_name"]: values}
        )
        scores = 1 / (1 + np.exp(-np.clip(logits[0].reshape(-1), -80, 80)))
        boxes = np.array([a["bbox_xywh"] for a in row["annotations"]], dtype=np.float32).reshape(
            -1, 4
        )
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes /= np.array([row["width"], row["height"]] * 2)
        keep = scores >= metadata["score_threshold"]
        result[digest] = match_teacher(
            boxes, predicted[0][keep], scores[keep], config["detector"]["teacher_match_iou"]
        )
        if index % 100 == 0:
            print(f"detector teacher {index + 1}/{len(unique)}", flush=True)
    write_json(destination, {"contract": contract, "rows": result})
    return result


class DistilledDetectorDataset(DetectorDataset):
    def __init__(self, rows, size, cache, teacher):
        super().__init__(rows, size, cache, "dfine")
        self.teacher = teacher

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        image = Image.fromarray(self.images[self.index[row["image_sha256"]]])
        boxes = np.array([a["bbox_xywh"] for a in row["annotations"]], np.float32).reshape(-1, 4)
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes /= np.array([row["width"], row["height"]] * 2)
        reference = self.teacher[row["image_sha256"]]
        turns = random.randrange(4)
        image = image.rotate(turns * 90)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.2))
        target = {
            "boxes": torch.from_numpy(rotate_boxes(boxes, turns).copy()),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "teacher_boxes": torch.from_numpy(
                rotate_boxes(np.array(reference["boxes"], np.float32).reshape(-1, 4), turns).copy()
            ),
            "teacher_scores": torch.tensor(reference["scores"], dtype=torch.float32),
        }
        values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return (
            values,
            target,
            row["image_sha256"] if row.get("kind") else None,
            row["parent_sha256"],
        )


def build_detector(config: dict):
    import torch

    settings = config["detector"]
    repository = Path(settings["repository"]).resolve()
    sys.path.insert(0, str(repository))
    from src.core import YAMLConfig

    cfg = YAMLConfig(
        str(repository / f"configs/dfine/{settings['architecture']}.yml"),
        num_classes=1,
        remap_mscoco_category=False,
    )
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(settings["weights"], map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    current = model.state_dict()
    transfer = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
    missing = model.load_state_dict(transfer, strict=False)
    return (
        model.cuda(),
        cfg.criterion.cuda(),
        {"loaded": len(transfer), "missing": missing.missing_keys},
    )


def train(config_path: Path, objective: str):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    config = load_json_config(config_path)
    settings = config["detector"]
    seed_everything(config["seed"])
    _, rows, _ = training_rows(
        Path(config["source_work"]), config["source_recipe"], config["source_seed"]
    )
    size = load_json_config(Path(config["runtime"]) / "metadata.json")["detector"]["input_size"][0]
    teacher = teacher_cache(config, rows, size)
    output = Path(config["output"]) / "detectors" / objective
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(Path(settings["weights"])),
        "teacher_cache_sha256": sha256_file(Path(config["output"]) / "detector-cache/teacher.json"),
        "objective": objective,
        "selection": config["checkpoint_selection"],
        "rows": len(rows),
    }
    if (output / "report.json").exists():
        report = load_json_config(output / "report.json")
        if (
            report["contract"] != contract
            or sha256_file(output / "detector.onnx") != report["onnx_sha256"]
        ):
            raise ValueError("detector output changed")
        return report
    if (output / "contract.json").exists() and load_json_config(
        output / "contract.json"
    ) != contract:
        raise ValueError("detector resume contract changed")
    write_json(output / "contract.json", contract)
    dataset = DistilledDetectorDataset(
        rows, size, Path(config["output"]) / "detector-cache/images", teacher
    )
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_detector,
        persistent_workers=True,
    )
    model, criterion, transfer = build_detector(config)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("backbone.")],
                "lr": settings["backbone_learning_rate"],
            },
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")],
                "lr": settings["learning_rate"],
            },
        ],
        weight_decay=0.0001,
    )
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
        total, count = 0.0, 0
        for step, (images, targets, _, _) in enumerate(loader):
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            gt = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.stack(images).cuda(), targets=gt)
            losses = criterion(
                prediction,
                gt,
                epoch=epoch,
                step=step,
                global_step=epoch * len(loader) + step,
                epoch_step=len(loader),
            )
            loss = sum(losses.values())
            if objective == "distilled":
                indices = criterion.matcher(prediction, gt)["indices"]
                terms = []
                for batch, (source, target) in enumerate(indices):
                    source = source.to(prediction["pred_boxes"].device)
                    target = target.to(prediction["pred_boxes"].device)
                    valid = targets[batch]["teacher_scores"][target] > 0
                    source, target = source[valid], target[valid]
                    if len(source):
                        terms.append(
                            F.l1_loss(
                                prediction["pred_boxes"][batch, source],
                                targets[batch]["teacher_boxes"][target],
                            )
                        )
                        terms.append(
                            F.binary_cross_entropy_with_logits(
                                prediction["pred_logits"][batch, source, 0],
                                targets[batch]["teacher_scores"][target],
                            )
                        )
                if terms:
                    loss = loss + torch.stack(terms).mean() * settings["teacher_weight"]
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite detector loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / count,
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
            f"D-FINE-N {objective} epoch {epoch + 1}/{settings['epochs']} loss={total / count:.5f}",
            flush=True,
        )
    torch.save(model.state_dict(), output / "model.pt")
    parameter_count = sum(p.numel() for p in model.parameters())
    export_dfine(model.eval(), output / "detector.onnx", size)
    result = {
        "contract": contract,
        "transfer": transfer,
        "parameters": parameter_count,
        "history": history,
        "onnx_sha256": sha256_file(output / "detector.onnx"),
        "deployment_selected": False,
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objective", choices=["supervised", "distilled"], required=True)
    args = parser.parse_args()
    train(args.config, args.objective)


if __name__ == "__main__":
    main()
