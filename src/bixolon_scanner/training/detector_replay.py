"""Fixed source-only detector replay with hash-locked consumption and resumability."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .input_isolation import protect_development
from .three_bakery_data import read_jsonl, write_json, write_jsonl
from .three_bakery_detector import (
    DetectorDataset,
    collate_detector,
    seed_everything,
    verify_coverage,
)
from .three_bakery_preparation import validate_parents


def train(config_path: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from .ssdlite_objectness_detector import (
        build_ssdlite_objectness,
        enable_empty_image_hard_negative_loss,
        export_ssdlite_onnx,
    )

    config = load_json_config(config_path)
    output = Path(config["output"])
    protect_development(Path(config["study_config"]), output)
    original_path = Path(config["original_manifest"])
    synthetic_path = Path(config["synthetic_manifest"])
    originals = read_jsonl(original_path)
    required = {r["image_sha256"] for r in originals}
    synthetic = read_jsonl(synthetic_path)
    rng = random.Random(config["seed"])
    selected = rng.sample(
        [r for r in synthetic if r["annotations"]], config["synthetic_positive_count"]
    )
    selected += rng.sample(
        [r for r in synthetic if not r["annotations"]], config["synthetic_empty_count"]
    )
    rows = originals * config["original_repeat"] + selected
    validate_parents(originals + selected, required)
    for row in originals + selected:
        if row["split"] != "train" or sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError("replay input split or checksum mismatch")
    base = Path(config["base_model"])
    base_report = load_json_config(base / "report.json")
    base_contract = load_json_config(base / "contract.json")
    if (
        base_report["contract_sha256"] != sha256_file(base / "contract.json")
        or base_report["output_sha256"]["model.pt"] != sha256_file(base / "model.pt")
        or base_contract["original_manifest_sha256"] != sha256_file(original_path)
        or set(base_report["observed_input_counts"]) != required
    ):
        raise ValueError("replay initialization provenance mismatch")
    contract = {
        "settings": config,
        "config_sha256": sha256_file(config_path),
        "code_sha256": sha256_file(Path(__file__)),
        "dataset_code_sha256": sha256_file(Path(__file__).with_name("three_bakery_detector.py")),
        "original_manifest_sha256": sha256_file(original_path),
        "synthetic_manifest_sha256": sha256_file(synthetic_path),
        "initial_checkpoint_sha256": sha256_file(base / "model.pt"),
        "base_report_sha256": sha256_file(base / "report.json"),
        "selected_synthetic_hashes": [r["image_sha256"] for r in selected],
        "row_count": len(rows),
        "command": sys.argv,
    }
    contract_path = output / "contract.json"
    if contract_path.exists():
        old = load_json_config(contract_path)
        if {k: v for k, v in old.items() if k != "command"} != {
            k: v for k, v in contract.items() if k != "command"
        }:
            raise ValueError("replay resume inputs changed")
        if (output / "report.json").exists():
            report = load_json_config(output / "report.json")
            for name, digest in report["output_sha256"].items():
                if sha256_file(output / name) != digest:
                    raise ValueError("replay output changed")
            return report
    else:
        write_json(contract_path, contract)
        write_jsonl(output / "consumption-manifest.jsonl", originals + selected)
    seed_everything(config["seed"])
    dataset = DetectorDataset(rows, config["image_size"], output / "cache", "ssdlite")
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True,
        collate_fn=collate_detector,
    )
    model = build_ssdlite_objectness(pretrained_detector_transfer=False, device="cuda")
    model.load_state_dict(torch.load(base / "model.pt", map_location="cpu", weights_only=True))
    enable_empty_image_hard_negative_loss(model, minimum_negatives=32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    start, history, observed = 0, [], Counter()
    if (output / "resume.pt").exists():
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history, observed = state["epoch"], state["history"], Counter(state["observed"])
    started = time.time()
    for epoch in range(start, config["epochs"]):
        seed_everything(config["seed"] + epoch)
        model.train()
        losses_total, count, consumed = 0.0, 0, Counter()
        for values, targets, parents, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            loss = sum(model([v.cuda(non_blocking=True) for v in values], targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite replay loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses_total += float(loss.detach())
            count += 1
            consumed.update(v for v in parents if v is not None)
        verify_coverage(consumed, required)
        observed.update(consumed)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": losses_total / count,
                "original_input_counts": dict(consumed),
                "steps": count,
                "elapsed_seconds": time.time() - started,
            }
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
        print(
            f"replay epoch {epoch + 1}/{config['epochs']}: loss={losses_total / count:.6f}",
            flush=True,
        )
    verify_coverage(observed, required)
    torch.save(model.state_dict(), output / "model.pt")
    model.eval()
    export_ssdlite_onnx(model, output / "detector.onnx", detector_class_count=1)
    report = {
        "contract_sha256": sha256_file(contract_path),
        "epochs": config["epochs"],
        "observed_original_count": len(observed),
        "observed_input_counts": dict(observed),
        "elapsed_seconds": time.time() - started,
        "output_sha256": {
            n: sha256_file(output / n) for n in ("model.pt", "detector.onnx", "history.json")
        },
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
