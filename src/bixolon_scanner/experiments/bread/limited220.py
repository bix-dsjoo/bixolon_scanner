"""Reproducible fixed-budget training stages and explicit development diagnostics."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ...configuration import load_json_config
from ...contracts.catalog import sha256_file
from ...training.limited_augmentation import generate_training_derivatives
from ...training.limited_source import prepare_sources, read_jsonl, verify_sources, write_json


def verify_derivatives(work: Path, source_report: dict) -> dict:
    derived = work / "derived"
    report = load_json_config(derived / "derivative-report.json")
    if report["source_manifest_sha256"] != source_report["source_manifest_sha256"]:
        raise ValueError("derivatives belong to a different source budget")
    allowed = {r["image_sha256"] for r in read_jsonl(work / "sources/originals.jsonl")}
    for name, key in (
        ("synthetic.jsonl", "synthetic_manifest_sha256"),
        ("multi-crops.jsonl", "crop_manifest_sha256"),
    ):
        if sha256_file(derived / name) != report[key]:
            raise ValueError("derived manifest changed")
        for row in read_jsonl(derived / name):
            if not set(row["parent_sha256s"]).issubset(allowed):
                raise ValueError("derivative contains an original outside the budget")
            if not row["parent_sha256s"] and row.get("annotations"):
                raise ValueError("positive training derivative has no source provenance")
            path = (derived / row["image_path"]).resolve()
            path.relative_to(derived.resolve())
            if sha256_file(path) != row["image_sha256"]:
                raise ValueError("derived image changed")
    return report


def make_plan(config_path: Path, work: Path, convnext: Path, vit: Path) -> dict:
    if (work / "plan.json").exists():
        raise FileExistsError(
            "keep the original plan; use another work directory for a new candidate"
        )
    config = load_json_config(config_path)
    source = verify_sources(work / "sources")
    derived = verify_derivatives(work, source)
    if source["config_sha256"] != sha256_file(config_path):
        raise ValueError("configuration changed since source selection")
    if config["training"]["checkpoint_selection"] != "fixed_last_epoch_without_evaluation":
        raise ValueError("same-item originals cannot select checkpoints using a validation split")
    convnext, vit, work = convnext.resolve(), vit.resolve(), work.resolve()
    if sha256_file(convnext) != "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54":
        raise ValueError("expected the official DINOv3 ConvNeXt-Tiny foundation weights")
    if sha256_file(vit) != "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c":
        raise ValueError("expected the official frozen DINOv3 ViT-B/16 foundation weights")
    repository = Path(__file__).resolve().parents[4]
    t = config["training"]
    root = Path(source["dataset_root"])
    stages = []

    def stage(name, argv, inputs, outputs, dependencies=()):
        stages.append(
            {
                "name": name,
                "argv": [str(v) for v in argv],
                "inputs": [str(v) for v in inputs],
                "outputs": [str(v) for v in outputs],
                "dependencies": list(dependencies),
            }
        )

    py = sys.executable
    for kind, manifest, dataset_root in (
        ("real", work / "sources/detector.jsonl", root),
        ("synthetic", work / "derived/synthetic.jsonl", work / "derived"),
    ):
        stage(
            f"cache-{kind}",
            [
                py,
                "-m",
                "bixolon_scanner.training.cache_detector",
                "--manifest",
                manifest,
                "--dataset-root",
                dataset_root,
                "--output-dir",
                work / f"cache/{kind}",
                "--image-size",
                "320",
            ],
            [manifest],
            [work / f"cache/{kind}/index.json", work / f"cache/{kind}/images.npy"],
        )
    detector_output = work / "detector"
    stage(
        "train-detector",
        [
            py,
            "-m",
            "bixolon_scanner.experiments.bread.ssdlite_objectness_detector",
            "--real-manifest",
            work / "sources/detector.jsonl",
            "--real-root",
            root,
            "--real-cache",
            work / "cache/real",
            "--synthetic-manifest",
            work / "derived/synthetic.jsonl",
            "--synthetic-root",
            work / "derived",
            "--synthetic-cache",
            work / "cache/synthetic",
            "--output-dir",
            detector_output,
            "--fixed-epochs-no-selection",
            "--pretrained-detector-transfer",
            "--quarter-turn-augmentation",
            "--minimum-empty-negatives",
            "32",
            "--epochs",
            t["detector_epochs"],
            "--batch-size",
            t["detector_batch_size"],
            "--real-repeat",
            t["detector_real_repeat"],
            "--learning-rate",
            t["detector_learning_rate"],
            "--workers",
            t["workers"],
            "--seed",
            config["seed"],
        ],
        [work / "cache/real/images.npy", work / "cache/synthetic/images.npy"],
        [detector_output / "report.json", detector_output / "detector.onnx"],
        ["cache-real", "cache-synthetic"],
    )
    classifier_output = work / "classifier"
    stage(
        "train-classifier",
        [
            py,
            repository / "scripts/train_store2_dinov3_classifier.py",
            "--manifest",
            work / "sources/classifier.jsonl",
            "--root",
            root,
            "--expected-support-count",
            source["single_count"],
            "--additional-manifest",
            work / "derived/multi-crops.jsonl",
            "--additional-root",
            work / "derived",
            "--additional-samples-per-class",
            "40",
            "--weights",
            convnext,
            "--weights-sha256",
            sha256_file(convnext),
            "--output-dir",
            classifier_output,
            "--image-size",
            "192",
            "--detail-image-size",
            "224",
            "--consistency-weight",
            t["resolution_consistency_weight"],
            "--epochs",
            t["classifier_epochs"],
            "--views",
            t["classifier_views"],
            "--batch-size",
            t["classifier_batch_size"],
            "--workers",
            t["workers"],
            "--backbone-learning-rate",
            t["classifier_backbone_learning_rate"],
            "--head-learning-rate",
            t["classifier_head_learning_rate"],
            "--seed",
            config["seed"],
        ],
        [work / "sources/classifier.jsonl", work / "derived/multi-crops.jsonl", convnext],
        [classifier_output / "report.json", classifier_output / "last.pt"],
    )
    for name, size in (("primary", 192), ("detail", 224)):
        stage(
            f"export-{name}",
            [
                py,
                repository / "scripts/export_store2_dinov3_classifier.py",
                "--checkpoint",
                classifier_output / "last.pt",
                "--weights",
                convnext,
                "--image-size",
                size,
                "--output-kind",
                "cosine-logits",
                "--output",
                work / f"exports/{name}.onnx",
            ],
            [classifier_output / "last.pt", convnext],
            [work / f"exports/{name}.onnx"],
            ["train-classifier"],
        )
    stage(
        "export-verifier",
        [
            py,
            repository / "scripts/export_frozen_dinov3_embedder.py",
            "--weights",
            vit,
            "--variant",
            "dinov3_vitb16",
            "--image-size",
            "160",
            "--output",
            work / "exports/verifier.onnx",
        ],
        [vit],
        [work / "exports/verifier.onnx"],
    )
    parent_artifacts = {}
    classifier_stage = next(s for s in stages if s["name"] == "train-classifier")
    if t.get("scene_samples_per_epoch", 0):
        classifier_stage["argv"].extend(
            [
                "--scene-manifest",
                str(work / "sources/detector.jsonl"),
                "--scene-root",
                str(root),
                "--scene-samples-per-epoch",
                str(t["scene_samples_per_epoch"]),
                "--scene-neighbor-mask",
                "--scene-neighbor-distance-bias",
                str(t["scene_neighbor_distance_bias"]),
                "--scene-minimum-margin",
                str(t["scene_minimum_margin"]),
                "--scene-maximum-margin",
                str(t["scene_maximum_margin"]),
            ]
        )
        classifier_stage["inputs"].append(str(work / "sources/detector.jsonl"))
    if t.get("normalized_margin_weight", 0):
        classifier_stage["argv"].extend(
            [
                "--normalized-margin-weight",
                str(t["normalized_margin_weight"]),
                "--normalized-margin-target",
                str(t["normalized_margin_target"]),
            ]
        )
    if t.get("initial_classifier_checkpoint"):
        checkpoint = Path(t["initial_classifier_checkpoint"]).resolve()
        parent_work = checkpoint.parent.parent
        parent_source = verify_sources(parent_work / "sources")
        evidence_path = parent_work / "stages/train-classifier.json"
        evidence = load_json_config(evidence_path)
        if (
            parent_source["source_manifest_sha256"] != source["source_manifest_sha256"]
            or evidence["source_manifest_sha256"] != source["source_manifest_sha256"]
            or evidence["returncode"] != 0
            or evidence["output_sha256s"].get(str(checkpoint)) != sha256_file(checkpoint)
        ):
            raise ValueError(
                "initial classifier must be a verified checkpoint from these same originals"
            )
        classifier_stage["argv"].extend(["--initial-checkpoint", str(checkpoint)])
        classifier_stage["inputs"].append(str(checkpoint))
        parent_artifacts = {
            str(checkpoint): sha256_file(checkpoint),
            str(evidence_path): sha256_file(evidence_path),
        }
    plan = {
        "schema_version": "1.0",
        "repository_root": str(repository),
        "work_dir": str(work),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "derivative_manifest_sha256": derived["synthetic_manifest_sha256"],
        "original_budget": source["original_count"],
        "pretrained_weights": {str(convnext): sha256_file(convnext), str(vit): sha256_file(vit)},
        "evaluation_role": source["evaluation_role"],
        "automatic_model_selection": False,
        "parent_artifacts": parent_artifacts,
        "stages": stages,
    }
    write_json(work / "plan.json", plan)
    (work / "plan.sha256").write_text(sha256_file(work / "plan.json") + "\n", encoding="ascii")
    return plan


def run_stage(work: Path, name: str) -> dict:
    plan_path = work / "plan.json"
    if sha256_file(plan_path) != (work / "plan.sha256").read_text().strip():
        raise ValueError("execution plan changed; generate a new reviewed experiment plan")
    plan = load_json_config(plan_path)
    if Path(plan["work_dir"]).resolve() != work.resolve():
        raise ValueError("plan work directory does not match")
    if sha256_file(Path(plan["config"])) != plan["config_sha256"]:
        raise ValueError("experiment configuration changed")
    source = verify_sources(work / "sources")
    if source["source_manifest_sha256"] != plan["source_manifest_sha256"]:
        raise ValueError("plan source budget changed")
    verify_derivatives(work, source)
    for path, digest in (plan["pretrained_weights"] | plan.get("parent_artifacts", {})).items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("foundation weights changed")
    stage = next((s for s in plan["stages"] if s["name"] == name), None)
    if stage is None:
        raise ValueError("unknown experiment stage")
    evidence_path = work / f"stages/{name}.json"
    if evidence_path.exists():
        raise FileExistsError("stage already attempted; use a new work directory for another run")
    for dependency in stage["dependencies"]:
        evidence = load_json_config(work / f"stages/{dependency}.json")
        if evidence["returncode"] != 0:
            raise ValueError("stage dependency failed")
        for path, digest in evidence["output_sha256s"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("stage dependency output changed")
    inputs = {path: sha256_file(Path(path)) for path in stage["inputs"]}
    for path in stage["outputs"]:
        Path(path).resolve().relative_to(work.resolve())
        if Path(path).exists():
            raise FileExistsError("stage output already exists")
    log_path = work / f"stages/{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(plan["repository_root"]) / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            stage["argv"],
            cwd=plan["repository_root"],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    outputs = {path: sha256_file(Path(path)) for path in stage["outputs"] if Path(path).is_file()}
    complete = result.returncode == 0 and len(outputs) == len(stage["outputs"])
    evidence = {
        "stage": name,
        "returncode": result.returncode if complete else result.returncode or 1,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "input_sha256s": inputs,
        "output_sha256s": outputs,
        "duration_seconds": time.perf_counter() - started,
        "argv": stage["argv"],
        "log": str(log_path),
    }
    write_json(evidence_path, evidence)
    if not complete:
        raise RuntimeError(f"experiment stage failed; see {log_path}")
    return evidence


def run_all(work: Path) -> dict:
    if sha256_file(work / "plan.json") != (work / "plan.sha256").read_text().strip():
        raise ValueError("execution plan changed")
    plan = load_json_config(work / "plan.json")
    if Path(plan["work_dir"]).resolve() != work.resolve():
        raise ValueError("plan work directory does not match")
    if sha256_file(Path(plan["config"])) != plan["config_sha256"]:
        raise ValueError("experiment configuration changed")
    source = verify_sources(work / "sources")
    if source["source_manifest_sha256"] != plan["source_manifest_sha256"]:
        raise ValueError("plan source budget changed")
    verify_derivatives(work, source)
    for path, digest in (plan["pretrained_weights"] | plan.get("parent_artifacts", {})).items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("foundation weights changed")
    completed = []
    for stage in plan["stages"]:
        name = stage["name"]
        evidence_path = work / f"stages/{name}.json"
        if evidence_path.exists():
            evidence = load_json_config(evidence_path)
            if evidence["returncode"] != 0:
                raise RuntimeError(
                    f"previous stage failed: {name}; inspect its log before a new run"
                )
            if evidence["source_manifest_sha256"] != source["source_manifest_sha256"] or set(
                evidence["output_sha256s"]
            ) != set(stage["outputs"]):
                raise ValueError("completed stage does not match this experiment")
            for path, digest in evidence["output_sha256s"].items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError("completed stage output changed")
        else:
            run_stage(work, name)
        completed.append(name)
        print(f"completed: {name}", flush=True)
    # Revalidate sources even when all stages were reused.
    source = verify_sources(work / "sources")
    verify_derivatives(work, source)
    if sha256_file(work / "plan.json") != (work / "plan.sha256").read_text().strip():
        raise ValueError("execution plan changed")
    return {
        "completed_stages": completed,
        "source_manifest_sha256": source["source_manifest_sha256"],
    }


def reuse_stage(work: Path, previous_work: Path, name: str) -> dict:
    plans = []
    for directory in (work, previous_work):
        if sha256_file(directory / "plan.json") != (directory / "plan.sha256").read_text().strip():
            raise ValueError("execution plan changed")
        source = verify_sources(directory / "sources")
        verify_derivatives(directory, source)
        current_plan = load_json_config(directory / "plan.json")
        if (
            Path(current_plan["work_dir"]).resolve() != directory.resolve()
            or sha256_file(Path(current_plan["config"])) != current_plan["config_sha256"]
            or current_plan["source_manifest_sha256"] != source["source_manifest_sha256"]
        ):
            raise ValueError("reuse plan configuration or source changed")
        for path, digest in (
            current_plan["pretrained_weights"] | current_plan.get("parent_artifacts", {})
        ).items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("reuse foundation or parent weights changed")
        plans.append(current_plan)
    plan, previous_plan = plans
    if plan["source_manifest_sha256"] != previous_plan["source_manifest_sha256"]:
        raise ValueError("cannot reuse a model from different originals")
    if plan["pretrained_weights"] != previous_plan["pretrained_weights"]:
        raise ValueError("cannot reuse a stage with different foundation weights")
    stage = next(s for s in plan["stages"] if s["name"] == name)
    old = next(s for s in previous_plan["stages"] if s["name"] == name)
    argv = [a.replace(str(work.resolve()), "{WORK}") for a in stage["argv"]]
    old_argv = [a.replace(str(previous_work.resolve()), "{WORK}") for a in old["argv"]]
    if argv != old_argv:
        raise ValueError("stage recipe differs; train/export it again")
    for old_input, new_input in zip(old["inputs"], stage["inputs"], strict=True):
        if sha256_file(Path(old_input)) != sha256_file(Path(new_input)):
            raise ValueError("stage input differs; train/export it again")
    previous_evidence_path = previous_work / f"stages/{name}.json"
    evidence = load_json_config(previous_evidence_path)
    if (
        evidence["returncode"] != 0
        or set(evidence["output_sha256s"]) != set(old["outputs"])
        or evidence["source_manifest_sha256"] != plan["source_manifest_sha256"]
        or evidence["argv"] != old["argv"]
    ):
        raise ValueError("previous stage did not complete")
    destination_evidence = work / f"stages/{name}.json"
    if destination_evidence.exists() or any(Path(p).exists() for p in stage["outputs"]):
        raise FileExistsError("reused stage destination already exists")
    for path, digest in evidence["output_sha256s"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("previous stage output changed")
    outputs = {}
    for old_name, new_name in zip(old["outputs"], stage["outputs"], strict=True):
        destination = Path(new_name)
        destination.resolve().relative_to(work.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_name, destination)
        outputs[new_name] = sha256_file(destination)
    result = {
        "stage": name,
        "returncode": 0,
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "argv": stage["argv"],
        "input_sha256s": {path: sha256_file(Path(path)) for path in stage["inputs"]},
        "output_sha256s": outputs,
        "reuse_evidence": str(previous_evidence_path),
        "reuse_evidence_sha256": sha256_file(previous_evidence_path),
        "duration_seconds": 0.0,
    }
    write_json(destination_evidence, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="220-original training and development diagnostics"
    )
    parser.add_argument("action", choices=("prepare", "generate", "plan", "run", "reuse"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bread/limited220.json")
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path)
    parser.add_argument("--vit-weights", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--from-work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if args.action == "prepare":
        value = prepare_sources(args.config, work / "sources")
    elif args.action == "generate":
        value = generate_training_derivatives(
            work / "sources", work / "derived", load_json_config(args.config)["synthetic"]
        )
    elif args.action == "plan":
        if args.convnext_weights is None or args.vit_weights is None:
            parser.error("plan requires both official foundation weight paths")
        value = make_plan(args.config, work, args.convnext_weights, args.vit_weights)
    elif args.action == "reuse":
        if args.stage is None or args.from_work_dir is None:
            parser.error("reuse requires --stage and --from-work-dir")
        value = reuse_stage(work, args.from_work_dir.resolve(), args.stage)
    else:
        if args.stage is None:
            parser.error("run requires --stage")
        value = run_all(work) if args.stage == "all" else run_stage(work, args.stage)
    print(value)


if __name__ == "__main__":
    main()
