"""전량 학습, 12개 구성 비교 및 고정 벤치마크 Worker 평가 진입점."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ...configuration import load_json_config
from ...contracts.catalog import load_store_catalog_package, sha256_file
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...evaluation.three_bakery import candidate_rank, robust_rank
from ...evaluation.three_bakery_cpu import diagnose_cpu
from ...evaluation.three_bakery_http import (
    final_records,
    interpreter_identity,
    measure,
    provider_parity,
)
from ...evaluation.three_bakery_objective import (
    apply_final_objective,
    load_objective,
    rank_candidate,
)
from ...evaluation.three_bakery_reporting import write_report
from ...operations.three_bakery_runtime import assemble
from ...training.three_bakery_data import (
    draft_annotations,
    freeze_sources,
    read_jsonl,
    verify_sources,
    write_json,
)
from ...training.three_bakery_preparation import (
    accept_review,
    generate_scenes,
    prepare_originals,
    verify_annotations,
)
from ...training.three_bakery_revision import accept_revision, draft_revision
from ..input_isolation import protect_development

ARCHITECTURES = ("ssdlite", "dfine")
METHODS = ("frozen", "finetune", "margin")
RECIPES = ("basic", "dense")


def candidates() -> list[dict]:
    return [
        {
            "id": f"{architecture}-{method}-{recipe}",
            "architecture": architecture,
            "method": method,
            "recipe": recipe,
        }
        for architecture, method, recipe in itertools.product(ARCHITECTURES, METHODS, RECIPES)
    ]


def prohibit_post_benchmark_training(work: Path) -> None:
    if (work / "final/benchmark-access.json").exists():
        raise ValueError(
            "the final benchmark has been accessed; further training/selection is prohibited"
        )


def command(work: Path, name: str, argv: list[str]) -> None:
    destination = work / "stages"
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"Running {name}", flush=True)
    evidence_name = name
    if (destination / f"{name}.log").exists() or (destination / f"{name}.json").exists():
        evidence_name = f"{name}-{int(started * 1_000_000)}"
    log_path = destination / f"{evidence_name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [environment.get("PYTHONPATH"), str(Path(__file__).resolve().parents[3])])
        )
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    write_json(
        destination / f"{evidence_name}.json",
        {
            "argv": argv,
            "started_at_unix": started,
            "elapsed_seconds": time.time() - started,
            "returncode": result.returncode,
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path),
        },
    )
    if result.returncode:
        raise RuntimeError(f"stage {name} failed; inspect {log_path}")


def prepare(args) -> None:
    prohibit_post_benchmark_training(args.work)
    source = args.work / "sources"
    if (source / "source-report.json").exists():
        verify_sources(source)
    else:
        freeze_sources(args.config, source)
    if not (source / "annotations.jsonl").exists():
        revision = load_json_config(args.config).get("source_revision")
        if revision:
            if not (source / "revision-mask-inputs.json").exists():
                draft_revision(source)
            if not args.review.exists():
                raise ValueError("inspect native revision masks and supply the agent review record")
            accept_revision(source, args.review)
        elif not args.review.exists():
            draft_annotations(source, args.annotation_prompts)
            raise ValueError(
                "review the generated source overlays and supply the hash-bound visual review JSON"
            )
        else:
            accept_review(source, args.review)
    verify_annotations(source)
    if (
        not args.review.is_file()
        or sha256_file(args.review)
        != load_json_config(source / "annotation-report.json")["review_sha256"]
    ):
        raise ValueError("annotation review changed; prepare in a new work directory")
    prepare_originals(source, args.work / "prepared")
    config = load_json_config(args.config)
    for recipe in RECIPES:
        generate_scenes(source, args.work / "prepared", recipe=recipe, seed=config["seeds"][0])
    generate_scenes(
        source,
        args.work / "prepared",
        recipe="diagnostic",
        seed=config["synthetic"]["diagnostic"]["seed"],
    )


def train_candidate(
    args, candidate: dict, seed: int, *, classifier: bool = True, detector: bool = True
) -> None:
    prohibit_post_benchmark_training(args.work)
    recipe = candidate["recipe"]
    generate_scenes(args.work / "sources", args.work / "prepared", recipe=recipe, seed=seed)
    if detector:
        architecture = candidate["architecture"]
        argv = [
            str(args.training_python),
            "-m",
            "bixolon_scanner.training.three_bakery_detector",
            "--work",
            str(args.work),
            "--architecture",
            architecture,
            "--recipe",
            recipe,
            "--seed",
            str(seed),
            "--weights",
            str(args.ssdlite_weights if architecture == "ssdlite" else args.dfine_weights),
        ]
        if architecture == "dfine":
            argv.extend(["--repository", str(args.dfine_repository)])
        command(args.work, f"{architecture}-{recipe}-{seed}", argv)
    if classifier:
        command(
            args.work,
            f"{candidate['method']}-{recipe}-{seed}",
            [
                str(args.training_python),
                "-m",
                "bixolon_scanner.training.three_bakery_classifier",
                "--work",
                str(args.work),
                "--method",
                candidate["method"],
                "--recipe",
                recipe,
                "--seed",
                str(seed),
                "--weights",
                str(args.convnext_weights),
            ],
        )


def train(args) -> None:
    config = load_json_config(args.config)
    seed = config["seeds"][0]
    for recipe in RECIPES:
        for architecture in ARCHITECTURES:
            train_candidate(
                args, {"architecture": architecture, "recipe": recipe}, seed, classifier=False
            )
        for method in METHODS:
            train_candidate(args, {"method": method, "recipe": recipe}, seed, detector=False)
    command(
        args.work,
        "export-frozen-verifier",
        [
            str(args.training_python),
            "-m",
            "bixolon_scanner.training.three_bakery_classifier",
            "--work",
            str(args.work),
            "--method",
            "verifier",
            "--weights",
            str(args.vit_weights),
        ],
    )


def diagnose(args, candidate: dict, seed: int) -> dict:
    config = load_json_config(args.config)
    assembled = assemble(
        args.work,
        candidate["architecture"],
        candidate["method"],
        candidate["recipe"],
        seed,
        args.runtime_template,
        roi_integrity_config=getattr(args, "roi_integrity_config", None),
        provider="cpu",
    )
    if config.get("cpu_optimization"):
        return diagnose_cpu(args, candidate, seed, assembled)
    real_manifest = args.work / "prepared/original_detection.jsonl"
    stress_manifest = (
        args.work
        / f"prepared/diagnostic-{config['synthetic']['diagnostic']['seed']}/manifest.jsonl"
    )
    summaries = {}
    for name, manifest in (("real", real_manifest), ("stress", stress_manifest)):
        rows = read_jsonl(manifest)
        result = measure(
            assembled,
            rows,
            assembled / f"diagnostic-{name}",
            python=args.cuda_python,
            provider="cuda",
            cuda_dll_dir=args.cuda_dll_dir,
            warmup=config["evaluation"]["warmup"],
            match_iou=config["evaluation"]["match_iou"],
            input_identity={"manifest_sha256": sha256_file(manifest)},
        )
        summaries[name] = result["summary"]
    rank = candidate_rank(summaries["real"], summaries["stress"])
    result = {
        "candidate": candidate,
        "seed": seed,
        "rank": rank,
        "diagnostics": summaries,
        "runtime_metadata_sha256": sha256_file(assembled / "runtime/metadata.json"),
        "catalog_checksums_sha256": sha256_file(assembled / "catalog/checksums.json"),
    }
    write_json(assembled / "comparison.json", result)
    return result


def selection_diagnostic(args, candidate: dict, seed: int) -> dict:
    measurement = diagnose(args, candidate, seed)
    objective_path = getattr(args, "objective_config", None)
    if objective_path is None:
        return measurement
    result = rank_candidate(measurement, load_objective(objective_path))
    directory = args.work / f"candidates/{candidate['id']}-{seed}"
    result["measurement_comparison_sha256"] = sha256_file(directory / "comparison.json")
    result["objective_config_sha256"] = sha256_file(objective_path)
    write_json(directory / "comparison-objects.json", result)
    return result


def compare(args) -> None:
    prohibit_post_benchmark_training(args.work)
    config = load_json_config(args.config)
    settings = {
        "config_sha256": sha256_file(args.config),
        "runtime_template_sha256": sha256_file(args.runtime_template),
        "candidates": candidates(),
        "seeds": config["seeds"],
        "selection_order": (
            ["development_accuracy_and_cpu_targets"] if config.get("cpu_optimization") else []
        )
        + [
            "wrong_approvals",
            "real_multi_complete",
            "synthetic_complete",
            "missed_objects",
            "unknown_top3_misses",
            "cpu_p95" if config.get("cpu_optimization") else "cuda_p95",
        ],
        "independent_validation_available": False,
        "cpu_interpreter_identity": interpreter_identity(args.cpu_python),
    }
    prespecified = args.work / "comparison-settings.json"
    objective_path = getattr(args, "objective_config", None)
    if objective_path is not None:
        objective = load_objective(objective_path)
        settings["objective_config_sha256"] = sha256_file(objective_path)
        settings["selection_order"] = objective["selection_order"]
        prespecified = args.work / "comparison-settings-objects.json"
    if getattr(args, "roi_integrity_config", None) is not None:
        settings["roi_integrity_config_sha256"] = sha256_file(args.roi_integrity_config)
    if prespecified.exists() and load_json_config(prespecified) != settings:
        raise ValueError("comparison settings changed")
    write_json(prespecified, settings)
    base = [selection_diagnostic(args, candidate, config["seeds"][0]) for candidate in candidates()]
    base.sort(key=lambda row: (row["rank"], row["candidate"]["id"]))
    write_json(args.work / "base-comparison.json", base)
    repeated = []
    for entry in base[:2]:
        runs = [entry]
        for seed in config["seeds"][1:]:
            train_candidate(args, entry["candidate"], seed)
            runs.append(selection_diagnostic(args, entry["candidate"], seed))
        repeated.append(
            {
                "candidate": entry["candidate"],
                "runs": runs,
                "robust_rank": robust_rank(
                    [tuple(row["rank"]) for row in runs], entry["candidate"]["id"]
                ),
            }
        )
    repeated.sort(key=lambda row: row["robust_rank"])
    selection = {
        "selected": repeated[0]["candidate"],
        "payload_seed": config["seeds"][0],
        "base_candidates": base,
        "repeated_candidates": repeated,
        "comparison_settings_sha256": sha256_file(prespecified),
        "evaluation_role": "same_physical_item_development_diagnostic",
    }
    if objective_path is not None:
        selection["objective_config"] = str(objective_path.resolve())
        selection["objective_config_sha256"] = sha256_file(objective_path)
    if config.get("cpu_optimization"):
        selection["development_targets"] = {
            "eligible_base_candidate_count": sum(row["rank"][0] == 0 for row in base),
            "selected_all_seeds_eligible": all(row["rank"][0] == 0 for row in repeated[0]["runs"]),
            "selection_when_unmet": "prespecified_best_available_candidate_without_policy_tuning",
        }
    write_json(args.work / "selection.json", selection)


def evaluation_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [root / "configuration.py", Path(__file__)]
    for directory in ("evaluation", "runtime", "pipeline", "contracts", "worker"):
        paths.extend((root / directory).rglob("*.py"))
    return {str(path.resolve()): sha256_file(path) for path in sorted(set(paths))}


def export(args) -> None:
    selection_path = args.work / "selection.json"
    selection = load_json_config(selection_path)
    candidate = selection["selected"]
    output = assemble(
        args.work,
        candidate["architecture"],
        candidate["method"],
        candidate["recipe"],
        selection["payload_seed"],
        args.runtime_template,
        roi_integrity_config=getattr(args, "roi_integrity_config", None),
        provider="cpu",
    )
    freeze = {
        "candidate": candidate,
        "seed": selection["payload_seed"],
        "candidate_path": str(output.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "config_sha256": sha256_file(args.config),
        "runtime_template_sha256": sha256_file(args.runtime_template),
        "runtime_metadata_sha256": sha256_file(output / "runtime/metadata.json"),
        "catalog_checksums_sha256": sha256_file(output / "catalog/checksums.json"),
        "product_version": load_json_config(args.config)["product_version"],
        "verifier_weights": str(args.vit_weights.resolve()),
        "dfine_repository": str(args.dfine_repository.resolve()),
        "evaluation_code_sha256": evaluation_code_hashes(),
        "interpreter_identities": {
            "cpu": interpreter_identity(args.cpu_python),
            "cuda": interpreter_identity(args.cuda_python),
        },
    }
    if "objective_config" in selection:
        objective_path = Path(selection["objective_config"])
        load_objective(objective_path)
        if sha256_file(objective_path) != selection["objective_config_sha256"]:
            raise ValueError("selected accuracy objective changed")
        freeze["objective_config"] = str(objective_path)
        freeze["objective_config_sha256"] = sha256_file(objective_path)
    if load_json_config(args.config).get("cpu_optimization"):
        profile_path = output / "cpu-profile.json"
        freeze["cpu_profile"] = load_json_config(profile_path)["selected_profile"]
        freeze["cpu_profile_sha256"] = sha256_file(profile_path)
        freeze["worker_cpu_settings_sha256"] = sha256_file(output / "worker-cpu.json")
    destination = args.work / "final-candidate.json"
    if getattr(args, "roi_integrity_config", None) is not None:
        freeze["roi_integrity_config"] = str(args.roi_integrity_config.resolve())
        freeze["roi_integrity_config_sha256"] = sha256_file(args.roi_integrity_config)
        integrity = (
            args.work
            / f"roi-integrity/{candidate['method']}-{candidate['recipe']}-{selection['payload_seed']}"
        )
        freeze["roi_integrity_head"] = str((integrity / "head.pt").resolve())
        freeze["roi_integrity_head_sha256"] = sha256_file(integrity / "head.pt")
    if destination.exists() and load_json_config(destination) != freeze:
        raise ValueError("final candidate/policy already frozen with different inputs")
    write_json(destination, freeze)
    command(
        args.work,
        "parity-pytorch-ort",
        [
            str(args.training_python),
            "-m",
            "bixolon_scanner.evaluation.three_bakery_parity",
            "--work",
            str(args.work),
        ],
    )
    command(
        args.work,
        "parity-ort-cpu-cuda",
        [
            str(args.cuda_python),
            "-m",
            "bixolon_scanner.evaluation.three_bakery_parity",
            "--work",
            str(args.work),
            "--cuda-dll-dir",
            str(args.cuda_dll_dir),
        ],
    )


def evaluate(args) -> None:
    freeze_path = args.work / "final-candidate.json"
    freeze = load_json_config(freeze_path)
    config = load_json_config(args.config)
    candidate = Path(freeze["candidate_path"])
    if (
        sha256_file(args.config) != freeze["config_sha256"]
        or (
            "objective_config" in freeze
            and sha256_file(Path(freeze["objective_config"])) != freeze["objective_config_sha256"]
        )
        or (
            "roi_integrity_config" in freeze
            and (
                sha256_file(Path(freeze["roi_integrity_config"]))
                != freeze["roi_integrity_config_sha256"]
                or sha256_file(Path(freeze["roi_integrity_head"]))
                != freeze["roi_integrity_head_sha256"]
            )
        )
        or sha256_file(args.runtime_template) != freeze["runtime_template_sha256"]
        or sha256_file(candidate / "runtime/metadata.json") != freeze["runtime_metadata_sha256"]
        or sha256_file(candidate / "catalog/checksums.json") != freeze["catalog_checksums_sha256"]
        or sha256_file(args.work / "selection.json") != freeze["selection_sha256"]
        or evaluation_code_hashes() != freeze.get("evaluation_code_sha256")
        or freeze.get("interpreter_identities")
        != {
            "cpu": interpreter_identity(args.cpu_python),
            "cuda": interpreter_identity(args.cuda_python),
        }
        or (
            "cpu_profile_sha256" in freeze
            and sha256_file(candidate / "cpu-profile.json") != freeze["cpu_profile_sha256"]
        )
        or (
            "worker_cpu_settings_sha256" in freeze
            and sha256_file(candidate / "worker-cpu.json") != freeze["worker_cpu_settings_sha256"]
        )
    ):
        raise ValueError("frozen final candidate or policy changed")
    load_runtime_package_v2(candidate / "runtime")
    load_store_catalog_package(candidate / "catalog", expected_store_id="three_bakery")
    output = args.work / "final"
    rows = final_records(config, freeze_path, output)
    _, sources = verify_sources(args.work / "sources")
    if {r["image_sha256"] for r in rows} & {r["image_sha256"] for r in sources}:
        raise ValueError("final benchmark contains identical training source images")
    identity = {
        "freeze_sha256": sha256_file(freeze_path),
        "manifest_sha256": sha256_file(output / "final-inputs.jsonl"),
    }
    results = {}
    objective = (
        load_objective(Path(freeze["objective_config"])) if "objective_config" in freeze else None
    )
    for provider, python in (("cpu", args.cpu_python), ("cuda", args.cuda_python)):
        results[provider] = measure(
            candidate,
            rows,
            output / provider,
            python=python,
            provider=provider,
            cuda_dll_dir=args.cuda_dll_dir if provider == "cuda" else None,
            warmup=config["evaluation"]["warmup"],
            repetitions=config["evaluation"]["repetitions"],
            minimum_complete_images=config["evaluation"]["minimum_complete_images"]
            if objective is None
            else None,
            match_iou=config["evaluation"]["match_iou"],
            input_identity=identity,
            cpu_profile=tuple(freeze.get("cpu_profile", [4, 4])),
            maximum_p95_ms=config["evaluation"].get("maximum_cpu_p95_ms")
            if provider == "cpu"
            else None,
        )
        if objective is not None:
            results[provider] = apply_final_objective(results[provider], objective)
            results[provider]["measurement_report_sha256"] = sha256_file(
                output / provider / "report.json"
            )
            write_json(output / provider / "objective-report.json", results[provider])
    parity = provider_parity(output / "cpu/responses.jsonl", output / "cuda/responses.jsonl")
    write_json(
        output / "report.json",
        {
            "providers": results,
            "provider_parity": parity,
            "target_met": all(r["target_met_all_repetitions"] for r in results.values()),
            "claim_scope": "observed_results_on_historically_used_300_image_fixed_benchmark",
            "freeze_sha256": sha256_file(freeze_path),
        },
    )
    write_report(args.work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "compare", "export", "evaluate"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bread/three_bakery.json")
    )
    parser.add_argument(
        "--work", type=Path, default=Path("artifacts/retraining/three-bakery-reviewed")
    )
    parser.add_argument(
        "--runtime-template",
        type=Path,
        default=Path("configs/experiments/bread/three_bakery_runtime.json"),
    )
    parser.add_argument(
        "--annotation-prompts",
        type=Path,
        default=Path("configs/experiments/bread/three_bakery_annotation_prompts.json"),
    )
    parser.add_argument("--review", type=Path)
    parser.add_argument("--training-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cuda-python",
        type=Path,
        default=Path("artifacts/build-envs/worker-cuda-py311/Scripts/python.exe"),
    )
    parser.add_argument(
        "--cuda-dll-dir",
        type=Path,
        default=Path("artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64"),
    )
    cache = Path.home() / ".cache/torch/hub/checkpoints"
    parser.add_argument(
        "--ssdlite-weights",
        type=Path,
        default=cache / "ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
    )
    parser.add_argument(
        "--convnext-weights",
        type=Path,
        default=cache / "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    )
    parser.add_argument("--vit-weights", type=Path, default=cache / "dinov3_vitb16.pt")
    parser.add_argument(
        "--dfine-repository", type=Path, default=Path("artifacts/third_party/D-FINE")
    )
    parser.add_argument(
        "--dfine-weights",
        type=Path,
        default=Path("artifacts/third_party/D-FINE/pretrained/dfine_s_coco.pth"),
    )
    parser.add_argument("--roi-integrity-config", type=Path)
    parser.add_argument("--objective-config", type=Path)
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.review = args.review or args.work / "sources/visual-review.json"
    if args.stage != "evaluate":
        protect_development(args.config, args.work)
    started = time.time()
    evidence = {
        "stage": args.stage,
        "argv": sys.orig_argv,
        "python_executable": sys.executable,
        "started_at_unix": started,
        "code_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config) if args.config.is_file() else None,
        "completed": False,
    }
    try:
        {
            "prepare": prepare,
            "train": train,
            "compare": compare,
            "export": export,
            "evaluate": evaluate,
        }[args.stage](args)
        evidence["completed"] = True
    finally:
        outputs = {
            "prepare": ["sources/*.jsonl", "prepared/*.json", "prepared/*/report.json"],
            "train": ["models/*/report.json"],
            "compare": ["base-comparison.json", "selection.json"],
            "export": ["final-candidate.json", "parity/*.json"],
            "evaluate": ["final/report.*", "final/*/responses.jsonl"],
        }
        evidence["elapsed_seconds"] = time.time() - started
        evidence["output_sha256"] = {
            path.relative_to(args.work).as_posix(): sha256_file(path)
            for pattern in outputs[args.stage]
            for path in sorted(args.work.glob(pattern))
            if path.is_file()
        }
        write_json(
            args.work / f"stages/orchestration-{args.stage}-{int(started * 1_000_000)}.json",
            evidence,
        )
    print(json.dumps({"stage": args.stage, "completed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
