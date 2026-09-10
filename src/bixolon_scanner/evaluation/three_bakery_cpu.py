"""Prespecified, serial CPU thread comparison using source-only HTTP diagnostics."""

from __future__ import annotations

from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..training.three_bakery_data import read_jsonl, write_json
from .three_bakery import candidate_rank
from .three_bakery_http import measure, provider_parity


def profile_order(row: dict) -> tuple:
    latency = row["latency"]["full_path"]
    return (
        latency["p95_ms"] if latency["p95_ms"] is not None else 1e308,
        latency["p99_ms"] if latency["p99_ms"] is not None else 1e308,
        sum(row["profile"]),
        tuple(row["profile"]),
    )


def developmental_rank(real: dict, stress: dict, latency: dict, settings: dict) -> tuple:
    rank = candidate_rank(real, stress)
    eligible = (
        rank[0] == 0
        and real["by_difficulty"]["multi"]["complete_images"]
        >= settings["minimum_real_multi_complete"]
        and latency["target_met"]
    )
    return (0 if eligible else 1, *rank[:-1], latency["worst_full_path_p95_ms"])


def diagnose_cpu(args, candidate: dict, seed: int, assembled: Path) -> dict:
    config = load_json_config(args.config)
    settings = config["cpu_optimization"]
    real_manifest = args.work / "prepared/original_detection.jsonl"
    stress_manifest = (
        args.work
        / f"prepared/diagnostic-{config['synthetic']['diagnostic']['seed']}/manifest.jsonl"
    )
    real = read_jsonl(real_manifest)
    stress = read_jsonl(stress_manifest)
    multi = [row for row in real if row["difficulty"] == "multi"]
    positive = [row for row in stress if row["annotations"]]
    diagnostic_count = config["synthetic"]["diagnostic"]["image_count"]
    expected_positive = diagnostic_count - round(
        diagnostic_count * config["synthetic"]["empty_probability"]
    )
    if (
        len(multi) != config["expected_counts"]["multi"]
        or len(stress) != diagnostic_count
        or len(positive) != expected_positive
    ):
        raise ValueError("CPU source diagnostic population mismatch")
    identity = {
        "real_manifest_sha256": sha256_file(real_manifest),
        "stress_manifest_sha256": sha256_file(stress_manifest),
        "configuration_sha256": sha256_file(args.config),
    }

    def run(name, records, profile, repetitions=1):
        return measure(
            assembled,
            records,
            assembled / name,
            python=args.cpu_python,
            provider="cpu",
            cpu_profile=tuple(profile),
            warmup=settings["warmup"],
            repetitions=repetitions,
            match_iou=config["evaluation"]["match_iou"],
            input_identity={**identity, "population": name},
        )

    base_seed = config["seeds"][0]
    if seed == base_seed:
        profiles = []
        baseline = settings["baseline"]
        baseline_name = f"cpu-profile-{baseline[0]}-{baseline[1]}"
        ordered = [baseline] + [p for p in settings["profiles"] if p != baseline]
        for profile in ordered:
            name = f"cpu-profile-{profile[0]}-{profile[1]}"
            result = run(name, multi, profile)
            if result["summary"]["image_status_counts"].get("ERROR", 0):
                raise ValueError("CPU profile measurement contains Worker errors")
            parity = provider_parity(
                assembled / baseline_name / "responses.jsonl",
                assembled / name / "responses.jsonl",
            )
            profiles.append({"profile": profile, "latency": result["latency"], "parity": parity})
        eligible = [p for p in profiles if p["parity"]["status_rank_parity"]]
        chosen = min(eligible, key=profile_order)["profile"]
        profile_record = {
            "selected_profile": chosen,
            "baseline": baseline,
            "profiles": profiles,
            "input_identity": identity,
            "cpu_model": settings["cpu_model"],
            "selection_order": ["full_path_p95", "full_path_p99", "thread_sum", "profile"],
        }
    else:
        base = args.work / f"candidates/{candidate['id']}-{base_seed}/cpu-profile.json"
        profile_record = load_json_config(base)
        base_result = load_json_config(base.parent / "comparison.json")
        if (
            sha256_file(base) != base_result["cpu_profile_sha256"]
            or profile_record["input_identity"] != identity
        ):
            raise ValueError("base-seed CPU profile or diagnostic inputs changed")
        chosen = profile_record["selected_profile"]
        profile_record = {
            **profile_record,
            "fixed_from_base_seed_sha256": sha256_file(base),
            "retuned": False,
        }
    write_json(assembled / "cpu-profile.json", profile_record)
    write_json(
        assembled / "worker-cpu.json",
        {
            "product_version": config["product_version"],
            "python_executable": str(args.cpu_python.resolve()),
            "entrypoint": "from bixolon_scanner.worker.cli import serve; serve()",
            "environment": {
                "BIXOLON_PACKAGE_DIR": str((assembled / "runtime").resolve()),
                "BIXOLON_CATALOG_DIR": str((assembled / "catalog").resolve()),
                "BIXOLON_CATALOG_STORE_ID": "three_bakery",
                "BIXOLON_PROVIDER": "cpu",
                "BIXOLON_HOST": "127.0.0.1",
                "BIXOLON_LOG_LEVEL": "INFO",
                "BIXOLON_REQUEST_TIMEOUT_SECONDS": "120",
                "BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS": str(chosen[0]),
                "BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS": str(chosen[1]),
            },
            "cpu_profile_sha256": sha256_file(assembled / "cpu-profile.json"),
        },
    )
    summaries = {
        "real": run("diagnostic-real", real, chosen)["summary"],
        "stress": run("diagnostic-stress", stress, chosen)["summary"],
    }
    timings = {
        "real_multi": run("latency-real-multi", multi, chosen, settings["repetitions"]),
        "synthetic_positive": run(
            "latency-stress-positive", positive, chosen, settings["repetitions"]
        ),
    }
    p95s = [
        repetition["latency"]["full_path"]["p95_ms"]
        for result in timings.values()
        for repetition in result["repetitions"]
    ]
    if any(
        repetition["image_status_counts"].get("ERROR", 0)
        for result in timings.values()
        for repetition in result["repetitions"]
    ):
        raise ValueError("development latency repetitions contain Worker errors")
    latency = {
        "populations": timings,
        "maximum_p95_ms": settings["maximum_p95_ms"],
        "target_met": all(p is not None and p <= settings["maximum_p95_ms"] for p in p95s),
        "worst_full_path_p95_ms": max(p if p is not None else 1e308 for p in p95s),
    }
    result = {
        "candidate": candidate,
        "seed": seed,
        "rank": developmental_rank(summaries["real"], summaries["stress"], latency, settings),
        "diagnostics": summaries,
        "cpu_latency": latency,
        "cpu_profile": chosen,
        "cpu_profile_sha256": sha256_file(assembled / "cpu-profile.json"),
        "runtime_metadata_sha256": sha256_file(assembled / "runtime/metadata.json"),
        "catalog_checksums_sha256": sha256_file(assembled / "catalog/checksums.json"),
    }
    write_json(assembled / "comparison.json", result)
    return result
