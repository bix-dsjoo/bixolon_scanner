"""Run serial HTTP comparisons on exposed development logs with frozen GT."""

import argparse
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.three_bakery_http import measure
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="+",
        choices=[
            "baseline",
            "score030",
            "score040",
            "score040-maskoff",
            "score040-bias0",
            "context040",
        ],
    )
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--run", default="development")
    args = parser.parse_args()
    work = Path("artifacts/retraining/recapture-0.2.1")
    records = read_jsonl(
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
    )
    for name in args.names:
        source = Path("artifacts/versions/0.2.0/staging")
        candidate = source if name == "baseline" else work / "candidates" / name
        if name != "baseline" and not candidate.exists():
            for directory in ("runtime", "catalog"):
                shutil.copytree(source / directory, candidate / directory)
            metadata = load_json_config(candidate / "runtime/metadata.json")
            metadata["detector"]["score_threshold"] = 0.03 if name == "score030" else 0.04
            if name == "context040":
                metadata["detector"]["score_threshold"] = 0.025
                metadata["quality"]["detector_output_score_threshold"] = 0.04
            if name == "score040-maskoff":
                metadata["embedder"]["neighbor_mask"] = False
                metadata["classifier_resolution_fallback"]["embedder"]["neighbor_mask"] = False
            if name == "score040-bias0":
                metadata["embedder"]["neighbor_distance_bias"] = 0.0
                metadata["classifier_resolution_fallback"]["embedder"]["neighbor_distance_bias"] = (
                    0.0
                )
            write_json(candidate / "runtime/metadata.json", metadata)
            write_json(
                candidate / "hypothesis.json",
                {
                    "role": "exposed log132 development; no weight training or GT edits",
                    "hypothesis": "Prune low objectness fragments before crop ownership; mask changes test occlusion-induced false multi-object rejection.",
                    "detector_score_threshold": metadata["detector"]["score_threshold"],
                    "minimum_correct_approved_count": 1078,
                    "maximum_missed_count": 0,
                    "maximum_wrong_approved_count": 0,
                    "maximum_extra_count": 37,
                    "maximum_recapture_count": 38,
                },
            )
        report = measure(
            candidate,
            records,
            work / "measurements" / args.run / f"{name}-{args.provider}",
            python=Path("artifacts/build-envs")
            / ("worker-cuda-py311" if args.provider == "cuda" else "worker-cpu-0.1.17-py311")
            / "Scripts/python.exe",
            provider=args.provider,
            cpu_profile=(8, 12),
            warmup=10,
            repetitions=1,
            cuda_dll_dir=Path("artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64")
            if args.provider == "cuda"
            else None,
        )
        print(name, args.provider, report["summary"], flush=True)


if __name__ == "__main__":
    main()
