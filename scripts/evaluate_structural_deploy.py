"""Exercise complete explicit-Catalog packages on CPU or actual Intel GPU."""

import argparse
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.experiments.bread.structural_regression import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bread/n100_structural.json")
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["same", "openvino_gpu"], required=True)
    args = parser.parse_args()
    config = load_json_config(args.config)
    for dataset in ["log", "original", "final"]:
        evaluate(
            config,
            args.output,
            dataset,
            detector_runtime=args.bundle / "runtime",
            catalog_dir=args.bundle / "catalog",
            embedder_provider=args.provider,
            timing_scope="complete standalone package accuracy; build-host training may be active",
        )


if __name__ == "__main__":
    main()
