"""Compare source-only detector candidates on exposed, untrained log images."""

from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.three_bakery_http import measure
from bixolon_scanner.operations.detector_variant import assemble_detector_variant
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    config = load_json_config(Path("configs/experiments/bread/n100_020.json"))
    work = Path(config["work"])
    baseline = Path(config["baseline"])
    model = work / "training/ssdlite-occlusion"
    candidate = work / "candidates/ssdlite-occlusion"
    assemble_detector_variant(
        baseline,
        baseline / "runtime",
        candidate,
        model / "report.json",
        model / "contract.json",
        Path(config["original_manifest"]),
        model / "detector.onnx",
    )
    report = measure(
        candidate,
        read_jsonl(Path(config["log_manifest"])),
        work / "measurements/ssdlite-occlusion",
        python=Path("artifacts/build-envs/worker-cpu-0.1.17-py311/Scripts/python.exe"),
        provider="cpu",
        cpu_profile=(8, 12),
        warmup=10,
        repetitions=1,
        maximum_p95_ms=200,
        input_identity={"role": "exposed development log132, never trained"},
    )
    print(report["summary"], flush=True)
    write_json(work / "occlusion-summary.json", report["summary"])


if __name__ == "__main__":
    main()
