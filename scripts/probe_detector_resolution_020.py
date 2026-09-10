"""Export fixed larger-resolution detector candidates from the authorized checkpoint."""

import shutil
from pathlib import Path

import torch

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2
from bixolon_scanner.training.ssdlite_objectness_detector import (
    build_ssdlite_objectness,
    export_ssdlite_onnx,
)
from bixolon_scanner.training.three_bakery_data import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    base = load_json_config(root / "configs/experiments/bread/n100_020.json")
    source = root / "artifacts/retraining/three-bakery-revised300/models/ssdlite-dense-20260908"
    training = load_json_config(source / "report.json")
    if training["output_sha256"]["model.pt"] != sha256_file(source / "model.pt"):
        raise ValueError("training checkpoint changed")
    torch.set_num_threads(4)
    model = build_ssdlite_objectness(pretrained_detector_transfer=False)
    model.load_state_dict(torch.load(source / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()
    for size in [512, 640]:
        output = root / base["work"] / "candidates" / f"ssdlite-{size}"
        if output.exists():
            raise ValueError("inspect existing resolution output before resuming")
        shutil.copytree(root / base["baseline"] / "runtime", output / "runtime")
        shutil.copytree(root / base["baseline"] / "catalog", output / "catalog")
        graph = output / "runtime/detector.onnx"
        export_ssdlite_onnx(model, graph, input_size=size, detector_class_count=1)
        metadata = load_json_config(output / "runtime/metadata.json")
        metadata["detector"]["input_size"] = [size, size]
        metadata["checksums"]["detector.onnx"] = sha256_file(graph)
        write_json(output / "runtime/metadata.json", metadata)
        load_runtime_package_v2(output / "runtime")
        write_json(
            output / "resolution-export.json",
            {
                "checkpoint_sha256": sha256_file(source / "model.pt"),
                "training_report_sha256": sha256_file(source / "report.json"),
                "input_size": size,
                "detector_sha256": sha256_file(graph),
                "weights_trained_again": False,
            },
        )
        c = {
            **base,
            "baseline": str(output),
            "work": f"artifacts/retraining/n100-0.2.0/ssdlite-{size}",
        }
        write_json(root / f"configs/experiments/bread/n100_020_ssdlite-{size}.json", c)
        print(size, flush=True)


if __name__ == "__main__":
    main()
