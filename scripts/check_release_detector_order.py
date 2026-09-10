"""Supplement raw SSDLite parity with class-blind active-box correspondence."""

import argparse
from pathlib import Path

import numpy as np

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.three_bakery_parity import matched_detector_outputs
from bixolon_scanner.runtime.onnx_session import OrtRunner
from bixolon_scanner.training.three_bakery_data import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["torch", "cuda"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = root / "artifacts/versions/0.1.17"
    work = version / "parity-work"
    metadata = load_json_config(version / "staging/runtime/metadata.json")["detector"]
    threshold = metadata["uncertainty_score_threshold"]
    if args.mode == "torch":
        import torch

        from bixolon_scanner.training.ssdlite_objectness_detector import (
            build_export_wrapper,
            build_ssdlite_objectness,
        )

        torch.set_num_threads(4)
        model = build_ssdlite_objectness(device="cpu")
        model.load_state_dict(
            torch.load(
                work / "models/ssdlite-dense-20260908/model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        runner = build_export_wrapper(model.eval(), detector_class_count=1)
    else:
        runner = OrtRunner(
            version / "staging/runtime/detector.onnx", "cuda", version / "staging/cuda-runtime"
        )
    checks = []
    fixtures = load_json_config(work / "parity/tensor-fixtures.json")
    for row in fixtures["fixtures"]:
        if row["model"] != "detector.onnx":
            continue
        path = work / "parity/tensors" / row["file"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("source-only parity fixture changed")
        with np.load(path) as fixture:
            if args.mode == "torch":
                with torch.inference_mode():
                    actual = [v.numpy() for v in runner(torch.from_numpy(fixture["input"].copy()))]
            else:
                actual = runner.run(row["output_names"], row["input_name"], fixture["input"])
            expected = [fixture[f"output_{i}"] for i in range(len(actual))]
        filtered = []
        for values in (actual, expected):
            logits = values[row["output_names"].index("logits")]
            probability = 1 / (1 + np.exp(-np.clip(logits[0, :, 0], -80, 80)))
            active = probability >= threshold
            filtered.append([value[:, active] for value in values])
        counts = [len(values[0][0]) for values in filtered]
        if counts == [0, 0]:
            matched = {"both_empty": True}
            passed = True
        elif counts[0] != counts[1]:
            matched = {"active_query_count_mismatch": counts}
            passed = False
        else:
            matched = matched_detector_outputs(row["output_names"], *filtered)
            passed = all(
                batch["complete_correspondence"]
                and all(v["allclose"] for v in batch["outputs"].values())
                for batch in matched["batches"]
            )
        checks.append(
            {
                "fixture": row["file"],
                "active_query_counts": counts,
                "passed": passed,
                "alignment": matched,
            }
        )
    if args.mode == "cuda":
        runner.close()
    write_json(
        work / f"parity/ssdlite-active-{args.mode}.json",
        {
            "scope": "supplementary_source_fixture_diagnostic_only; original raw-query failures retained",
            "threshold_source": "detector.uncertainty_score_threshold",
            "threshold": threshold,
            "comparison": f"{args.mode}_vs_ORT_CPU",
            "checks": checks,
            "all_active_queries_close": bool(checks) and all(c["passed"] for c in checks),
            "code_sha256": sha256_file(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
