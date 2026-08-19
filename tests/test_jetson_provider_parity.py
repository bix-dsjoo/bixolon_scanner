from pathlib import Path

import numpy as np

from bixolon_scanner.experiments.rpc200.jetson_provider_parity import _compare
from bixolon_scanner.experiments.rpc200.tensorrt_native import TensorRTRunner

ROOT = Path(__file__).resolve().parents[1]


def _response(*, bbox_x: int = 10, confidence: float = 0.9, class_id: str = "1"):
    return {
        "status": "SEGMENTATION",
        "reason_codes": [],
        "segmentations": [
            {
                "bbox": {"x": bbox_x, "y": 20, "width": 30, "height": 40},
                "status": "APPROVED",
                "reason_codes": [],
                "prediction": {"class_id": class_id, "class_name": class_id},
                "top3": [],
                "confidence": confidence,
            }
        ],
    }


def test_jetson_provider_parity_accepts_one_pixel_and_small_confidence_error():
    assert _compare(_response(), _response(bbox_x=11, confidence=0.909)) == []


def test_jetson_provider_parity_rejects_changed_class_and_large_bbox_error():
    errors = _compare(_response(), _response(bbox_x=12, class_id="2"))

    assert "segmentation[0] prediction differs" in errors
    assert "segmentation[0] bbox differs by 2px" in errors


def test_tensorrt_runner_splits_batches_at_the_engine_profile_limit():
    runner = TensorRTRunner.__new__(TensorRTRunner)
    runner._engine = type(
        "Engine",
        (),
        {"get_tensor_profile_shape": lambda _self, _name, _profile: ((1,), (2,), (2,))},
    )()
    batch_sizes: list[int] = []

    def run_inputs(_output_names, inputs):
        values = inputs["pixel_values"]
        batch_sizes.append(len(values))
        return [values.copy()]

    runner.run_inputs = run_inputs
    values = np.arange(5, dtype=np.float32).reshape(5, 1)

    (actual,) = runner.run(["logits"], "pixel_values", values)

    assert batch_sizes == [2, 2, 1]
    np.testing.assert_array_equal(actual, values)


def test_jetson_bundle_passes_native_engine_paths_to_parity_and_benchmark():
    script = (ROOT / "scripts" / "prepare_rpc200_v18_jetson.ps1").read_text(encoding="utf-8")

    for filename in (
        "validation_benchmark.py",
        "jetson_provider_parity.py",
        "tensorrt_native.py",
    ):
        assert filename in script
    assert '--detector-engine "$DETECTOR_ENGINE"' in script
    assert '--classifier-engine "$CLASSIFIER_ENGINE"' in script
    assert "python3 -m venv --system-site-packages .venv" in script
    assert "benchmarks CUDA EP and native TensorRT" in script
    assert "192.168." not in script
    assert "idp-robot" not in script


def test_portable_bundle_records_requested_commit_and_writes_bomless_manifest():
    script = (ROOT / "scripts" / "prepare_rpc200_v18_portable.ps1").read_text(encoding="utf-8")

    assert '$runScript.Replace("__SOURCE_COMMIT__", $Commit)' in script
    assert "[IO.File]::WriteAllLines(" in script
    assert "[Text.UTF8Encoding]::new($false)" in script
    assert "C:\\workspace" not in script
    assert "COMPUTERNAME" not in script
