from types import SimpleNamespace

import numpy as np
import pytest

from bixolon_scanner.contracts.errors import ModelExecutionError
from bixolon_scanner.runtime.inference_timing import (
    active_inference_timings,
    collect_inference_timings,
)
from bixolon_scanner.runtime.onnx_session import OrtRunner


def runner(run):
    value = OrtRunner.__new__(OrtRunner)
    value.model_name = "primary.onnx"
    value.provider = "cpu"
    value.cuda = False
    value.session = SimpleNamespace(run=run)
    return value


def test_model_timings_are_opt_in_and_nested_scopes_are_isolated():
    value = runner(lambda names, inputs: [inputs["input"]])
    tensor = np.ones((2, 3), dtype=np.float32)
    assert active_inference_timings() is None
    with collect_inference_timings() as outer:
        np.testing.assert_array_equal(value.run(["output"], "input", tensor)[0], tensor)
        with collect_inference_timings() as inner:
            value.run(["output"], "input", tensor[:1])
        assert len(outer) == len(inner) == 1
        assert outer[0]["batch_size"] == 2
        assert inner[0]["batch_size"] == 1
        assert outer[0]["succeeded"]
        assert set(outer[0]) == {"model", "provider", "batch_size", "elapsed_ms", "succeeded"}
    assert active_inference_timings() is None


def test_model_failure_remains_an_error_and_is_timed():
    def fail(*args):
        raise RuntimeError("internal failure")

    with collect_inference_timings() as calls:
        with pytest.raises(ModelExecutionError):
            runner(fail).run(["output"], "input", np.zeros((1, 3)))
    assert calls[0]["succeeded"] is False
    assert calls[0]["elapsed_ms"] >= 0
    assert active_inference_timings() is None
