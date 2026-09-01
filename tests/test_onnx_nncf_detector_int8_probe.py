from bixolon_scanner.experiments.bread.onnx_nncf_detector_int8_probe import (
    _sample_rows,
)


def test_detector_calibration_sampling_covers_both_manifest_ends() -> None:
    rows = [{"index": index} for index in range(10)]

    selected = _sample_rows(rows, 4)

    assert [row["index"] for row in selected] == [0, 3, 6, 9]


def test_detector_calibration_sampling_keeps_short_manifest() -> None:
    rows = [{"index": index} for index in range(3)]

    assert _sample_rows(rows, 4) == rows
