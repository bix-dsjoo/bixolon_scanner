from pathlib import Path

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.dinov3_exact_count_probe import (
    count_logits,
    count_metrics,
    create_ort_session,
    cross_validated_count_probe,
    export_exact_count_onnx,
    safety_sweep,
)


def test_cross_validated_count_probe_keeps_auxiliary_rows_out_of_validation() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [-2.8, 0.1],
            [2.8, 0.1],
            [0.1, 2.8],
            [-2.6, -0.1],
            [2.6, -0.1],
            [-0.1, 2.6],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 6, 7, 1, 6, 7, 1, 6, 7], dtype=np.int64)
    folds = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    auxiliary_features = np.asarray([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    auxiliary_labels = np.asarray([1, 6, 7], dtype=np.int64)

    result = cross_validated_count_probe(
        features,
        labels,
        folds,
        regularization_values=[0.1],
        seed=7,
        auxiliary_features=auxiliary_features,
        auxiliary_labels=auxiliary_labels,
    )

    assert result["selected_metrics"]["image_count"] == 9
    assert result["selected_metrics"]["exact_accuracy"] == pytest.approx(1.0)
    assert result["selected_logits"].shape == (9, 3)


def test_cross_validated_count_probe_rejects_partial_auxiliary_input() -> None:
    features = np.zeros((6, 2), dtype=np.float32)
    labels = np.asarray([1, 6, 1, 6, 1, 6], dtype=np.int64)
    folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="provided together"):
        cross_validated_count_probe(
            features,
            labels,
            folds,
            regularization_values=[0.1],
            seed=7,
            auxiliary_features=np.zeros((2, 2), dtype=np.float32),
        )


def test_safety_sweep_catches_unsafe_miss_and_counts_only_new_false_recapture() -> None:
    records = [
        {"image_id": 1, "count_label": 6},
        {"image_id": 2, "count_label": 6},
        {"image_id": 3, "count_label": 6},
    ]
    classes = np.asarray([5, 6, 7], dtype=np.int64)
    logits = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    detector_values = {
        1: {"status": "SEGMENTATION", "detector_count": 5, "segment_recapture": False},
        2: {"status": "SEGMENTATION", "detector_count": 6, "segment_recapture": False},
        3: {"status": "IMAGE_RECAPTURE", "detector_count": 0, "segment_recapture": False},
    }

    result = safety_sweep(
        records,
        logits,
        classes,
        detector_values,
        temperature=1.0,
    )[0]

    assert result == {
        "confidence_threshold": 0.0,
        "trigger_count": 3,
        "dangerous_miss_image_count": 1,
        "caught_dangerous_miss_count": 1,
        "unnecessary_new_recapture_count": 1,
    }


def test_exported_exact_count_head_matches_numpy(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source_path = tmp_path / "presence.onnx"
    output_path = tmp_path / "exact.onnx"
    source = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["input"], ["feature"]),
                helper.make_node("Sub", ["feature", "feature_mean"], ["centered"]),
                helper.make_node("MatMul", ["centered", "coefficient"], ["linear"]),
                helper.make_node("Add", ["linear", "intercept"], ["logits"]),
            ],
            "synthetic-presence-verifier",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
            [
                numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="feature_mean"),
                numpy_helper.from_array(np.eye(2, dtype=np.float32), name="coefficient"),
                numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="intercept"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    source.ir_version = 9
    onnx.save(source, source_path)
    head = {
        "classes": np.asarray([1, 6, 7], dtype=np.int64),
        "mean": np.asarray([0.5, -0.5], dtype=np.float32),
        "scale": np.asarray([2.0, 4.0], dtype=np.float32),
        "coefficient": np.asarray([[1.0, -1.0], [-0.5, 0.5], [0.25, 0.75]], dtype=np.float32),
        "intercept": np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
    }

    export_exact_count_onnx(source_path, output_path, head)
    session = create_ort_session(str(output_path), "cpu")
    features = np.asarray([[2.5, 3.5]], dtype=np.float32)
    actual = session.run(["logits"], {"input": features})[0]

    assert output_path.is_file()
    assert actual == pytest.approx(count_logits(features, head), abs=1e-6)
    metrics = count_metrics(
        np.asarray([int(head["classes"][actual.argmax(axis=1)[0]])]),
        actual,
        head["classes"],
    )
    assert metrics["exact_accuracy"] == pytest.approx(1.0)
