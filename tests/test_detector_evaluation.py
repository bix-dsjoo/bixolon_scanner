import numpy as np
import json
from argparse import Namespace

from bixolon_scanner.training.aggregate_detector import aggregate
from bixolon_scanner.training.evaluate_detector import _iou, _metrics


def test_detector_metrics_match_boxes_and_count():
    records = [
        {
            "annotations": [
                {"bbox_xywh": [10.0, 20.0, 30.0, 40.0]},
                {"bbox_xywh": [100.0, 100.0, 20.0, 20.0]},
            ]
        }
    ]
    predictions = [
        {
            "boxes_xyxy": [[10.0, 20.0, 40.0, 60.0], [100.0, 100.0, 120.0, 120.0]],
            "scores": [0.99, 0.98],
        }
    ]
    result = _metrics(
        records,
        predictions,
        score_threshold=0.5,
        nms_iou_threshold=0.7,
        match_iou_threshold=0.5,
        max_queries=300,
    )
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["count_accuracy"] == 1.0
    assert _iou(np.asarray([0, 0, 10, 10]), np.asarray([20, 20, 30, 30])) == 0.0


def test_oof_aggregation_selects_one_common_threshold(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dataset_version": "bread-test"}), encoding="utf-8"
    )
    prediction = tmp_path / "fold0.jsonl"
    output = tmp_path / "report.json"
    record = {
        "record_type": "detection",
        "split": "development",
        "image_id": 7,
        "annotations": [{"bbox_xywh": [10.0, 20.0, 30.0, 40.0]}],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    prediction.write_text(
        json.dumps(
            {
                "image_id": 7,
                "boxes_xyxy": [[10.0, 20.0, 40.0, 60.0]],
                "scores": [0.8],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    aggregate(
        Namespace(
            manifest=manifest,
            predictions=[prediction],
            output=output,
            nms_threshold=0.7,
            match_iou_threshold=0.5,
            target_recall=0.99,
            min_score_threshold=0.5,
            max_score_threshold=0.9,
            threshold_steps=5,
            max_queries=300,
        )
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["threshold_policy"] == "selected_on_oof-development"
    assert report["selected_score_threshold"] == 0.8
    assert report["metrics"]["recall"] == 1.0
