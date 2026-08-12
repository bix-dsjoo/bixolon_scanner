import numpy as np
import json
from argparse import Namespace

import bixolon_scanner.training.evaluate_detector as detector_evaluation
from bixolon_scanner.training.aggregate_detector import aggregate
from bixolon_scanner.training.evaluate_detector import _iou, _metrics, _metrics_grid


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


def _assert_grid_matches_brute_force(records, predictions, thresholds, nms_threshold):
    optimized = _metrics_grid(
        records,
        predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=nms_threshold,
        match_iou_threshold=0.5,
        max_queries=7,
    )
    brute_force = []
    for threshold in thresholds:
        metrics = _metrics(
            records,
            predictions,
            score_threshold=float(threshold),
            nms_iou_threshold=nms_threshold,
            match_iou_threshold=0.5,
            max_queries=7,
        )
        metrics["score_threshold"] = float(threshold)
        brute_force.append(metrics)
    assert optimized == brute_force


def test_metric_grid_cached_nms_matches_brute_force_for_overlaps_ties_and_edges():
    records = [
        {
            "annotations": [
                {"bbox_xywh": [0.0, 0.0, 10.0, 10.0]},
                {"bbox_xywh": [20.0, 20.0, 10.0, 10.0]},
            ]
        }
    ]
    predictions = [
        {
            "boxes_xyxy": [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ],
            "scores": [0.5, 0.5, 0.25, 1.0, 0.0],
        }
    ]
    thresholds = [0.5, 0.0, 1.0, 0.5, 0.25, 0.5000000001]
    for nms_threshold in (0.0, 0.5, 1.0):
        _assert_grid_matches_brute_force(
            records, predictions, thresholds, nms_threshold
        )


def test_metric_grid_cached_nms_matches_brute_force_on_randomized_boxes():
    thresholds = [0.9, 0.05, 0.5, 0.2, 0.5, 0.95]
    for seed in range(10):
        rng = np.random.default_rng(seed)
        records = []
        predictions = []
        for _ in range(4):
            gt_xy = rng.uniform(0, 80, size=(3, 2))
            gt_wh = rng.uniform(5, 30, size=(3, 2))
            records.append(
                {
                    "annotations": [
                        {"bbox_xywh": [*xy.tolist(), *wh.tolist()]}
                        for xy, wh in zip(gt_xy, gt_wh)
                    ]
                }
            )
            anchors = rng.uniform(0, 80, size=(20, 2))
            sizes = rng.uniform(5, 35, size=(20, 2))
            boxes = np.concatenate([anchors, anchors + sizes], axis=1).tolist()
            # Add exact duplicates and tied scores to exercise stable ordering.
            boxes.extend([boxes[0], boxes[0], boxes[1]])
            scores = rng.choice(
                np.asarray([0.05, 0.2, 0.5, 0.9, 0.95]), size=20
            ).tolist()
            scores.extend([0.5, 0.5, 0.2])
            predictions.append({"boxes_xyxy": boxes, "scores": scores})
        for nms_threshold in (0.3, 0.7):
            _assert_grid_matches_brute_force(
                records, predictions, thresholds, nms_threshold
            )


def test_metric_grid_runs_nms_once_per_image_not_once_per_threshold(monkeypatch):
    calls = 0
    original = detector_evaluation._nms

    def counted_nms(detections, threshold):
        nonlocal calls
        calls += 1
        return original(detections, threshold)

    monkeypatch.setattr(detector_evaluation, "_nms", counted_nms)
    records = [{"annotations": []}, {"annotations": []}]
    predictions = [
        {"boxes_xyxy": [[0.0, 0.0, 1.0, 1.0]], "scores": [0.5]},
        {"boxes_xyxy": [[1.0, 1.0, 2.0, 2.0]], "scores": [0.6]},
    ]

    _metrics_grid(
        records,
        predictions,
        score_thresholds=np.linspace(0.05, 0.95, 91),
        nms_iou_threshold=0.7,
        match_iou_threshold=0.5,
        max_queries=300,
    )

    assert calls == len(records)


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


def test_oof_aggregation_keeps_precommitted_threshold(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dataset_version": "bread-test"}), encoding="utf-8"
    )
    prediction = tmp_path / "fold0.jsonl"
    output = tmp_path / "report.json"
    manifest.write_text(
        json.dumps(
            {
                "record_type": "detection",
                "split": "development",
                "image_id": 7,
                "annotations": [{"bbox_xywh": [10.0, 20.0, 30.0, 40.0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
            score_threshold=0.56,
            min_score_threshold=0.05,
            max_score_threshold=0.95,
            threshold_steps=91,
            max_queries=300,
        )
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["threshold_policy"] == "fixed"
    assert report["selected_score_threshold"] == 0.56


def test_oof_aggregation_excludes_quality_only_records(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dataset_version": "bread-test"}), encoding="utf-8"
    )
    prediction = tmp_path / "fold0.jsonl"
    output = tmp_path / "report.json"
    records = [
        {
            "record_type": "detection",
            "split": "development",
            "image_id": 7,
            "annotations": [{"bbox_xywh": [10.0, 20.0, 30.0, 40.0]}],
        },
        {
            "record_type": "detection",
            "split": "development",
            "image_id": 8,
            "annotations": [],
            "exclude_from_detector_training": True,
        },
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
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
            score_threshold=0.56,
            min_score_threshold=0.05,
            max_score_threshold=0.95,
            threshold_steps=91,
            max_queries=300,
        )
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["image_count"] == 1
