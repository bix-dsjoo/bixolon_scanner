import numpy as np
import pytest

from bixolon_scanner.contracts import ScanResponse
from bixolon_scanner.evaluation.three_bakery import (
    match_boxes,
    robust_rank,
    score_response,
    summarize,
)


def response(items=(), status="SEGMENTATION"):
    return ScanResponse.model_validate(
        {
            "request_id": "three-bakery-test",
            "status": status,
            "reason_codes": (
                ["SEGMENT_BELOW_APPROVAL_THRESHOLD"]
                if any(i["status"] == "UNKNOWN" for i in items)
                else []
            )
            if status == "SEGMENTATION"
            else ["IMAGE_RECAPTURE_REQUIRED"]
            if status == "IMAGE_RECAPTURE"
            else ["MODEL_EXECUTION_FAILED"],
            "segmentations": list(items),
            "processing_time_ms": 1,
            "worker_version": "0.1.16",
            "detector_version": "0.1.16",
            "classifier_version": "0.1.16" if items else None,
        }
    )


def item(i=1, category=1, x=0):
    return {
        "segmentation_id": f"segmentation_{i:03}",
        "bbox": {"x": x, "y": 0, "width": 10, "height": 10},
        "status": "APPROVED",
        "reason_codes": [],
        "prediction": {"class_id": f"bread_{category:02}", "class_name": "Bread"},
        "top3": [],
        "confidence": 0.95,
    }


GT = [{"bbox_xywh": [0, 0, 10, 10], "category_id": 1}]


def test_matching_maximizes_cardinality_before_iou(monkeypatch):
    monkeypatch.setattr(
        "bixolon_scanner.evaluation.three_bakery.box_iou_matrix",
        lambda a, b: np.array([[0.9, 0.6], [0.7, 0.1]]),
    )
    assert match_boxes([[0] * 4] * 2, [[0] * 4] * 2, 0.5) == {0: 1, 1: 0}


def test_complete_requires_exact_correct_public_outputs():
    assert score_response(response([item()]), GT)["complete_image"]
    wrong = score_response(response([item(category=2)]), GT)
    assert not wrong["complete_image"] and wrong["wrong_class_approved_count"] == 1
    duplicate = score_response(response([item(), item(2)]), GT)
    assert not duplicate["complete_image"] and duplicate["unmatched_approved_count"] == 1
    assert duplicate["duplicate_count"] == duplicate["duplicate_approved_count"] == 1
    assert duplicate["background_approved_count"] == 0
    background = score_response(response([item(), item(2, x=30)]), GT)
    assert background["background_approved_count"] == background["wrong_approved_count"] == 1
    assert background["duplicate_count"] == 0


@pytest.mark.parametrize("status", ["IMAGE_RECAPTURE", "ERROR"])
def test_nonsegmentation_keeps_all_gt_in_denominator(status):
    metrics = score_response(response(status=status), GT)
    assert metrics["missed_count"] == 1 and not metrics["complete_image"]
    summary = summarize([{"metrics": metrics, "elapsed_ms": 1}], minimum_complete_images=1)
    assert summary["image_count"] == 1 and summary["correct_approved_rate"] == 0
    assert summary["target_met"] is False


def test_unknown_is_not_an_approval_and_top3_is_scored():
    value = item()
    value.update(
        status="UNKNOWN",
        prediction=None,
        reason_codes=["BELOW_APPROVAL_THRESHOLD"],
        top3=[{"class_id": "bread_01", "class_name": "Bread", "confidence": 0.5}],
    )
    metrics = score_response(response([value]), GT)
    assert not metrics["complete_image"] and metrics["wrong_approved_count"] == 0
    assert metrics["unknown_top3_miss_count"] == 0


def test_robust_selection_uses_worst_then_median():
    assert robust_rank([(0, -49), (0, -48), (0, -47)], "a") < robust_rank(
        [(0, -50), (1, -50), (0, -50)], "b"
    )
    with pytest.raises(ValueError, match="three"):
        robust_rank([(0,)], "a")


def test_matching_tie_breaks_on_total_iou_without_class_information(monkeypatch):
    monkeypatch.setattr(
        "bixolon_scanner.evaluation.three_bakery.box_iou_matrix",
        lambda a, b: np.array([[0.51, 0.9], [0.8, 0.6]]),
    )
    assert match_boxes([[0] * 4] * 2, [[0] * 4] * 2, 0.5) == {0: 1, 1: 0}


def test_unknown_extra_segmentation_prevents_complete_image():
    extra = item(2, x=20)
    extra.update(
        status="UNKNOWN",
        prediction=None,
        reason_codes=["BELOW_APPROVAL_THRESHOLD"],
        top3=[{"class_id": "bread_01", "class_name": "Bread", "confidence": 0.5}],
    )
    scored = score_response(response([item(), extra]), GT)
    assert not scored["complete_image"]
    assert scored["wrong_approved_count"] == 0 and scored["extra_count"] == 1


def test_every_evaluation_image_stays_in_300_image_denominator():
    correct = {"metrics": score_response(response([item()]), GT), "elapsed_ms": 1}
    failed = {"metrics": score_response(response(status="ERROR"), GT), "elapsed_ms": 1}
    report = summarize([correct] * 297 + [failed] * 3, minimum_complete_images=297)
    assert report["image_count"] == 300 and report["complete_image_rate"] == 0.99
    assert report["ground_truth_count"] == 300 and report["target_met"]
    assert report["image_status_counts"]["ERROR"] == 3


def test_training_coverage_counts_actual_original_inputs():
    from collections import Counter

    from bixolon_scanner.training.three_bakery_detector import verify_coverage

    verify_coverage(Counter({"positive": 4, "empty": 4}), {"positive", "empty"})
    with pytest.raises(ValueError, match="coverage"):
        verify_coverage(Counter({"positive": 4}), {"positive", "empty"})
    with pytest.raises(ValueError, match="coverage"):
        verify_coverage(Counter({"positive": 4, "empty": 0}), {"positive", "empty"})


@pytest.mark.parametrize("parents", [[], ["external"], ["allowed", "external"]])
def test_derivative_provenance_rejects_missing_and_external_sources(parents):
    from bixolon_scanner.training.three_bakery_preparation import validate_parents

    with pytest.raises(ValueError, match="provenance|external"):
        validate_parents([{"parent_sha256": parents}], {"allowed"})


def test_object_parent_must_belong_to_its_scene():
    from bixolon_scanner.training.three_bakery_preparation import validate_parents

    with pytest.raises(ValueError, match="object parent"):
        validate_parents(
            [{"parent_sha256": ["allowed"], "annotations": [{"source_sha256": "other"}]}],
            {"allowed", "other"},
        )


@pytest.mark.parametrize("box", [[-1, 0, 3, 3], [0, 0, 101, 3], [0, 5, 4, 5]])
def test_annotation_boxes_use_valid_original_coordinates(box):
    from bixolon_scanner.training.three_bakery_preparation import validate_annotation

    with pytest.raises(ValueError, match="coordinates"):
        validate_annotation(
            {
                "kind": "single",
                "width": 100,
                "height": 100,
                "category_id": 1,
                "annotations": [{"bbox_xyxy": box, "category_id": 1}],
            },
            20,
        )


def test_final_dataset_is_not_opened_without_frozen_candidate(tmp_path, monkeypatch):
    from bixolon_scanner.evaluation.three_bakery_http import final_records

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation ground truth was accessed before freeze")

    monkeypatch.setattr("bixolon_scanner.evaluation.three_bakery_http.load_json_config", forbidden)
    with pytest.raises(ValueError, match="freeze"):
        final_records({}, tmp_path / "missing-freeze.json", tmp_path)


def test_training_is_prohibited_after_final_benchmark_access(tmp_path):
    from bixolon_scanner.experiments.bread.three_bakery import prohibit_post_benchmark_training

    (tmp_path / "final").mkdir()
    (tmp_path / "final/benchmark-access.json").write_text("{}")
    with pytest.raises(ValueError, match="prohibited"):
        prohibit_post_benchmark_training(tmp_path)


def test_robust_median_quality_precedes_latency():
    stable = [(0, -48, 100), (0, -48, 100), (0, -47, 100)]
    variable = [(0, -49, 1), (0, -47, 1), (0, -47, 1)]
    assert robust_rank(stable, "stable") < robust_rank(variable, "variable")


def test_candidate_matrix_has_exactly_twelve_distinct_configurations():
    from bixolon_scanner.experiments.bread.three_bakery import candidates

    configurations = candidates()
    assert len(configurations) == len({c["id"] for c in configurations}) == 12
    assert {c["method"] for c in configurations} == {"frozen", "finetune", "margin"}


def test_empty_background_produces_positive_ssd_classification_loss():
    from types import SimpleNamespace

    import torch

    from bixolon_scanner.training.ssdlite_objectness_detector import (
        enable_empty_image_hard_negative_loss,
    )

    model = SimpleNamespace(
        neg_to_pos_ratio=3, box_coder=SimpleNamespace(encode_single=lambda boxes, anchors: boxes)
    )
    enable_empty_image_hard_negative_loss(model, minimum_negatives=2)
    logits = torch.zeros(1, 4, 2, requires_grad=True)
    losses = model.compute_loss(
        [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}],
        {"bbox_regression": torch.zeros(1, 4, 4), "cls_logits": logits},
        [torch.zeros(4, 4)],
        [torch.full((4,), -1, dtype=torch.long)],
    )
    assert losses["classification"] > 0 and losses["bbox_regression"] == 0
    losses["classification"].backward()
    assert logits.grad.abs().sum() > 0


def test_runtime_rejects_training_with_superseded_annotations(tmp_path):
    from bixolon_scanner.contracts.catalog import sha256_file
    from bixolon_scanner.operations.three_bakery_runtime import verify_training
    from bixolon_scanner.training.three_bakery_data import write_json

    source = {"source_manifest_sha256": "source", "config_sha256": "config"}
    write_json(
        tmp_path / "contract.json",
        {**source, "annotation_sha256": "old", "settings": {"epochs": 8}},
    )
    write_json(
        tmp_path / "report.json",
        {"contract_sha256": sha256_file(tmp_path / "contract.json"), "epochs": 8},
    )
    with pytest.raises(ValueError, match="annotation contract"):
        verify_training(tmp_path, source, "corrected")


def test_failed_benchmark_parse_still_prohibits_further_training(tmp_path):
    from bixolon_scanner.evaluation.three_bakery_http import final_records
    from bixolon_scanner.experiments.bread.three_bakery import prohibit_post_benchmark_training
    from bixolon_scanner.training.three_bakery_data import write_json

    write_json(tmp_path / "freeze.json", {"candidate": "fixed"})
    write_json(tmp_path / "ground-truth.json", {"images": [], "annotations": []})
    config = {
        "evaluation": {
            "dataset_root": str(tmp_path),
            "annotations": "ground-truth.json",
            "expected_images": 300,
            "expected_objects": 1410,
        }
    }
    with pytest.raises(ValueError, match="count mismatch"):
        final_records(config, tmp_path / "freeze.json", tmp_path / "final")
    with pytest.raises(ValueError, match="prohibited"):
        prohibit_post_benchmark_training(tmp_path)


def test_same_version_catalog_from_another_trained_feature_space_is_rejected():
    from types import SimpleNamespace

    from bixolon_scanner.runtime.catalog import OnnxCatalogClassifier

    runtime = SimpleNamespace(
        metadata=SimpleNamespace(
            embedder=SimpleNamespace(embedder_id="dinov3-convnext-tiny-model-a", version="0.1.16"),
            classifier_policy=SimpleNamespace(version="0.1.16"),
        )
    )
    catalog = SimpleNamespace(
        metadata=SimpleNamespace(
            embedder_id="dinov3-convnext-tiny-model-b",
            embedder_version="0.1.16",
            classifier_policy_version="0.1.16",
        )
    )
    with pytest.raises(ValueError, match="not compatible"):
        OnnxCatalogClassifier(runtime, catalog, object())
