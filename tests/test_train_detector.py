from __future__ import annotations

import json
import math
from argparse import Namespace

import pytest

from bixolon_scanner.training.train_detector import (
    detector_optimizer_parameter_groups,
    detector_optimizer_recipe,
    evaluate_detector_validation_predictions,
    initialize_detector_classification_head_biases,
    postprocess_detector_validation_batch,
    select_detector_metric_candidate,
    validate_detector_progress_identity,
    validate_detector_run_identity,
)


def _args(**overrides):
    values = {
        "learning_rate": 1e-5,
        "head_lr_multiplier": 1.0,
        "class_head_prior_probability": 0.5,
        "warmup_epochs": 0,
        "weight_decay": 1e-4,
        "epochs": 100,
        "min_score_threshold": 0.05,
        "max_score_threshold": 0.95,
        "threshold_steps": 91,
        "nms_iou_threshold": 0.7,
        "match_iou_threshold": 0.5,
        "target_recall": 0.99,
        "max_queries": 300,
    }
    values.update(overrides)
    return Namespace(**values)


def _tiny_detector(torch):
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 4)
            self.enc_score_head = torch.nn.Linear(4, 1)
            self.denoising_class_embed = torch.nn.Embedding(2, 4)

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
            self.class_embed = torch.nn.ModuleList(
                [torch.nn.Linear(4, 1), torch.nn.Linear(4, 1)]
            )

    return Detector()


def test_one_class_encoder_and_decoder_biases_use_low_foreground_prior():
    torch = pytest.importorskip("torch")
    model = _tiny_detector(torch)
    denoising_before = model.model.denoising_class_embed.weight.detach().clone()

    actual = initialize_detector_classification_head_biases(model, 0.01)

    expected = math.log(0.01 / 0.99)
    assert actual == pytest.approx(expected)
    assert model.model.enc_score_head.bias.tolist() == pytest.approx([expected])
    for head in model.class_embed:
        assert head.bias.tolist() == pytest.approx([expected])
    assert torch.equal(model.model.denoising_class_embed.weight, denoising_before)


def test_optimizer_groups_are_disjoint_complete_and_apply_configured_learning_rates():
    torch = pytest.importorskip("torch")
    model = _tiny_detector(torch)
    recipe = detector_optimizer_recipe(_args())

    groups = detector_optimizer_parameter_groups(model, recipe)

    assert [group["group_name"] for group in groups] == [
        "base",
        "randomly_reinitialized_modules",
    ]
    assert [group["lr"] for group in groups] == pytest.approx([1e-5, 1e-5])
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    expected_head_ids = {
        id(parameter)
        for head in [
            model.model.enc_score_head,
            *model.class_embed,
            model.model.denoising_class_embed,
        ]
        for parameter in head.parameters()
    }
    assert {id(parameter) for parameter in groups[1]["params"]} == expected_head_ids


def test_optimizer_recipe_records_all_resume_sensitive_values():
    recipe = detector_optimizer_recipe(_args())

    assert recipe == {
        "schema_version": 4,
        "optimizer": "AdamW",
        "base_learning_rate": 1e-5,
        "randomly_reinitialized_learning_rate": 1e-5,
        "head_lr_multiplier": 1.0,
        "weight_decay": 1e-4,
        "class_head_prior_probability": 0.5,
        "class_head_bias": pytest.approx(0.0),
        "scheduler": "cosine",
        "warmup_epochs": 0,
        "total_epochs": 100,
        "randomly_reinitialized_modules": [
            "model.enc_score_head",
            "class_embed.*",
            "model.denoising_class_embed",
        ],
        "checkpoint_selection": {
            "policy": "target_recall_then_count_precision_threshold",
            "min_score_threshold": 0.05,
            "max_score_threshold": 0.95,
            "threshold_steps": 91,
            "nms_iou_threshold": 0.7,
            "match_iou_threshold": 0.5,
            "target_recall": 0.99,
            "max_queries": 300,
        },
    }


def test_resume_rejects_any_optimizer_recipe_change():
    args = _args()
    recipe = detector_optimizer_recipe(args)
    progress = {
        "total_epochs": 100,
        "fold": 2,
        "final_training": False,
        "optimizer_recipe": recipe,
    }
    identity = Namespace(**vars(args), fold=2, final_training=False)
    validate_detector_progress_identity(progress, identity, recipe)

    changed = detector_optimizer_recipe(_args(head_lr_multiplier=20.0))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_detector_progress_identity(progress, identity, changed)

    changed_metrics = detector_optimizer_recipe(_args(nms_iou_threshold=0.6))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_detector_progress_identity(progress, identity, changed_metrics)


def test_resume_run_provenance_requires_exact_optimizer_recipe(tmp_path):
    recipe = detector_optimizer_recipe(_args())
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({"optimizer_recipe": recipe}), encoding="utf-8")
    validate_detector_run_identity(run_path, recipe)

    changed = detector_optimizer_recipe(_args(class_head_prior_probability=0.02))
    with pytest.raises(ValueError, match="optimizer recipe mismatch"):
        validate_detector_run_identity(run_path, changed)


def test_metric_candidate_selection_prefers_recall_gate_then_count_precision_threshold():
    candidates = [
        {"recall": 0.999, "count_accuracy": 0.4, "precision": 0.7, "score_threshold": 0.1},
        {"recall": 0.991, "count_accuracy": 0.6, "precision": 0.6, "score_threshold": 0.2},
        {"recall": 0.989, "count_accuracy": 1.0, "precision": 1.0, "score_threshold": 0.3},
        {"recall": 0.991, "count_accuracy": 0.6, "precision": 0.8, "score_threshold": 0.4},
        {"recall": 0.991, "count_accuracy": 0.6, "precision": 0.8, "score_threshold": 0.5},
    ]

    selected, key = select_detector_metric_candidate(candidates, 0.99)

    assert selected["score_threshold"] == 0.5
    assert key == (1.0, 0.6, 0.8, 0.5)

    selected, key = select_detector_metric_candidate(candidates, 1.0)
    assert selected["recall"] == 0.999
    assert key == (0.0, 0.999, 0.4, 0.7)


def test_validation_metrics_use_zero_threshold_predictions_and_select_grid_candidate():
    recipe = detector_optimizer_recipe(
        _args(min_score_threshold=0.05, max_score_threshold=0.5, threshold_steps=2)
    )
    records = [
        {
            "image_id": 10,
            "annotations": [{"bbox_xywh": [0.0, 0.0, 10.0, 10.0]}],
        }
    ]
    predictions = [
        {
            "image_id": 10,
            "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
            "scores": [0.9, 0.4],
        }
    ]

    result = evaluate_detector_validation_predictions(records, predictions, recipe)

    assert result["target_recall_satisfied"] is True
    assert result["selected_score_threshold"] == pytest.approx(0.5)
    assert result["detector_metrics"]["recall"] == 1.0
    assert result["detector_metrics"]["precision"] == 1.0
    assert result["detector_metrics"]["count_accuracy"] == 1.0
    with pytest.raises(ValueError, match="order is not aligned"):
        evaluate_detector_validation_predictions(
            records, [{**predictions[0], "image_id": 11}], recipe
        )


def test_validation_batch_postprocessing_preserves_record_order_and_uses_zero_threshold():
    torch = pytest.importorskip("torch")

    class Processor:
        def __init__(self):
            self.threshold = None
            self.target_sizes = None

        def post_process_object_detection(self, outputs, *, threshold, target_sizes):
            self.threshold = threshold
            self.target_sizes = target_sizes.tolist()
            return [
                {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]), "scores": torch.tensor([0.2])},
                {"boxes": torch.tensor([[5.0, 6.0, 7.0, 8.0]]), "scores": torch.tensor([0.8])},
            ]

    processor = Processor()
    records = [
        {"image_id": 2, "height": 20, "width": 30},
        {"image_id": 1, "height": 40, "width": 50},
    ]
    predictions = postprocess_detector_validation_batch(
        processor, object(), records, torch=torch, device=torch.device("cpu")
    )

    assert processor.threshold == 0.0
    assert processor.target_sizes == [[20, 30], [40, 50]]
    assert [row["image_id"] for row in predictions] == [2, 1]
    assert predictions[0]["scores"] == pytest.approx([0.2])


def test_validation_resume_requires_metric_history_and_best_quality_key():
    args = Namespace(**vars(_args()), fold=1, final_training=False)
    recipe = detector_optimizer_recipe(args)
    progress = {
        "total_epochs": 100,
        "fold": 1,
        "final_training": False,
        "completed_epochs": 1,
        "optimizer_recipe": recipe,
        "history": [{"epoch": 1, "validation_loss": 1.0}],
        "best_detector_quality_key": None,
    }

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_detector_progress_identity(progress, args, recipe)

    metric_record = {
        "epoch": 1,
        "validation_loss": 1.0,
        "detector_metrics": {"recall": 0.99, "precision": 0.7, "count_accuracy": 0.5},
        "selected_score_threshold": 0.2,
        "target_recall_satisfied": True,
        "detector_quality_key": [1.0, 0.5, 0.7, 0.2],
    }
    progress.update(
        {
            "history": [metric_record],
            "best_detector_quality_key": metric_record["detector_quality_key"],
            "last_detector_metrics": metric_record["detector_metrics"],
            "last_selected_score_threshold": 0.2,
            "last_target_recall_satisfied": True,
        }
    )
    validate_detector_progress_identity(progress, args, recipe)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"class_head_prior_probability": 0.0}, "prior"),
        ({"head_lr_multiplier": 0.0}, "multiplier"),
        ({"warmup_epochs": 100}, "warmup"),
    ],
)
def test_invalid_optimizer_recipe_is_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        detector_optimizer_recipe(_args(**override))
