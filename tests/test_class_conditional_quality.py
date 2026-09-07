from __future__ import annotations

import pytest

from bixolon_scanner.training.class_conditional_quality import (
    build_class_conditional_quality_samples,
    joint_partial_quality_label,
)


def _records() -> list[dict]:
    rows = []
    for image_id in (1, 10):
        annotations = []
        for category_id in range(1, 21):
            annotations.extend(
                [
                    {
                        "category_id": category_id,
                        "bbox_xywh": [0, 0, 10, 10],
                        "visible_fraction": 0.95,
                    },
                    {
                        "category_id": category_id,
                        "bbox_xywh": [0, 0, 5, 10],
                        "visible_fraction": 0.4,
                    },
                ]
            )
        rows.append(
            {
                "image_id": image_id,
                "source_dataset": "bix_bakery_dataset",
                "annotations": annotations,
            }
        )
    return rows


@pytest.mark.parametrize(("split", "image_id"), [("train", 1), ("validation", 10)])
def test_quality_samples_are_balanced_and_group_split(split: str, image_id: int) -> None:
    samples = build_class_conditional_quality_samples(
        _records(),
        split=split,
        complete_minimum=0.85,
        incomplete_maximum=0.6,
        maximum_per_class_label=2,
        seed=7,
    )

    assert len(samples) == 40
    assert {sample.image_id for sample in samples} == {image_id}
    assert {(sample.class_index, sample.is_complete) for sample in samples} == {
        (class_index, label) for class_index in range(20) for label in (False, True)
    }


def test_quality_samples_reject_external_training_records() -> None:
    records = _records()
    records[0]["source_dataset"] = "bread_dataset"

    with pytest.raises(ValueError, match="outside the locked source dataset"):
        build_class_conditional_quality_samples(
            records,
            split="train",
            complete_minimum=0.85,
            incomplete_maximum=0.6,
            maximum_per_class_label=2,
            seed=7,
        )


def test_quality_samples_balance_complete_and_partial_per_class() -> None:
    records = _records()
    records[0]["annotations"].extend(
        {
            "category_id": 1,
            "bbox_xywh": [0, 0, 10, 10],
            "visible_fraction": 0.95,
        }
        for _ in range(5)
    )

    samples = build_class_conditional_quality_samples(
        records,
        split="train",
        complete_minimum=0.85,
        incomplete_maximum=0.6,
        maximum_per_class_label=10,
        seed=7,
    )

    class_one = [sample for sample in samples if sample.class_index == 0]
    assert sum(sample.is_complete for sample in class_one) == 1
    assert sum(not sample.is_complete for sample in class_one) == 1


def test_joint_partial_quality_label_maps_only_visibility_extremes() -> None:
    options = {
        "partial_quality_threshold": 0.45,
        "identity_minimum_visible_fraction": 0.75,
    }

    assert joint_partial_quality_label(6, 0.4, **options) == 20
    assert joint_partial_quality_label(6, 0.6, **options) is None
    assert joint_partial_quality_label(6, 0.9, **options) == 5
