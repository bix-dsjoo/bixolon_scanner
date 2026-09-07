from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ClassConditionalQualitySample:
    image_id: int
    box_xyxy: tuple[float, float, float, float]
    class_index: int
    is_complete: bool
    visible_fraction: float


def joint_partial_quality_label(
    category_id: int,
    visible_fraction: float,
    *,
    partial_quality_threshold: float,
    identity_minimum_visible_fraction: float,
) -> int | None:
    if not 1 <= category_id <= 20:
        raise ValueError("joint quality category must be between 1 and 20")
    if not 0.0 <= partial_quality_threshold < identity_minimum_visible_fraction <= 1.0:
        raise ValueError("joint quality visibility thresholds are invalid")
    if visible_fraction <= partial_quality_threshold:
        return 20
    if visible_fraction >= identity_minimum_visible_fraction:
        return category_id - 1
    return None


def build_class_conditional_quality_samples(
    records: Iterable[dict[str, Any]],
    *,
    split: str,
    complete_minimum: float,
    incomplete_maximum: float,
    maximum_per_class_label: int,
    seed: int,
) -> list[ClassConditionalQualitySample]:
    """Build class-balanced complete/partial samples from source-only composites."""
    if split not in {"train", "validation"}:
        raise ValueError("quality split must be train or validation")
    if not 0.0 <= incomplete_maximum < complete_minimum <= 1.0:
        raise ValueError("quality visibility thresholds are invalid")
    if maximum_per_class_label < 1:
        raise ValueError("maximum_per_class_label must be positive")

    buckets: dict[tuple[int, bool], list[ClassConditionalQualitySample]] = {
        (class_index, is_complete): [] for class_index in range(20) for is_complete in (False, True)
    }
    for record in records:
        image_id = int(record["image_id"])
        record_split = "validation" if image_id % 10 == 0 else "train"
        if record_split != split:
            continue
        if record.get("training_allowed") is False:
            raise ValueError("training-prohibited image cannot train the quality verifier")
        source_is_locked = str(record.get("source_dataset")) == "bix_bakery_dataset"
        composite_is_locked = str(record.get("source")) == (
            "bix_bakery_dataset_source_only_composite"
        )
        if not (source_is_locked or composite_is_locked):
            raise ValueError("quality verifier record is outside the locked source dataset")
        for annotation in record.get("annotations", []):
            visible_fraction = float(annotation.get("visible_fraction", 1.0))
            if visible_fraction >= complete_minimum:
                is_complete = True
            elif visible_fraction <= incomplete_maximum:
                is_complete = False
            else:
                continue
            x, y, width, height = (float(value) for value in annotation["bbox_xywh"])
            class_index = int(annotation["category_id"]) - 1
            if class_index not in range(20) or width <= 0.0 or height <= 0.0:
                raise ValueError("quality annotation is invalid")
            buckets[(class_index, is_complete)].append(
                ClassConditionalQualitySample(
                    image_id=image_id,
                    box_xyxy=(x, y, x + width, y + height),
                    class_index=class_index,
                    is_complete=is_complete,
                    visible_fraction=visible_fraction,
                )
            )

    missing = [key for key, rows in buckets.items() if not rows]
    if missing:
        raise ValueError(f"quality manifest has empty class/label buckets: {missing}")
    generator = np.random.default_rng(seed + (0 if split == "train" else 1))
    selected = []
    for class_index in range(20):
        paired_count = min(
            maximum_per_class_label,
            len(buckets[(class_index, False)]),
            len(buckets[(class_index, True)]),
        )
        for is_complete in (False, True):
            rows = buckets[(class_index, is_complete)]
            indexes = generator.choice(len(rows), paired_count, replace=False)
            selected.extend(rows[int(index)] for index in indexes)
    generator.shuffle(selected)
    return selected


__all__ = [
    "ClassConditionalQualitySample",
    "build_class_conditional_quality_samples",
    "joint_partial_quality_label",
]
