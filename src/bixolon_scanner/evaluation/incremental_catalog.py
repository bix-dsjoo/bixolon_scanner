from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..contracts.catalog import load_store_catalog_package
from ..operations.catalog_activation import (
    fit_append_only_ridge_adapter,
    fit_ridge_adapter,
)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True).clip(min=1e-12)


def _append_only_branch(
    support_embeddings: np.ndarray,
    evaluation_embeddings: np.ndarray,
    support_labels: np.ndarray,
    *,
    held_out_class: int,
    class_count: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    old_classes = np.asarray(
        [class_index for class_index in range(class_count) if class_index != held_out_class],
        dtype=np.int64,
    )
    remapped = np.full(class_count, -1, dtype=np.int64)
    remapped[old_classes] = np.arange(class_count - 1, dtype=np.int64)
    remapped[held_out_class] = class_count - 1
    old_mask = support_labels != held_out_class
    base_weight, base_bias = fit_ridge_adapter(
        support_embeddings[old_mask],
        remapped[support_labels[old_mask]],
        alpha=alpha,
        class_count=class_count - 1,
    )
    extended_weight, extended_bias = fit_append_only_ridge_adapter(
        base_weight,
        base_bias,
        support_embeddings,
        remapped[support_labels],
        alpha=alpha,
        class_count=class_count,
    )
    base_logits = evaluation_embeddings @ base_weight + base_bias
    extended_logits = evaluation_embeddings @ extended_weight + extended_bias
    return old_classes[np.argmax(base_logits, axis=1)], np.argmax(extended_logits, axis=1)


def evaluate_append_only_consensus(
    support_branches: tuple[np.ndarray, np.ndarray, np.ndarray],
    evaluation_branches: tuple[np.ndarray, np.ndarray, np.ndarray],
    evaluation_targets: np.ndarray,
    *,
    class_count: int,
    supports_per_class: int,
    alpha: float,
) -> dict:
    """Simulate each known class as a newly appended SKU without target-label calibration."""
    targets = np.asarray(evaluation_targets, dtype=np.int64)
    expected_support_count = class_count * supports_per_class
    if class_count < 2 or supports_per_class < 1 or alpha <= 0:
        raise ValueError("incremental Catalog evaluation configuration is invalid")
    if targets.ndim != 1 or np.any(targets < 0) or np.any(targets >= class_count):
        raise ValueError("incremental Catalog evaluation targets are invalid")
    if any(len(branch) != expected_support_count for branch in support_branches):
        raise ValueError("incremental Catalog support branches have invalid row counts")
    if any(len(branch) != len(targets) for branch in evaluation_branches):
        raise ValueError("incremental Catalog evaluation branches are not aligned")

    supports = tuple(_l2_normalize(branch) for branch in support_branches)
    evaluations = tuple(_l2_normalize(branch) for branch in evaluation_branches)
    support_labels = np.repeat(np.arange(class_count, dtype=np.int64), supports_per_class)
    scenarios = []
    existing_decision_count = 0
    existing_output_change_count = 0
    existing_correct_to_incorrect_count = 0
    existing_incorrect_to_correct_count = 0
    new_sku_object_count = 0
    new_sku_selected_count = 0
    for held_out_class in range(class_count):
        branch_results = [
            _append_only_branch(
                support,
                evaluation,
                support_labels,
                held_out_class=held_out_class,
                class_count=class_count,
                alpha=alpha,
            )
            for support, evaluation in zip(supports, evaluations, strict=True)
        ]
        base_prediction = branch_results[0][0]
        appended_index = class_count - 1
        unanimous_new_sku = np.logical_and.reduce(
            [extended_prediction == appended_index for _, extended_prediction in branch_results]
        )
        final_prediction = np.where(unanimous_new_sku, held_out_class, base_prediction)
        existing_mask = targets != held_out_class
        new_sku_mask = ~existing_mask
        changed = existing_mask & (final_prediction != base_prediction)
        regressed = existing_mask & (base_prediction == targets) & (final_prediction != targets)
        improved = existing_mask & (base_prediction != targets) & (final_prediction == targets)
        selected = new_sku_mask & unanimous_new_sku
        scenario = {
            "held_out_class_index": held_out_class,
            "existing_decision_count": int(np.sum(existing_mask)),
            "existing_output_change_count": int(np.sum(changed)),
            "existing_correct_to_incorrect_count": int(np.sum(regressed)),
            "existing_incorrect_to_correct_count": int(np.sum(improved)),
            "new_sku_object_count": int(np.sum(new_sku_mask)),
            "new_sku_selected_count": int(np.sum(selected)),
        }
        scenarios.append(scenario)
        existing_decision_count += scenario["existing_decision_count"]
        existing_output_change_count += scenario["existing_output_change_count"]
        existing_correct_to_incorrect_count += scenario["existing_correct_to_incorrect_count"]
        existing_incorrect_to_correct_count += scenario["existing_incorrect_to_correct_count"]
        new_sku_object_count += scenario["new_sku_object_count"]
        new_sku_selected_count += scenario["new_sku_selected_count"]

    return {
        "class_holdout_scenario_count": class_count,
        "existing_decision_count": existing_decision_count,
        "existing_output_change_count": existing_output_change_count,
        "existing_correct_to_incorrect_count": existing_correct_to_incorrect_count,
        "existing_incorrect_to_correct_count": existing_incorrect_to_correct_count,
        "new_sku_object_count": new_sku_object_count,
        "new_sku_selected_count": new_sku_selected_count,
        "new_sku_selection_rate": new_sku_selected_count / new_sku_object_count,
        "existing_output_preserved": existing_output_change_count == 0,
        "existing_performance_regression_free": existing_correct_to_incorrect_count == 0,
        "scenarios": scenarios,
    }


def _trace_targets(
    trace_path: Path,
    image_ids: np.ndarray,
    detection_indices: np.ndarray,
    class_ids: list[str],
) -> np.ndarray:
    class_indices = {class_id: index for index, class_id in enumerate(class_ids)}
    lookup: dict[tuple[int, int], int] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for diagnostic in row.get("matched_classifier_diagnostics", []):
            key = (int(row["image_id"]), int(diagnostic["detection_index"]))
            lookup[key] = class_indices[str(diagnostic["target_class_id"])]
    keys = [
        (int(image_id), int(detection_index))
        for image_id, detection_index in zip(image_ids, detection_indices, strict=True)
    ]
    if len(lookup) != len(keys) or set(lookup) != set(keys):
        raise ValueError("trace targets and cached embeddings are not one-to-one")
    return np.asarray([lookup[key] for key in keys], dtype=np.int64)


def _load_embeddings(
    path: Path, *, mean_rotation: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if mean_rotation:
            embeddings = np.asarray(
                (payload["raw_embeddings_angle_0"] + payload["raw_embeddings_angle_180"])
                * np.float32(0.5),
                dtype=np.float32,
            )
        else:
            embeddings = np.asarray(payload["raw_embeddings"], dtype=np.float32)
        return (
            embeddings,
            np.asarray(payload["image_ids"], dtype=np.int64),
            np.asarray(payload["detection_indices"], dtype=np.int64),
        )


def evaluate_artifacts(
    catalog_root: Path,
    primary_embeddings_path: Path,
    rotation_embeddings_path: Path,
    independent_embeddings_path: Path,
    trace_path: Path,
    source_comparison_path: Path,
) -> dict:
    catalog = load_store_catalog_package(catalog_root)
    if catalog.rotation_catalog_root is None or catalog.independent_catalog_root is None:
        raise ValueError("incremental evaluation requires three consensus Catalogs")
    rotation_catalog = load_store_catalog_package(catalog.rotation_catalog_root)
    independent_catalog = load_store_catalog_package(catalog.independent_catalog_root)
    catalogs = (catalog, rotation_catalog, independent_catalog)
    class_ids = [label.class_id for label in catalog.metadata.labels]
    expected_ids = tuple(class_ids)
    if any(
        tuple(label.class_id for label in item.metadata.labels) != expected_ids for item in catalogs
    ):
        raise ValueError("incremental evaluation Catalog labels differ")
    primary, image_ids, detection_indices = _load_embeddings(primary_embeddings_path)
    rotation, rotation_image_ids, rotation_detection_indices = _load_embeddings(
        rotation_embeddings_path, mean_rotation=True
    )
    independent, independent_image_ids, independent_detection_indices = _load_embeddings(
        independent_embeddings_path
    )
    if not (
        np.array_equal(image_ids, rotation_image_ids)
        and np.array_equal(image_ids, independent_image_ids)
        and np.array_equal(detection_indices, rotation_detection_indices)
        and np.array_equal(detection_indices, independent_detection_indices)
    ):
        raise ValueError("incremental evaluation embedding caches differ in row order")
    targets = _trace_targets(trace_path, image_ids, detection_indices, class_ids)
    supports = tuple(np.load(item.supports_path, allow_pickle=False) for item in catalogs)
    statistics = json.loads(catalog.statistics_path.read_text(encoding="utf-8"))
    alpha = float(statistics["adapter_fit"]["alpha"])
    metrics = evaluate_append_only_consensus(
        supports,
        (primary, rotation, independent),
        targets,
        class_count=len(class_ids),
        supports_per_class=catalog.metadata.support_count_per_class,
        alpha=alpha,
    )
    for scenario, class_id in zip(metrics["scenarios"], class_ids, strict=True):
        scenario["held_out_class_id"] = class_id
    source_comparison = json.loads(source_comparison_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "evaluation": "append_only_three_way_consensus_new_sku",
        "product_version": "0.1.6",
        "method": {
            "base_adapter_columns": "bit_stable",
            "new_outputs": "independent_binary_ridge",
            "new_sku_acceptance": "unanimous_primary_rotation_independent_top1",
            "class_specific_thresholds": False,
            "decision_threshold_selected": False,
            "ridge_alpha": alpha,
        },
        "data_contract": {
            "support_source": source_comparison["selected_source"],
            "evaluation_source": "multi_object_and_operational_objects",
            "source_evaluation_exact_sha256_overlap_count": source_comparison[
                "source_evaluation_exact_sha256_overlap_count"
            ],
            "multi_object_product_labels_used_for_training": False,
            "multi_object_product_labels_used_for_calibration": False,
            "multi_object_product_labels_used_for_evaluation_only": True,
            "held_out_test_set": False,
        },
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate append-only Catalog new-SKU stability")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--primary-embeddings", type=Path, required=True)
    parser.add_argument("--rotation-embeddings", type=Path, required=True)
    parser.add_argument("--independent-embeddings", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_artifacts(
        args.catalog,
        args.primary_embeddings,
        args.rotation_embeddings,
        args.independent_embeddings,
        args.trace,
        args.source_comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
