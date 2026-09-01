from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from ...training.data import read_manifest
from .dinov3_class_conditional_ranker_oof import pair_features
from .dinov3_relative_class_ranker_oof import relative_pair_features
from .proposal_ranker import proposal_iou_matrix


def selected_segment_features(
    relative_features: np.ndarray,
    boxes: np.ndarray,
    candidates: np.ndarray,
    relative_scores: dict[str, np.ndarray],
    *,
    proposal_index: int,
    class_index: int,
    selected_box: np.ndarray,
    patch_confidence: float,
    cls_confidence: float,
    count_agreement: bool,
) -> np.ndarray:
    positions = np.flatnonzero(candidates[proposal_index] == class_index)
    if not len(positions):
        raise ValueError("selected class is absent from proposal candidates")
    rank = int(positions[0])
    anchor = boxes[proposal_index]
    anchor_overlap = proposal_iou_matrix(anchor[None], selected_box[None])[0, 0]
    width = max(float(anchor[2] - anchor[0]), 1e-6)
    height = max(float(anchor[3] - anchor[1]), 1e-6)
    selected_width = max(float(selected_box[2] - selected_box[0]), 1e-6)
    selected_height = max(float(selected_box[3] - selected_box[1]), 1e-6)
    score_values = np.asarray(
        [
            relative_scores["class_objectness"][proposal_index, rank],
            relative_scores["class_positive"][proposal_index, rank],
            relative_scores["class_iou"][proposal_index, rank],
        ],
        dtype=np.float32,
    )
    geometry = np.asarray(
        [
            anchor_overlap,
            ((selected_box[0] + selected_box[2]) - (anchor[0] + anchor[2])) / (2.0 * width),
            ((selected_box[1] + selected_box[3]) - (anchor[1] + anchor[3])) / (2.0 * height),
            np.log(selected_width / width),
            np.log(selected_height / height),
            patch_confidence,
            cls_confidence,
            float(count_agreement),
        ],
        dtype=np.float32,
    )
    return np.concatenate((relative_features[proposal_index, rank], score_values, geometry))


def _positive_score(model: object, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    position = np.flatnonzero(classes == 1)
    if not len(position):
        return np.zeros(len(features), dtype=np.float32)
    return model.predict_proba(features)[:, int(position[0])].astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    records = {
        int(row["image_id"]): row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and not row.get("exclude_from_detector_training", False)
    }
    scenes = [
        json.loads(line) for line in args.scenes.read_text(encoding="utf-8").splitlines() if line
    ]
    patch_counts = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.patch_count_oof.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    cls_counts = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.cls_count_oof.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    with np.load(args.dense_features, allow_pickle=False) as payload:
        geometry = payload["geometry"].astype(np.float32)
        image_ids = payload["image_id"].copy()
        boxes = payload["boxes_xyxy"].astype(np.float32)
    with np.load(args.catalog_logits, allow_pickle=False) as payload:
        catalog = {name: payload[name].copy() for name in payload.files}
    with np.load(args.proposal_ranker_scores, allow_pickle=False) as payload:
        proposal_ranker = {name: payload[name].copy() for name in payload.files}
    with np.load(args.base_class_scores, allow_pickle=False) as payload:
        base_scores = {name: payload[name].copy() for name in payload.files}
    with np.load(args.relative_class_scores, allow_pickle=False) as payload:
        relative_scores = {name: payload[name].copy() for name in payload.files}
    candidates = relative_scores["candidate_classes"].astype(np.int64)
    for name, payload in (
        ("Catalog", catalog),
        ("proposal ranker", proposal_ranker),
        ("base class scores", base_scores),
        ("relative class scores", relative_scores),
    ):
        if not np.array_equal(payload["image_id"], image_ids):
            raise ValueError(f"{name} cache does not align with dense features")
    proposal_scores = np.column_stack(
        [
            proposal_ranker[name]
            for name in (
                "dense_objectness",
                "compact_objectness",
                "compact_assignment",
                "compact_iou",
            )
        ]
    ).astype(np.float32)
    base_features = pair_features(
        geometry,
        proposal_scores,
        catalog["logits"],
        catalog["retrieval_logits"],
        candidates,
        catalog["approval_scores"],
        catalog["approval_blocked"],
        catalog["segment_recapture"],
    )
    relative_features = relative_pair_features(
        base_features,
        boxes,
        image_ids,
        candidates,
        base_scores["class_objectness"],
        base_scores["class_iou"],
    )

    feature_rows = []
    label_rows = []
    fold_rows = []
    scene_segment_offsets = []
    for scene_index, scene in enumerate(scenes):
        start = len(feature_rows)
        if scene.get("selectable", False):
            image_id = int(scene["image_id"])
            image_mask = image_ids == image_id
            global_indices = np.flatnonzero(image_mask)
            image_boxes = boxes[image_mask]
            image_candidates = candidates[image_mask]
            image_features = relative_features[image_mask]
            image_scores = {
                name: values[image_mask]
                for name, values in relative_scores.items()
                if name.startswith("class_")
            }
            target_by_class = {
                int(annotation["category_id"]) - 1: np.asarray(
                    [
                        annotation["bbox_xywh"][0],
                        annotation["bbox_xywh"][1],
                        annotation["bbox_xywh"][0] + annotation["bbox_xywh"][2],
                        annotation["bbox_xywh"][1] + annotation["bbox_xywh"][3],
                    ],
                    dtype=np.float32,
                )
                for annotation in records[image_id]["annotations"]
            }
            del global_indices
            for proposal_index, class_index, selected_box in zip(
                scene["selected_indices"],
                scene["selected_classes"],
                scene["selected_boxes_xyxy"],
                strict=True,
            ):
                feature_rows.append(
                    selected_segment_features(
                        image_features,
                        image_boxes,
                        image_candidates,
                        image_scores,
                        proposal_index=int(proposal_index),
                        class_index=int(class_index),
                        selected_box=np.asarray(selected_box, dtype=np.float32),
                        patch_confidence=float(patch_counts[image_id]["confidence"]),
                        cls_confidence=float(cls_counts[image_id]["confidence"]),
                        count_agreement=(
                            patch_counts[image_id]["predicted_count"]
                            == cls_counts[image_id]["predicted_count"]
                        ),
                    )
                )
                target = target_by_class.get(int(class_index))
                overlap = (
                    0.0
                    if target is None
                    else float(
                        proposal_iou_matrix(
                            np.asarray(selected_box, dtype=np.float32)[None], target[None]
                        )[0, 0]
                    )
                )
                label_rows.append(overlap >= 0.5)
                fold_rows.append(int(scene["fold"]))
        scene_segment_offsets.append((scene_index, start, len(feature_rows)))

    features = np.stack(feature_rows)
    labels = np.asarray(label_rows, dtype=bool)
    segment_folds = np.asarray(fold_rows, dtype=np.int8)
    predictions = {
        "segment_safety_extra_trees": np.zeros(len(features), dtype=np.float32),
        "segment_safety_hist_gradient": np.zeros(len(features), dtype=np.float32),
    }
    diagnostics = []
    for held_fold in sorted(np.unique(segment_folds)):
        training = segment_folds != held_fold
        held = segment_folds == held_fold
        models = {
            "segment_safety_extra_trees": ExtraTreesClassifier(
                n_estimators=600,
                min_samples_leaf=3,
                max_features=0.75,
                class_weight="balanced",
                random_state=args.seed + int(held_fold),
                n_jobs=-1,
            ),
            "segment_safety_hist_gradient": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=15,
                min_samples_leaf=12,
                l2_regularization=0.5,
                class_weight="balanced",
                random_state=args.seed + 10 + int(held_fold),
            ),
        }
        fold_diagnostic = {"held_out_fold": int(held_fold)}
        for name, model in models.items():
            model.fit(features[training], labels[training])
            predictions[name][held] = _positive_score(model, features[held])
            fold_diagnostic[f"{name}_roc_auc"] = float(
                roc_auc_score(labels[held], predictions[name][held])
            )
        diagnostics.append(fold_diagnostic)

    output_scenes = [dict(scene) for scene in scenes]
    for scene_index, start, end in scene_segment_offsets:
        for name, values in predictions.items():
            output_scenes[scene_index][name] = values[start:end].tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in output_scenes),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_selected_segment_verifier_true_group_oof",
        "selection_scope": "every selected segment is scored by models trained on other folds",
        "segment_count": len(features),
        "safe_segment_count": int(labels.sum()),
        "feature_dimension": int(features.shape[1]),
        "folds": diagnostics,
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit a selected DINOv3 segment safety verifier"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--proposal-ranker-scores", type=Path, required=True)
    parser.add_argument("--base-class-scores", type=Path, required=True)
    parser.add_argument("--relative-class-scores", type=Path, required=True)
    parser.add_argument("--patch-count-oof", type=Path, required=True)
    parser.add_argument("--cls-count-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
