import json
from pathlib import Path

import pytest

from bixolon_scanner.contracts.model_package import ModelSource
from bixolon_scanner.experiments.bread.build_scanner_v2 import (
    _embedder_source,
    build_runtime_package,
)


def _source(architecture: str) -> ModelSource:
    return ModelSource(architecture=architecture)


def test_embedder_source_accepts_current_runtime_source_name() -> None:
    expected = _source("current")

    assert _embedder_source({"embedder": expected}) is expected


def test_embedder_source_keeps_legacy_classifier_compatibility() -> None:
    expected = _source("legacy")

    assert _embedder_source({"classifier": expected}) is expected


def test_runtime_builder_packages_single_detector_without_lifecycle_metadata(
    tmp_path: Path,
) -> None:
    source = _ensemble_source_payload()
    source["detector"]["score_threshold"] = 0.31
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    detector_source_path = tmp_path / "detector-source.json"
    detector_source_path.write_text(
        json.dumps(
            {
                "architecture": "D-FINE-N",
                "training_dataset_version": "bread-multi-object-only",
                "training_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    detector = tmp_path / "trained-on-multi-object-scenes.onnx"
    embedder = tmp_path / "embedder.onnx"
    detector.write_bytes(b"single-detector")
    embedder.write_bytes(b"embedder")

    payload = build_runtime_package(
        source_path,
        detector,
        None,
        embedder,
        tmp_path / "runtime",
        version="0.1.2",
        detector_score_threshold=0.17,
        detector_source_path=detector_source_path,
    )

    assert payload["detector"]["filename"] == "detector.onnx"
    assert payload["detector"]["ensemble"] is None
    assert payload["detector"]["score_threshold"] == 0.17
    assert payload["sources"]["detector"]["architecture"] == "single D-FINE detector"
    assert payload["sources"]["detector"]["training_dataset_version"] == "bread-multi-object-only"
    assert "promotion_status" not in payload


def _ensemble_source_payload() -> dict:
    filenames = [f"detector-{name}.onnx" for name in ("fold0", "fold1", "fold2", "production")]
    return {
        "detector_class_count": 2,
        "detector": {
            "filename": filenames[0],
            "version": "0.0.2",
            "score_threshold": 0.5,
            "nms_iou_threshold": 0.5,
            "max_queries": 100,
            "ensemble": {
                "members": [
                    {"filename": filename, "score_threshold": 0.2} for filename in filenames
                ],
                "fusion": {"cluster_iou_threshold": 0.5},
                "base_selection": {
                    "score_threshold": 0.2,
                    "nms_iou_threshold": 0.5,
                    "containment_threshold": 0.9,
                    "group_minimum": 2,
                },
                "policy_consensus": {
                    "policies": [
                        {
                            "member_filename": filename,
                            "score_threshold": 0.2,
                            "nms_iou_threshold": 0.5,
                            "containment_threshold": 0.9,
                            "group_minimum": 2,
                        }
                        for filename in filenames
                    ],
                    "agreement_iou_threshold": 0.5,
                    "minimum_agreeing_policy_count": 4,
                },
                "draft_refinement": {
                    "draft_size": 1000,
                    "ambiguity_refinement_maximum_selected_count": 6,
                    "maximum_agreeing_policy_count": 4,
                    "minimum_selected_count": 1,
                    "minimum_selected_box_aspect_ratio_extremity": 1.5,
                    "consensus_ambiguity_bypass_maximum_selected_count": 2,
                    "consensus_ambiguity_bypass_minimum_selected_score": 0.5,
                    "unanimous_ambiguity_bypass_maximum_selected_count": 2,
                    "unanimous_ambiguity_bypass_minimum_selected_score": 0.5,
                    "full_resolution_unresolved_ambiguity_maximum_selected_count": 2,
                    "full_resolution_on_selected_count_change": True,
                },
                "ambiguity_union": [
                    {
                        "availability_score_threshold": 0.2,
                        "availability_nms_iou_threshold": 0.5,
                        "availability_containment_threshold": 0.9,
                        "availability_group_minimum": 0,
                        "minimum_selected_count": 1,
                        "extra_candidate_count": 1,
                        "extra_count_mode": "at_least",
                        "next_score_threshold_inclusive": 0.2,
                    }
                ],
                "class_verified_selector": {
                    "candidate_minimum_score": 0.2,
                    "candidate_minimum_support": 4,
                    "candidate_duplicate_iou": 0.5,
                    "base_match_iou": 0.5,
                    "group_relation_iou": 0.5,
                    "group_area_ratio": 0.5,
                    "group_margin_ratio": 0.1,
                    "group_novel_margin": 0.1,
                    "group_minimum_score": 0.2,
                    "independent_maximum_iou": 0.2,
                    "independent_margin": 0.1,
                    "independent_minimum_score": 0.2,
                },
                "maximum_box_area_ratio": 0.9,
            },
        },
        "sources": {
            "detector": {"architecture": "D-FINE ensemble"},
            "embedder": {"architecture": "DINOv3 ConvNeXt-Tiny"},
        },
        "licenses": {"detector": "Apache-2.0", "classifier": "DINOv3"},
        "quality": {},
    }


def test_runtime_builder_selects_and_recalibrates_three_detector_members(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_ensemble_source_payload()), encoding="utf-8")
    detector_paths = []
    for name in ("fold0", "fold1", "fold2", "production"):
        path = tmp_path / f"detector-{name}.onnx"
        path.write_bytes(name.encode())
        detector_paths.append(path)
    embedder_path = tmp_path / "embedder.onnx"
    embedder_path.write_bytes(b"embedder")

    payload = build_runtime_package(
        source_path,
        detector_paths[0],
        None,
        embedder_path,
        tmp_path / "runtime",
        version="0.1.1",
        detector_member_paths=[detector_paths[0], detector_paths[1], detector_paths[3]],
    )

    ensemble = payload["detector"]["ensemble"]
    expected = ["detector-fold0.onnx", "detector-fold1.onnx", "detector-production.onnx"]
    assert [member["filename"] for member in ensemble["members"]] == expected
    assert [
        policy["member_filename"] for policy in ensemble["policy_consensus"]["policies"]
    ] == expected
    assert ensemble["policy_consensus"]["minimum_agreeing_policy_count"] == 3
    assert ensemble["draft_refinement"]["maximum_agreeing_policy_count"] == 3
    assert ensemble["class_verified_selector"]["candidate_minimum_support"] == 3
    assert payload["detector_ambiguity"]["dense_agreement_count_minimum"] == 3


def test_runtime_builder_rejects_unknown_detector_member(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_ensemble_source_payload()), encoding="utf-8")
    known = tmp_path / "detector-fold0.onnx"
    unknown = tmp_path / "detector-unknown.onnx"
    embedder = tmp_path / "embedder.onnx"
    known.write_bytes(b"known")
    unknown.write_bytes(b"unknown")
    embedder.write_bytes(b"embedder")

    with pytest.raises(ValueError, match="at least two known members"):
        build_runtime_package(
            source_path,
            known,
            None,
            embedder,
            tmp_path / "runtime",
            version="0.1.1",
            detector_member_paths=[known, unknown],
        )


def test_runtime_builder_configures_two_detector_selective_cascade(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_ensemble_source_payload()), encoding="utf-8")
    fold1 = tmp_path / "detector-fold1.onnx"
    production = tmp_path / "detector-production.onnx"
    embedder = tmp_path / "embedder.onnx"
    embedder_report = tmp_path / "embedder-report.json"
    fold1.write_bytes(b"fold1")
    production.write_bytes(b"production")
    embedder.write_bytes(b"embedder")
    embedder_report.write_text(
        json.dumps(
            {
                "backbone_kind": "dinov3_convnext_tiny",
                "embedding_dimension": 768,
                "image_size": 192,
                "l2_normalized": True,
                "embedding_space": "trained_adapter",
                "training_scope": "backbone.stages[-1]",
                "source_revision": "revision",
            }
        ),
        encoding="utf-8",
    )

    payload = build_runtime_package(
        source_path,
        fold1,
        None,
        embedder,
        tmp_path / "runtime",
        version="0.1.1",
        embedder_report_path=embedder_report,
        detector_member_paths=[fold1, production],
        cascade_primary_member_filename="detector-production.onnx",
        cascade_secondary_trigger_selected_counts=[6],
        cascade_secondary_trigger_minimum_score_maximum=0.5,
        cascade_secondary_trigger_minimum_score_minimum=0.87,
        cascade_secondary_trigger_on_uncertain=True,
    )

    assert payload["detector"]["ensemble"]["selective_cascade"] == {
        "primary_member_filename": "detector-production.onnx",
        "secondary_trigger_selected_counts": [6],
        "secondary_trigger_minimum_score_maximum": 0.5,
        "secondary_trigger_minimum_score_minimum": 0.87,
        "secondary_trigger_on_uncertain": True,
    }
    assert payload["embedder"]["input_size"] == [192, 192]
