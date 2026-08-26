from bixolon_scanner.contracts.runtime_package_v2 import RuntimePackageV2Metadata
from bixolon_scanner.operations.scanner_014_runtime import candidate_metadata


def test_candidate_metadata_adds_crowding_policy_and_aligns_versions() -> None:
    base = RuntimePackageV2Metadata.model_validate(
        {
            "worker_version": "0.1.3",
            "dataset_version": "bread-source",
            "detector_policy_version": "0.1.3",
            "detector_class_count": 1,
            "detector": {
                "filename": "detector.onnx",
                "version": "0.1.3",
                "score_threshold": 0.65,
                "nms_iou_threshold": 0.4,
                "max_queries": 300,
            },
            "embedder": {
                "filename": "embedder.onnx",
                "embedder_id": "embedder",
                "version": "0.1.3",
                "embedding_dimension": 4,
            },
            "metric_projection": {"input_dimension": 4, "output_dimension": 4},
            "classifier_policy": {
                "version": "0.1.3",
                "prototype_weight": 0.5,
                "support_top_k": 1,
                "approval_minimum_similarity": 0.8,
                "approval_minimum_margin": 0.1,
                "ood_maximum_similarity": 0.5,
                "top3_minimum_similarity": 0.0,
                "catalog_conflict_similarity": 0.9,
                "ridge_approval_minimum_margin": 0.1,
                "ridge_top3_minimum_inverse_entropy": -3.0,
            },
            "quality": {},
            "checksums": {
                "detector.onnx": "a" * 64,
                "embedder.onnx": "b" * 64,
            },
            "licenses": {},
        }
    )

    candidate = candidate_metadata(base)

    assert candidate.worker_version == "0.1.4"
    assert candidate.detector.version == "0.1.4"
    assert candidate.detector_policy_version == "0.1.4"
    assert candidate.embedder.version == "0.1.4"
    assert candidate.classifier_policy.version == "0.1.4"
    assert candidate.detector_crowding is not None
    assert candidate.detector_crowding.minimum_image_aspect_ratio == 1.0
    assert candidate.detector_crowding.large_proposal_minimum_area_ratio == 0.21
    assert candidate.detector_crowding.query_duplicate_minimum_fraction == 0.93
    assert candidate.detector_crowding.rotation_recovery_degrees == [90, 180]
    assert candidate.detector_crowding.rotation_recovery_minimum_selected_count == 5
    assert candidate.detector_crowding.rotation_recovery_maximum_selected_count == 5
    assert candidate.detector_crowding.rotation_recovery_minimum_selected_area_fraction == 0.4
    assert candidate.detector_crowding.rotation_recovery_minimum_normalized_center_distance == 0.63
    assert candidate.detector_crowding.rotation_recovery_minimum_count_gain == 1
    assert candidate.detector_crowding.rotation_recovery_agreement_iou_threshold == 0.5
