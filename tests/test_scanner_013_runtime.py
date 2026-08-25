from bixolon_scanner.contracts.runtime_package_v2 import RuntimePackageV2Metadata
from bixolon_scanner.operations.scanner_013_runtime import candidate_metadata


def test_candidate_metadata_makes_detector_product_independent_and_disables_corroboration():
    base = RuntimePackageV2Metadata.model_validate(
        {
            "worker_version": "0.1.2",
            "dataset_version": "bread-old",
            "detector_policy_version": "0.1.2",
            "detector_class_count": 20,
            "detector": {
                "filename": "old.onnx",
                "version": "0.1.2",
                "score_threshold": 0.3,
                "nms_iou_threshold": 0.5,
                "max_queries": 300,
            },
            "count_verifier": {
                "filename": "count.onnx",
                "version": "0.1.2",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "count_labels": [0, 1],
                "comparison_mode": "object_presence",
                "confidence_threshold": 0.5,
            },
            "embedder": {
                "filename": "embedder.onnx",
                "embedder_id": "embedder",
                "version": "0.1.2",
                "embedding_dimension": 4,
            },
            "metric_projection": {"input_dimension": 4, "output_dimension": 4},
            "classifier_policy": {
                "version": "0.1.2",
                "prototype_weight": 0.5,
                "support_top_k": 1,
                "approval_minimum_similarity": 0.8,
                "approval_minimum_margin": 0.1,
                "ood_maximum_similarity": 0.5,
                "top3_minimum_similarity": 0.0,
                "catalog_conflict_similarity": 0.9,
                "ridge_approval_minimum_margin": 0.1,
                "ridge_top3_minimum_inverse_entropy": -3.0,
                "detector_corroboration_minimum_score": 0.9,
                "detector_corroboration_maximum_approval_score": 0.2,
            },
            "quality": {},
            "checksums": {
                "old.onnx": "a" * 64,
                "count.onnx": "b" * 64,
                "embedder.onnx": "c" * 64,
            },
            "licenses": {},
            "sources": {"embedder": {"architecture": "test embedder"}},
        }
    )

    candidate = candidate_metadata(
        base,
        version="0.1.3",
        dataset_version="bread-new",
        detector_sha256="d" * 64,
        detector_checkpoint_name="best.pt",
        detector_checkpoint_sha256="e" * 64,
        detector_manifest_sha256="f" * 64,
        license_checksums={"embedder.onnx": "c" * 64},
        horizontal_flip_tta=True,
        rotation_180_tta=True,
        crop_mode="square_context",
    )

    assert candidate.detector_class_count == 1
    assert candidate.detector.score_threshold == 0.65
    assert candidate.detector.nms_iou_threshold == 0.4
    assert candidate.detector.nms_containment_threshold == 0.8
    assert candidate.count_verifier is None
    assert candidate.embedder.neighbor_distance_bias == 0.0
    assert candidate.embedder.horizontal_flip_tta is True
    assert candidate.embedder.rotation_180_tta is True
    assert candidate.embedder.crop_margin_ratio == 0.0
    assert candidate.embedder.crop_mode == "square_context"
    assert candidate.classifier_policy.support_augmentation.views_per_source == 0
    assert candidate.classifier_policy.ridge_alpha == 0.01
    assert candidate.embedder.neighbor_mask is True
    assert candidate.classifier_policy.detector_corroboration_minimum_score is None
    assert candidate.classifier_policy.detector_corroboration_maximum_approval_score is None
    assert set(candidate.sources) == {"detector", "embedder"}
