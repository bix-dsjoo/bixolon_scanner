from types import SimpleNamespace

import numpy as np
import pytest

from bixolon_scanner.pipeline.ports import ClassificationResult, Detection
from bixolon_scanner.runtime.catalog import ConsensusCatalogClassifier


def _result(
    top1: int,
    *,
    approval_score: float,
    retrieval_top1: int | None = None,
    class_count: int = 3,
    segment_recapture_reason: str | None = None,
) -> ClassificationResult:
    logits = np.full((1, class_count), -1.0, dtype=np.float32)
    logits[0, top1] = 1.0
    retrieval = logits.copy()
    if retrieval_top1 is not None:
        retrieval.fill(-1.0)
        retrieval[0, retrieval_top1] = 1.0
    return ClassificationResult(
        logits=logits,
        ranking_logits=logits,
        retrieval_logits=retrieval,
        approval_scores=np.asarray([approval_score], dtype=np.float32),
        segment_recapture_reasons=(segment_recapture_reason,),
        unknown_reasons=(None,),
        approval_blocked=np.asarray([False]),
    )


class _PrimaryEmbedder:
    _view_count = 1

    def prepare_detection_tensors(self, _image, detections):
        return np.zeros((len(detections), 3, 2, 2), dtype=np.float32)

    def prepare_selected_detection_tensors(self, _image, _detections, detection_indices):
        self.selected_indices = tuple(int(value) for value in detection_indices)
        return np.zeros((len(detection_indices), 3, 2, 2), dtype=np.float32)

    def embed_prepared_tensors_raw(self, tensors):
        return np.zeros((len(tensors), 2), dtype=np.float32)


class _IndependentEmbedder(_PrimaryEmbedder):
    metadata = SimpleNamespace(fixed_batch_size=1)


class _Classifier:
    def __init__(
        self,
        result,
        *,
        fixed_batch_size=None,
        threshold=0.0,
        base_result=None,
        append_only_base_class_count=None,
    ):
        self.result = result
        self.base_result = base_result
        self.labels = [SimpleNamespace(class_id=value) for value in ("a", "b", "c", "d")][
            : result.logits.shape[1]
        ]
        self.version = "0.1.3"
        self.metadata = SimpleNamespace(
            approval_threshold=threshold,
            approval_thresholds=None,
        )
        self.append_only_base_class_count = append_only_base_class_count
        self.embedder = (
            _IndependentEmbedder()
            if fixed_batch_size is not None
            else SimpleNamespace(metadata=SimpleNamespace(fixed_batch_size=None))
        )

    def classify_embeddings(self, _embeddings, _detections=None, *, class_limit=None):
        if class_limit is not None:
            assert self.base_result is not None
            assert class_limit == self.append_only_base_class_count
            return self.base_result
        return self.result

    def classify(self, _image, _detections):
        return self.result


def _consensus(
    rotation,
    independent,
    *,
    verify_unknown_recapture: bool = False,
    verify_any_unknown_recapture: bool = False,
    verify_all_approved_candidates: bool = False,
    primary_approval_score: float = 0.2,
    primary_threshold: float | None = None,
):
    primary = _Classifier(
        _result(0, approval_score=primary_approval_score),
        threshold=(
            0.3
            if verify_unknown_recapture and primary_threshold is None
            else primary_threshold or 0.0
        ),
    )
    primary.embedder = _PrimaryEmbedder()
    return ConsensusCatalogClassifier(
        primary,
        _Classifier(rotation),
        _Classifier(independent, fixed_batch_size=1, threshold=0.212),
        ambiguity_maximum_approval_score=0.5,
        verify_all_approved_candidates=verify_all_approved_candidates,
        unknown_recapture_on_dual_verifier_rejection=verify_unknown_recapture,
        unknown_recapture_on_any_verifier_rejection=verify_any_unknown_recapture,
    )


def test_all_approved_policy_verifies_candidate_above_ambiguity_band() -> None:
    classifier = _consensus(
        _result(1, approval_score=0.7),
        _result(1, approval_score=0.7, retrieval_top1=1),
        verify_all_approved_candidates=True,
        primary_approval_score=0.7,
        primary_threshold=0.1,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert classifier.independent.embedder.selected_indices == (0,)
    assert result.approval_blocked.tolist() == [True]


def test_verifier_uses_the_same_per_class_threshold_as_final_decision() -> None:
    classifier = _consensus(
        _result(1, approval_score=0.7),
        _result(1, approval_score=0.7, retrieval_top1=1),
        primary_approval_score=0.2,
        primary_threshold=0.8,
    )
    classifier.metadata.approval_thresholds = [0.1, None, None]
    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])
    assert classifier.independent.embedder.selected_indices == (0,)
    assert result.approval_blocked.tolist() == [True]


def test_default_policy_skips_approved_candidate_above_ambiguity_band() -> None:
    classifier = _consensus(
        _result(1, approval_score=0.7),
        _result(1, approval_score=0.7, retrieval_top1=1),
        primary_approval_score=0.7,
        primary_threshold=0.1,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert not hasattr(classifier.independent.embedder, "selected_indices")
    assert result.approval_blocked.tolist() == [False]


def test_all_approved_policy_maps_any_verifier_quality_rejection_to_recapture() -> None:
    classifier = _consensus(
        _result(
            0,
            approval_score=0.7,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        _result(0, approval_score=0.7, retrieval_top1=0),
        verify_any_unknown_recapture=True,
        verify_all_approved_candidates=True,
        primary_approval_score=0.7,
        primary_threshold=0.1,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.segment_recapture_reasons == ("CLASSIFIER_TOP3_UNSAFE",)


def test_rotation_disagreement_keeps_primary_only_with_two_head_corroboration() -> None:
    classifier = _consensus(
        _result(1, approval_score=0.4),
        _result(0, approval_score=0.3, retrieval_top1=0),
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.approval_blocked.tolist() == [False]
    assert classifier.independent.embedder.selected_indices == (0,)


def test_selected_consensus_preserves_full_detection_context_indices() -> None:
    classifier = _consensus(
        _result(0, approval_score=0.4),
        _result(0, approval_score=0.3, retrieval_top1=0),
    )
    detections = [
        Detection(0, 0, 1, 1, 0.9),
        Detection(1, 0, 2, 1, 0.9),
        Detection(2, 0, 3, 1, 0.9),
    ]

    result = classifier.classify_selected(None, detections, np.asarray([2]))

    assert result.approval_blocked.tolist() == [False]
    assert classifier.primary.embedder.selected_indices == (2,)
    assert classifier.independent.embedder.selected_indices == (2,)


def test_rotation_disagreement_without_independent_corroboration_blocks_approval() -> None:
    classifier = _consensus(
        _result(1, approval_score=0.4),
        _result(0, approval_score=0.3, retrieval_top1=1),
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.approval_blocked.tolist() == [True]
    assert result.unknown_reasons == ("CLASSIFIER_AMBIGUOUS_TOP2",)


def test_agreeing_geometric_views_are_blocked_by_strong_independent_disagreement() -> None:
    classifier = _consensus(
        _result(0, approval_score=0.4),
        _result(1, approval_score=0.3, retrieval_top1=1),
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.approval_blocked.tolist() == [True]


def test_dual_verifier_rejection_marks_unknown_top3_unsafe() -> None:
    classifier = _consensus(
        _result(
            1,
            approval_score=0.1,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        _result(
            1,
            approval_score=0.1,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        verify_unknown_recapture=True,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.segment_recapture_reasons == ("CLASSIFIER_TOP3_UNSAFE",)


def test_single_verifier_rejection_preserves_safe_unknown_top3() -> None:
    classifier = _consensus(
        _result(
            1,
            approval_score=0.1,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        _result(0, approval_score=0.1),
        verify_unknown_recapture=True,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.segment_recapture_reasons == (None,)


def test_single_verifier_rejection_can_mark_unknown_top3_unsafe() -> None:
    classifier = _consensus(
        _result(0, approval_score=0.1),
        _result(
            1,
            approval_score=0.1,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        verify_any_unknown_recapture=True,
        primary_threshold=0.3,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.segment_recapture_reasons == ("CLASSIFIER_TOP3_UNSAFE",)


def test_dual_rejection_marks_verifier_blocked_approval_top3_unsafe() -> None:
    classifier = _consensus(
        _result(
            1,
            approval_score=0.1,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        _result(
            1,
            approval_score=0.1,
            retrieval_top1=1,
            segment_recapture_reason="CLASSIFIER_OUT_OF_CATALOG",
        ),
        verify_unknown_recapture=True,
        primary_threshold=0.1,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert result.approval_blocked.tolist() == [True]
    assert result.segment_recapture_reasons == ("CLASSIFIER_TOP3_UNSAFE",)


def _append_consensus(rotation_top1: int, independent_top1: int):
    base = _result(0, approval_score=0.8, class_count=3)
    extended = _result(3, approval_score=0.7, class_count=4)
    primary = _Classifier(
        extended,
        base_result=base,
        append_only_base_class_count=3,
    )
    primary.embedder = _PrimaryEmbedder()
    return ConsensusCatalogClassifier(
        primary,
        _Classifier(
            _result(rotation_top1, approval_score=0.7, class_count=4),
            append_only_base_class_count=3,
        ),
        _Classifier(
            _result(independent_top1, approval_score=0.7, class_count=4),
            fixed_batch_size=1,
            threshold=0.212,
            append_only_base_class_count=3,
        ),
        ambiguity_maximum_approval_score=0.5,
    )


def test_append_only_class_requires_identical_three_way_consensus() -> None:
    classifier = _append_consensus(3, 3)

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert ConsensusCatalogClassifier._top1(result).tolist() == [3]
    assert result.approval_scores.tolist() == pytest.approx([0.7])


def test_append_only_disagreement_restores_bit_stable_base_decision() -> None:
    classifier = _append_consensus(3, 1)

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert ConsensusCatalogClassifier._top1(result).tolist() == [0]
    assert result.approval_scores.tolist() == pytest.approx([0.8])
    assert np.isneginf(result.logits[0, 3])


def test_selected_append_only_consensus_preserves_full_detection_context() -> None:
    classifier = _append_consensus(3, 3)
    detections = [
        Detection(0, 0, 1, 1, 0.9),
        Detection(1, 0, 2, 1, 0.9),
        Detection(2, 0, 3, 1, 0.9),
    ]

    result = classifier.classify_selected(None, detections, np.asarray([2]))

    assert ConsensusCatalogClassifier._top1(result).tolist() == [3]
    assert classifier.primary.embedder.selected_indices == (2,)
    assert classifier.independent.embedder.selected_indices == (2,)


def test_append_only_fallback_preserves_base_selective_verification() -> None:
    base = _result(0, approval_score=0.2, class_count=3)
    primary = _Classifier(
        _result(3, approval_score=0.7, class_count=4),
        base_result=base,
        append_only_base_class_count=3,
    )
    primary.embedder = _PrimaryEmbedder()
    classifier = ConsensusCatalogClassifier(
        primary,
        _Classifier(
            _result(2, approval_score=0.7, class_count=4),
            base_result=_result(1, approval_score=0.2, class_count=3),
            append_only_base_class_count=3,
        ),
        _Classifier(
            _result(3, approval_score=0.7, class_count=4),
            fixed_batch_size=1,
            threshold=0.212,
            base_result=_result(1, approval_score=0.2, retrieval_top1=1, class_count=3),
            append_only_base_class_count=3,
        ),
        ambiguity_maximum_approval_score=0.5,
    )

    result = classifier.classify(None, [Detection(0, 0, 1, 1, 0.9)])

    assert ConsensusCatalogClassifier._top1(result).tolist() == [0]
    assert result.approval_blocked.tolist() == [True]
    assert result.unknown_reasons == ("CLASSIFIER_AMBIGUOUS_TOP2",)
