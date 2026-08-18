import pytest

from bixolon_scanner.experiments.bread.selective_classifier_dataset import (
    recaptured_image_ids,
)


def test_recaptured_image_ids_reads_disagreement_gate():
    assert recaptured_image_ids(
        {"disagreement_recapture_diagnostic": {"recaptured_image_ids": [3, 7]}}
    ) == {3, 7}


def test_recaptured_image_ids_reads_fixed_rule_union():
    assert recaptured_image_ids({"recaptured_image_ids": [2, 5]}) == {2, 5}


def test_recaptured_image_ids_requires_gate_result():
    with pytest.raises(ValueError, match="no recapture"):
        recaptured_image_ids({})
