import pytest

from bixolon_scanner.experiments.bread.detector_recapture_union import (
    union_recapture_ids,
)


def test_union_recapture_ids_combines_rules() -> None:
    reports = [
        {"selected": {"recaptured_image_ids": [1, 2]}},
        {"selected": {"recaptured_image_ids": [2, 3]}},
    ]

    assert union_recapture_ids(reports) == {1, 2, 3}


def test_union_recapture_ids_rejects_missing_contract() -> None:
    with pytest.raises(ValueError, match="recaptured_image_ids"):
        union_recapture_ids([{"selected": {}}])
