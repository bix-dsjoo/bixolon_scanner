import numpy as np
import pytest

from bixolon_scanner.training.staged_classifier_export import view_affine


def test_view_affine_defines_flips_and_rotations():
    np.testing.assert_array_equal(
        view_affine("vflip"),
        np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32),
    )
    matrix = view_affine("rot15")
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(2), atol=1e-6)


def test_view_affine_rejects_unknown_view():
    with pytest.raises(ValueError, match="unsupported"):
        view_affine("diagonal")
