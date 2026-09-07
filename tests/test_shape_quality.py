from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from bixolon_scanner.training.shape_quality import extract_shape_quality_features


def test_shape_quality_features_are_finite_and_color_invariant() -> None:
    first = Image.new("RGB", (120, 80), "white")
    second = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(first).ellipse((20, 10, 100, 70), fill=(180, 90, 30))
    ImageDraw.Draw(second).ellipse((20, 10, 100, 70), fill=(90, 45, 15))
    first_features = extract_shape_quality_features(first)
    second_features = extract_shape_quality_features(second)
    assert first_features.shape == (21,)
    assert np.isfinite(first_features).all()
    np.testing.assert_allclose(first_features, second_features, atol=0.12)
