from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def classifier_metadata():
    from bixolon_scanner.package import ClassifierMetadata, ClassLabel

    return ClassifierMetadata(
        filename="classifier.onnx",
        version="1.0.0",
        input_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        approval_threshold=0.8,
        temperature=1.0,
        labels=[
            ClassLabel(class_id="bread_01", class_name="Walnut Donut"),
            ClassLabel(class_id="bread_02", class_name="Croffle"),
            ClassLabel(class_id="bread_03", class_name="Waffle"),
        ],
    )


@pytest.fixture
def quality_metadata():
    from bixolon_scanner.package import QualityMetadata

    return QualityMetadata(min_object_area_ratio=0.001, border_margin_ratio=0.0)
