from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.experiments.bread.log_improvement import prepare
from bixolon_scanner.training.three_bakery_data import write_json, write_jsonl


@pytest.fixture
def source_config(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "log").mkdir(parents=True)
    original = root / "original.png"
    log = root / "log/new.png"
    Image.new("RGB", (32, 24), "red").save(original)
    Image.new("RGB", (32, 24), "blue").save(log)
    originals = tmp_path / "originals.jsonl"
    write_jsonl(
        originals,
        [
            {
                "image_path": str(original),
                "image_sha256": sha256_file(original),
                "annotations": [{"category_id": 1, "bbox_xywh": [0, 0, 32, 24]}],
                "source_group": "shared-physical-items",
            }
        ],
    )
    annotation = {
        "image_id": 1,
        "image_path": "log/new.png",
        "image_sha256": sha256_file(log),
        "width": 32,
        "height": 24,
        "source_group": "shared-physical-items",
        "capture_session_id": "new-session",
        "annotations": [
            {"bbox_xyxy": [0, 0, 32, 24], "category_id": 1, "reviewed": True, "object_number": 1}
        ],
    }
    write_jsonl(root / "log/annotations.jsonl", [annotation])
    write_json(
        root / "log/confirmation.json",
        {
            "user_confirmed": True,
            "review_status": "confirmed",
            "annotation_sha256": sha256_file(root / "log/annotations.jsonl"),
        },
    )
    config = {
        "work": str(tmp_path / "work"),
        "dataset_root": str(root),
        "log_annotations": "log/annotations.jsonl",
        "log_confirmation": "log/confirmation.json",
        "original_manifest": str(originals),
        "expected_training_images": 1,
        "expected_training_objects": 1,
        "expected_log_images": 1,
        "expected_log_objects": 1,
        "data_policy": "Untrained photos are not independent physical items.",
    }
    path = tmp_path / "config.json"
    write_json(path, config)
    return path, root, annotation


def test_frozen_log_audit_is_resumable_and_reports_shared_group(source_config):
    path, _, _ = source_config
    first = prepare(path)
    assert prepare(path) == first
    assert first["sha256_overlap"] == 0
    assert first["shares_physical_group_with_training"]
    assert first["classes"] == {"1": 1}


def test_log_changed_after_user_confirmation_is_rejected(source_config):
    path, root, annotation = source_config
    annotation["annotations"][0]["category_id"] = 2
    write_jsonl(root / "log/annotations.jsonl", [annotation])
    with pytest.raises(ValueError, match="confirmed annotation checksum"):
        prepare(path)


@pytest.mark.parametrize("case", ["duplicate", "bounds", "unreviewed", "dimensions"])
def test_invalid_log_input_cannot_enter_comparison(source_config, case):
    path, root, annotation = source_config
    if case == "duplicate":
        annotation["image_path"] = "original.png"
        annotation["image_sha256"] = sha256_file(root / "original.png")
    elif case == "bounds":
        annotation["annotations"][0]["bbox_xyxy"][2] = 33
    elif case == "unreviewed":
        annotation["annotations"][0]["reviewed"] = False
    else:
        annotation["width"] = 33
    write_jsonl(root / "log/annotations.jsonl", [annotation])
    write_json(
        root / "log/confirmation.json",
        {
            "user_confirmed": True,
            "review_status": "confirmed",
            "annotation_sha256": sha256_file(root / "log/annotations.jsonl"),
        },
    )
    with pytest.raises(ValueError):
        prepare(path)
