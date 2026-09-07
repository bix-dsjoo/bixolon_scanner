from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.experiments.bread.limited220 import reuse_stage, verify_derivatives
from bixolon_scanner.training.limited_augmentation import generate_training_derivatives
from bixolon_scanner.training.limited_source import (
    prepare_sources,
    read_jsonl,
    select_scenes,
    verify_sources,
    write_json,
)


def _fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    images = []
    annotations = []
    for category in (1, 2):
        directory = root / f"single_objects_2/bread_{category:02d}_name"
        directory.mkdir(parents=True)
        for shot in (1, 2):
            Image.new("RGB", (64, 64), (category * 70, shot * 60, 20)).save(
                directory / f"{shot}.png"
            )
    for i in range(1, 5):
        path = root / f"multi_object_scenes/easy/{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), (i * 40, 100, 50)).save(path)
        images.append(
            {
                "id": i,
                "file_name": f"../multi_object_scenes/easy/{i}.png",
                "width": 100,
                "height": 100,
            }
        )
        for category in (1, 2):
            annotations.append(
                {
                    "id": i * 2 + category,
                    "image_id": i,
                    "category_id": category,
                    "bbox": [5 + (category - 1) * 40, 10, 30, 30],
                }
            )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "One"}, {"id": 2, "name": "Two"}],
    }
    write_json(root / "annotations/instances.json", coco)
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "schema_version": "1.0",
            "dataset_root": str(root),
            "single_directory": "single_objects_2",
            "multi_annotations": "annotations/instances.json",
            "class_count": 2,
            "shots_per_class": 2,
            "multi_count": 2,
            "original_budget": 6,
            "seed": 123,
        },
    )
    return config, root, coco


def test_selection_is_deterministic_and_covers_classes(tmp_path):
    _, _, coco = _fixture(tmp_path)
    selected = select_scenes(coco, 2, 123)
    assert selected == select_scenes(coco, 2, 123)
    assert len({r["id"] for r in selected}) == 2
    assert {a["category_id"] for r in selected for a in r["annotations"]} == {1, 2}


def test_source_budget_and_same_item_group_are_preserved(tmp_path):
    config, root, _ = _fixture(tmp_path)
    out = tmp_path / "sources"
    result = prepare_sources(config, out)
    assert result["original_count"] == 6
    assert result["independent_validation_available"] is False
    assert {r["source_group"] for r in read_jsonl(out / "originals.jsonl")} == {
        "shared_physical_item_collection"
    }
    verify_sources(out)
    source = read_jsonl(out / "originals.jsonl")[0]
    (root / source["image_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="original image changed"):
        verify_sources(out)


@pytest.mark.parametrize("background_style", ["gradient", "surfaces"])
def test_synthetic_derivatives_never_add_originals(tmp_path, background_style):
    config, _, _ = _fixture(tmp_path)
    work = tmp_path / "work"
    source = prepare_sources(config, work / "sources")
    result = generate_training_derivatives(
        work / "sources",
        work / "derived",
        {
            "image_size": 64,
            "image_count": 20,
            "maximum_objects": 3,
            "minimum_visible_fraction": 0.65,
            "empty_probability": 0.2,
            "seed": 21,
            "background_style": background_style,
        },
    )
    assert result["original_image_count"] == 6
    assert result["synthetic_count"] == 20
    assert result["multi_crop_count"] == 4
    verify_derivatives(work, source)
    for row in read_jsonl(work / "derived/synthetic.jsonl"):
        for a in row["annotations"]:
            x, y, w, h = a["bbox_xywh"]
            assert 0 <= x < x + w <= 64 and 0 <= y < y + h <= 64
            assert a["visible_fraction"] >= 0.65
    image = work / "derived" / read_jsonl(work / "derived/synthetic.jsonl")[0]["image_path"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="derived image changed"):
        verify_derivatives(work, source)


def test_manifest_tampering_is_rejected(tmp_path):
    config, _, _ = _fixture(tmp_path)
    output = tmp_path / "sources"
    prepare_sources(config, output)
    (output / "classifier.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="manifest changed"):
        verify_sources(output)


@pytest.mark.parametrize("changed", [None, "input", "output", "recipe"])
def test_stage_reuse_requires_same_recipe_inputs_and_intact_outputs(tmp_path, changed):
    config, _, _ = _fixture(tmp_path)
    previous, work = tmp_path / "previous", tmp_path / "next"
    for directory in (previous, work):
        source = prepare_sources(config, directory / "sources")
        generate_training_derivatives(
            directory / "sources",
            directory / "derived",
            {
                "image_size": 64,
                "image_count": 2,
                "maximum_objects": 2,
                "minimum_visible_fraction": 0.65,
                "empty_probability": 0.2,
                "seed": 21,
            },
        )
        inputs, output = directory / "input.bin", directory / "output.bin"
        inputs.write_bytes(b"changed" if changed == "input" and directory == work else b"input")
        argv = ["python", str(directory / "train.py")]
        if changed == "recipe" and directory == work:
            argv.append("--different")
        stage = {"name": "model", "argv": argv, "inputs": [str(inputs)], "outputs": [str(output)]}
        plan = {
            "work_dir": str(directory),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "pretrained_weights": {},
            "stages": [stage],
        }
        write_json(directory / "plan.json", plan)
        (directory / "plan.sha256").write_text(sha256_file(directory / "plan.json"))
        if directory == previous:
            output.write_bytes(b"trained")
            write_json(
                directory / "stages/model.json",
                {
                    "returncode": 0,
                    "argv": argv,
                    "source_manifest_sha256": source["source_manifest_sha256"],
                    "output_sha256s": {str(output): sha256_file(output)},
                },
            )
            if changed == "output":
                output.write_bytes(b"tampered")
    if changed:
        with pytest.raises(
            ValueError,
            match={
                "input": "input differs",
                "output": "output changed",
                "recipe": "recipe differs",
            }[changed],
        ):
            reuse_stage(work, previous, "model")
        assert not (work / "output.bin").exists()
    else:
        result = reuse_stage(work, previous, "model")
        assert result["returncode"] == 0
        assert (work / "output.bin").read_bytes() == b"trained"
