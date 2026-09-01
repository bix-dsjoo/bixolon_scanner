import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from bixolon_scanner.training.grid_compositor import (
    GRID_POSITIONS,
    POSITION_ANCHORS,
    SIDES,
    GridCompositeRecipe,
    GridScenePlan,
    discover_capture_grid,
    discover_pose_cutouts,
    generate_grid_composite_dataset,
)


def _capture_grid(root: Path, *, image_size: int = 160) -> Path:
    root.mkdir()
    background = Image.new("RGB", (image_size, image_size), (235, 233, 228))
    draw = ImageDraw.Draw(background)
    draw.rectangle((6, 7, image_size - 7, image_size - 8), outline=(222, 221, 217), width=2)
    background.save(root / "background.png")
    for category_id in range(1, 21):
        directory = root / f"bread_{category_id:02d}_class_{category_id:02d}"
        directory.mkdir()
        for side_index, side in enumerate(SIDES):
            for position_index, position in enumerate(GRID_POSITIONS):
                image = background.copy()
                draw = ImageDraw.Draw(image)
                center_x = int(round(POSITION_ANCHORS[position][0] * image_size))
                center_y = int(round(POSITION_ANCHORS[position][1] * image_size))
                radius_x = 9 + category_id % 4
                radius_y = 8 + category_id % 3
                color = (
                    118 + category_id * 5,
                    55 + side_index * 38 + category_id,
                    20 + position_index * 6,
                )
                draw.ellipse(
                    (
                        center_x - radius_x,
                        center_y - radius_y,
                        center_x + radius_x,
                        center_y + radius_y,
                    ),
                    fill=color,
                )
                draw.point(
                    (center_x - radius_x + 1 + position_index, center_y),
                    fill=(category_id, side_index, position_index),
                )
                image.save(directory / f"{side}_{position}.png")
    return root


def _small_recipe() -> GridCompositeRecipe:
    return GridCompositeRecipe(
        plan=GridScenePlan(
            normal=4,
            dense=2,
            occlusion=2,
            edge=2,
            lighting=2,
            recapture=5,
        ),
        tray_roi_xyxy=(0.03, 0.03, 0.97, 0.97),
        placement_attempts=180,
        scene_attempts=20,
    )


def _pose_cutouts(root: Path, background_path: Path, *, image_size: int = 160) -> Path:
    root.mkdir()
    background = Image.new("RGB", (image_size, image_size), (235, 233, 228))
    ImageDraw.Draw(background).rectangle(
        (6, 7, image_size - 7, image_size - 8), outline=(222, 221, 217), width=2
    )
    background.save(background_path)
    views = (
        "ground_30_dir_01",
        "ground_30_dir_02",
        "ground_30_dir_03",
        "ground_30_dir_04",
        "vertical",
    )
    for category_id in range(1, 21):
        directory = root / f"bread_{category_id:02d}_class_{category_id:02d}"
        directory.mkdir()
        for side_index, side in enumerate(SIDES):
            for view_index, view in enumerate(views):
                width = 34 + category_id % 4
                height = 38 if view == "vertical" else 28 + category_id % 3
                image = Image.new("RGB", (width, height), (255, 255, 255))
                draw = ImageDraw.Draw(image)
                color = (
                    118 + category_id * 5,
                    55 + side_index * 38 + category_id,
                    20 + view_index * 6,
                )
                draw.ellipse((2, 2, width - 3, height - 3), fill=color)
                draw.point(
                    (3 + view_index, height // 2), fill=(category_id, side_index, view_index)
                )
                image.save(directory / f"bread_{category_id:02d}_{side}_{view}.jpg", quality=98)
    return root


def test_default_grid_scene_plan_contains_exactly_1500_images():
    plan = GridScenePlan()

    assert plan.total == 1500
    assert Counter(plan.expanded()) == {
        "NORMAL": 600,
        "DENSE": 300,
        "OCCLUSION": 225,
        "EDGE": 150,
        "LIGHTING": 125,
        "RECAPTURE": 100,
    }


def test_capture_grid_requires_20_classes_and_10_views_each(tmp_path):
    capture_root = _capture_grid(tmp_path / "captures")

    grid = discover_capture_grid(capture_root)

    assert len(grid.assets) == 200
    assert {row.category_id for row in grid.assets} == set(range(1, 21))
    assert Counter((row.side, row.grid_position) for row in grid.assets) == {
        (side, position): 20 for side in SIDES for position in GRID_POSITIONS
    }


def test_pose_cutouts_accept_four_directions_and_vertical_with_external_background(tmp_path):
    background_path = tmp_path / "empty_tray.png"
    capture_root = _pose_cutouts(tmp_path / "captures", background_path)

    grid = discover_pose_cutouts(capture_root, background_path)

    assert len(grid.assets) == 200
    assert grid.contract == "bread-pose-cutouts-20x2x5"
    assert {row.capture_view for row in grid.assets} == {
        "ground_30_dir_01",
        "ground_30_dir_02",
        "ground_30_dir_03",
        "ground_30_dir_04",
        "vertical",
    }
    assert {row.extraction_mode for row in grid.assets} == {"white_background_cutout"}


def test_pose_cutout_compositor_generates_visible_annotations(tmp_path):
    background_path = tmp_path / "empty_tray.png"
    capture_root = _pose_cutouts(tmp_path / "captures", background_path)
    output_root = tmp_path / "output"

    metadata = generate_grid_composite_dataset(
        capture_root,
        output_root,
        seed=47,
        recipe=_small_recipe(),
        background_image=background_path,
    )

    rows = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    provenance = [
        json.loads(line)
        for line in (output_root / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["dataset_version"] == "bread-pose-cutouts-20x2x5"
    assert metadata["generated_image_count"] == 17
    assert all(row["source_dataset"] == "bread-pose-cutouts-20x2x5" for row in rows)
    assert {obj["source_capture_view"] for row in provenance for obj in row["objects"]} <= {
        "ground_30_dir_01",
        "ground_30_dir_02",
        "ground_30_dir_03",
        "ground_30_dir_04",
        "vertical",
    }


def test_grid_compositor_exports_balanced_visible_annotations(tmp_path):
    capture_root = _capture_grid(tmp_path / "captures")
    output_root = tmp_path / "output"

    metadata = generate_grid_composite_dataset(
        capture_root,
        output_root,
        seed=47,
        recipe=_small_recipe(),
    )

    rows = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["source_image_count"] == 200
    assert metadata["generated_image_count"] == 17
    assert metadata["scenario_counts"] == {
        "DENSE": 2,
        "EDGE": 2,
        "LIGHTING": 2,
        "NORMAL": 4,
        "OCCLUSION": 2,
        "RECAPTURE": 5,
    }
    assert metadata["recapture_subtype_counts"] == {
        "EMPTY_TRAY": 1,
        "OVEREXPOSED": 1,
        "SEVERE_BLUR": 1,
        "SEVERE_OCCLUSION": 1,
        "UNDEREXPOSED": 1,
    }
    assert all(row["split"] == "train_synthetic" and row["fold"] == 0 for row in rows)
    assert sum(not row["annotations"] for row in rows) == 1
    occlusion_rows = [row for row in rows if row["scenario"] == "OCCLUSION"]
    assert all("OBJECT_OCCLUSION_10_30" in row["condition_tags"] for row in occlusion_rows)
    assert all(
        any(annotation["occlusion_fraction"] >= 0.10 for annotation in row["annotations"])
        for row in occlusion_rows
    )
    edge_tags = {tag for row in rows if row["scenario"] == "EDGE" for tag in row["condition_tags"]}
    assert edge_tags <= {"FRAME_CLIPPED", "TRAY_EDGE_TOUCH"}
    assert edge_tags
    assert all(
        0 < annotation["visible_fraction"] <= 1
        and annotation["bbox_xywh"][0] >= 0
        and annotation["bbox_xywh"][1] >= 0
        and annotation["bbox_xywh"][0] + annotation["bbox_xywh"][2] <= 160
        and annotation["bbox_xywh"][1] + annotation["bbox_xywh"][3] <= 160
        for row in rows
        for annotation in row["annotations"]
    )
    assert (output_root / "instances.json").is_file()
    assert (output_root / "source-manifest.json").is_file()
    assert (output_root / "provenance.jsonl").is_file()
    assert (output_root / "preview.jpg").is_file()


def test_grid_compositor_is_seeded_and_rejects_nonempty_output(tmp_path):
    capture_root = _capture_grid(tmp_path / "captures")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_grid_composite_dataset(
        capture_root,
        first_root,
        seed=91,
        recipe=_small_recipe(),
    )
    second = generate_grid_composite_dataset(
        capture_root,
        second_root,
        seed=91,
        recipe=_small_recipe(),
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["instances_sha256"] == second["instances_sha256"]
    try:
        generate_grid_composite_dataset(
            capture_root,
            first_root,
            seed=91,
            recipe=_small_recipe(),
        )
    except FileExistsError as error:
        assert "must be empty" in str(error)
    else:
        raise AssertionError("non-empty output root must be rejected")
