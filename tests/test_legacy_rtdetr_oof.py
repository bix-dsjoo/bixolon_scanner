from bixolon_scanner.experiments.bread.legacy_rtdetr_oof import (
    checkpoint_route,
    unseen_checkpoint_routes,
)


def test_checkpoint_route_uses_legacy_outer_fold_for_development_scene():
    record = {"source_image_id": 10, "evaluation_set": "multi_object_scenes"}
    old = {10: {"split": "development", "fold": 2}}

    assert checkpoint_route(record, old) == ("fold2", "legacy_outer_fold")


def test_checkpoint_route_uses_unseen_final_model_for_new_sources():
    new_scene = {"source_image_id": 300, "evaluation_set": "multi_object_scenes"}

    assert checkpoint_route(new_scene, {}) == (
        "final",
        "new_multi_object_unseen_by_final_training",
    )


def test_all_unseen_routes_use_every_checkpoint_for_legacy_test():
    record = {"source_image_id": 7, "evaluation_set": "multi_object_scenes"}
    old = {7: {"split": "test", "fold": 2}}

    assert unseen_checkpoint_routes(record, old) == (
        ("fold0", "fold1", "fold2", "final"),
        "legacy_locked_test_unseen_by_every_legacy_checkpoint",
    )


def test_all_unseen_routes_keep_only_outer_fold_for_legacy_development():
    record = {"source_image_id": 7, "evaluation_set": "multi_object_scenes"}
    old = {7: {"split": "development", "fold": 2}}

    assert unseen_checkpoint_routes(record, old) == (
        ("fold2",),
        "legacy_outer_fold",
    )
