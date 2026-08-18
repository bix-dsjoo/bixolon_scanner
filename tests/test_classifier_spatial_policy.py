from __future__ import annotations

from bixolon_scanner.experiments.bread.classifier_spatial_policy import spatial_recipes


def test_spatial_recipes_limit_runtime_to_two_views() -> None:
    recipes = list(spatial_recipes(("a", "b", "c"), (0.25, 0.5, 0.75)))

    assert len(recipes) == 12
    assert all(len(recipe.names) <= 2 for recipe in recipes)
    assert all(abs(sum(recipe.weights) - 1.0) < 1e-12 for recipe in recipes)
