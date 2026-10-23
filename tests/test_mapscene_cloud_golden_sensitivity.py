"""The cloud recipe's pixel gate must reject a real loss of cloud shadows."""
import os
from pathlib import Path

import forge3d.map_scene as ms

from _concordia_screen_fixtures import cloud_fixture_measurements
from test_recipe_goldens import MEAN_ABS_MAX, _cloud_shadows


def test_cloud_golden_detects_disabled_shadows_on_the_same_route(tmp_path):
    directory = os.environ.get("FORGE3D_CONCORDIA_ARTIFACT_DIR")
    model = _cloud_shadows(tmp_path)
    assert "ssr" not in ms._mapscene_native_effects(model.recipe)
    metrics = cloud_fixture_measurements(model,
        Path(directory)/"cloud-golden" if directory else None)
    assert "ssr" not in ms._mapscene_native_effects(model.recipe)
    assert metrics["changed_above_tolerance_fraction"] >= 0.05
    assert metrics["whole_image_mean_abs_difference"] > MEAN_ABS_MAX
