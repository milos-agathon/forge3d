from __future__ import annotations

import numpy as np

import forge3d as f3d
from forge3d import presets
from forge3d.config import load_renderer_config


def _terrain() -> f3d.TerrainSource:
    return f3d.TerrainSource(
        data=np.zeros((12, 16), dtype=np.float32),
        crs="EPSG:32610",
        metadata={"width": 16, "height": 12, "source_id": "inline-dem"},
        elevation_sampling_available=True,
    )


def test_premium_presets_are_self_contained_renderer_configs() -> None:
    for name in ("rainier_showcase", "rainier_relief"):
        preset = presets.get(name)
        cfg = load_renderer_config(preset)
        cfg_dict = cfg.to_dict()

        assert cfg.shadows.enabled
        assert cfg.shadows.cascades >= 3
        assert cfg.atmosphere.enabled
        assert cfg_dict["camera"] == preset["camera"]
        assert cfg_dict["sun"] == preset["sun"]
        assert cfg_dict["ibl"] == preset["ibl"]
        assert cfg_dict["exaggeration"] == preset["exaggeration"]
        assert preset["camera"]["radius_scale"] > 1.0
        assert "azimuth_deg" in preset["sun"]
        assert "intensity" in preset["ibl"]

    assert "CLI" not in (presets.rainier_showcase.__doc__ or "")
    assert "CLI" not in (presets.rainier_relief.__doc__ or "")


def test_mapscene_resolves_named_lighting_preset_into_recipe_fields() -> None:
    scene = f3d.MapScene(
        terrain=_terrain(),
        lighting=f3d.LightingPreset(name="rainier_showcase"),
        output=f3d.OutputSpec(width=64, height=64, path="out.png"),
    )

    assert scene.recipe.camera.distance > 20.0
    assert scene.recipe.camera.azimuth_deg == 135.0
    assert scene.recipe.camera.elevation_deg == 45.0
    assert scene.recipe.lighting.sun_direction is not None
    assert scene.recipe.lighting.intensity == 4.0
    assert scene.recipe.lighting.settings["resolved_preset"] == "rainier_showcase"
    assert scene.recipe.lighting.settings["renderer_config"]["shadows"]["technique"] == "pcss"
    assert scene.recipe.reproducibility_profile is not None
    assert scene.recipe.reproducibility_profile.renderer_backend == "gpu_terrain"

    report = scene.validate()
    assert report.status == "ok"
    assert report.supported_features["mapscene.presets"] == "supported"


def test_mapscene_lighting_preset_overrides_are_recorded_as_info() -> None:
    scene = f3d.MapScene(
        terrain=_terrain(),
        lighting=f3d.LightingPreset(
            name="rainier_showcase",
            overrides={
                "camera": {"azimuth_deg": 10.0},
                "sun": {"intensity": 7.0},
                "reproducibility": {"seed": 99},
            },
        ),
        output=f3d.OutputSpec(width=64, height=64, path="out.png"),
    )

    report = scene.validate()
    override = next(diagnostic for diagnostic in report.diagnostics if diagnostic.code == "preset_override")

    assert report.status == "ok"
    assert scene.recipe.camera.azimuth_deg == 10.0
    assert scene.recipe.lighting.intensity == 7.0
    assert scene.recipe.reproducibility_profile is not None
    assert scene.recipe.reproducibility_profile.seed == 99
    assert override.severity == "info"
    assert override.details["preset"] == "rainier_showcase"
    assert override.details["fields"] == ["camera", "reproducibility", "sun"]
