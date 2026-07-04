from __future__ import annotations

import numpy as np

import forge3d as f3d


def test_recipe_manifest_is_exported_and_deterministic() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        lighting=f3d.LightingPreset(name="rainier_showcase"),
        output=f3d.OutputSpec(width=64, height=64, samples=4, aovs=("albedo",), hdr=True),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[{"id": "r1", "geometry": {"type": "LineString", "coordinates": [(0, 0), (1, 1)]}}],
            )
        ],
    )

    first = f3d.recipe_manifest(scene)
    second = f3d.recipe_manifest(scene.recipe)

    assert first == second
    assert first["schema"] == "forge3d.mapscene.recipe_manifest.v1"
    assert first["terrain"]["source_id"] == "inline-dem"
    assert first["output"]["samples"] == 4
    assert first["output"]["aovs"] == ["albedo"]
    assert first["layers"][0]["layer_id"] == "roads"
    assert "recipe_manifest" in f3d.__all__


def test_recipe_manifest_records_golden_fixture_intent() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=32, height=32),
    )

    manifest = f3d.recipe_manifest(
        scene,
        golden_fixture_intent={
            "scene_id": "mapscene_fixture",
            "family": "terrain_raster",
            "golden_path": "tests/golden/recipes/mapscene_fixture.png",
            "command": "pytest tests/test_recipe_goldens.py -k mapscene_fixture",
            "backend": "placeholder",
        },
    )

    intent = manifest["golden_fixture_intent"]
    assert intent["schema"] == "forge3d.mapscene.golden_fixture_intent.v1"
    assert intent["status"] == "active"
    assert intent["family"] == "terrain_raster"
    assert intent["tolerance"] == {"ssim_min": 0.995, "mean_abs_max": 2.0}


def test_map_scene_pyi_exists_for_public_recipe_api() -> None:
    from pathlib import Path

    stub = Path("python/forge3d/map_scene.pyi").read_text(encoding="utf-8")

    for symbol in ("class MapScene", "class OutputSpec", "allow_placeholder", "samples", "class LightingPreset"):
        assert symbol in stub
