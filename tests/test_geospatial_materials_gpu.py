"""C01 existing render bindings, using synthetic fixtures and a physical adapter."""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import forge3d as f3d


def test_all_surface_bindings_reach_physical_renderers(tmp_path):
    adapter = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert adapter["status"] == "ok" and adapter["device_type"] in {"DiscreteGpu", "IntegratedGpu"}
    assert not adapter.get("software_fallback")
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    palette = f3d.CategoricalRasterStyle({1: (120, 100, 80, 255), 2: (230, 220, 210, 255)}, {1: "Low", 2: "High"})
    material = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("base", (120, 100, 80)),), categorical=palette)
    base = f3d.SceneRecipe(f3d.TerrainSource(data=np.zeros((48, 48), np.float32), crs=grid.crs,
        elevation_sampling_available=True,
        metadata={"source_id": "synthetic material fixture", "bounds":[499999,4599905,500095,4600001],
                  "resolution":[2,2], "width":48,"height":48}),
        f3d.OrbitCamera(distance=300, azimuth_deg=140, elevation_deg=55),
        f3d.LightingPreset("custom", settings={"cli_params": {"camera_mode": "mesh"}}),
        output=f3d.OutputSpec(width=384, height=256))
    classes = np.ones((48, 48), np.int16)
    classes[:, 24:] = 2
    bound = material.bind_raster(base, np.ones((48, 48), np.float32), grid, units="kg", classes=classes)
    scene = f3d.MapScene(recipe=replace(bound, map_furniture=None))
    scene.render(tmp_path / "raster.png")
    assert scene.last_render_backend == "gpu_terrain"
    raster = np.asarray(Image.open(tmp_path / "raster.png"))
    for color in palette.palette.values():
        assert np.any(np.all(raster[..., :3] == color[:3], axis=-1))
    bundle = tmp_path / "raster.forge3d"
    scene.save_bundle(bundle)
    restored = f3d.MapScene.load_bundle(bundle)
    assert restored.recipe.layers[0].style.to_dict() == scene.recipe.layers[0].style.to_dict()
    restored.render(tmp_path / "raster-replay.png")
    np.testing.assert_array_equal(raster, np.asarray(Image.open(tmp_path / "raster-replay.png")))

    # Provided feature coordinates and material anchors share the source CRS.
    vector_features = [{"id": str(i), "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[500010+i*40,4599990],[500030+i*40,4599990],
        [500030+i*40,4599950],[500010+i*40,4599950],[500010+i*40,4599990]]]}} for i in range(2)]
    vector = f3d.VectorOverlay("material-vector", features=vector_features, crs=grid.crs,
                              metadata={"coordinate_space": "world"})
    vector = material.bind_vector(vector, [[500020,4599970],[500060,4599970]], units="m", classes=[1,2])
    vector_scene = f3d.MapScene(recipe=replace(base, layers=(vector,)))
    vector_scene.render(tmp_path / "vector.png")
    assert vector_scene.last_render_backend == "gpu_terrain"
    vector_pixels = np.asarray(Image.open(tmp_path / "vector.png"))
    for color in palette.palette.values():
        assert np.any(np.all(vector_pixels[..., :3] == color[:3], axis=-1))

    footprints = [{"id": str(i), "type": "Feature", "properties": {"height":20},
        "geometry": {"type":"Polygon", "coordinates": [[[500000+i*30,4599970],
        [500020+i*30,4599970],[500020+i*30,4599950],[500000+i*30,4599950],[500000+i*30,4599970]]]}}
        for i in range(2)]
    buildings = f3d.MapSceneBuildingLayer("material-buildings", features=footprints,
                                         metadata={"crs":grid.crs, "source_id":"synthetic building fixture"})
    before = f3d.MapScene(recipe=replace(base, layers=(buildings,)))
    before.render(tmp_path / "buildings-before.png")
    buildings = material.bind_buildings(buildings, [[500010,4599960],[500040,4599960]], units="m", classes=[1,2])
    after = f3d.MapScene(recipe=replace(base, layers=(buildings,)))
    after.render(tmp_path / "buildings-after.png")
    assert after.last_render_backend == "gpu_terrain"
    prior = np.asarray(Image.open(tmp_path / "buildings-before.png"))
    current = np.asarray(Image.open(tmp_path / "buildings-after.png"))
    changed = int(np.any(prior != current, axis=-1).sum())
    assert changed > 0
    record = {"adapter":adapter, "raster_palette_observed":True,
              "vector_palette_observed":True, "vector_backend":vector_scene.last_render_metadata.get("vector_backend"),
              "building_changed_pixels":changed, "building_metadata":after.last_render_metadata,
              "limits":"Synthetic source geometry; scalar building albedo is lit, not exact display swatches."}
    if os.environ.get("FORGE3D_C01_ARTIFACT_DIR"):
        path = Path(os.environ["FORGE3D_C01_ARTIFACT_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "all-surface-gpu-evidence.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record))
