"""C01 physical draw evidence. These tests fail when the GPU path is absent."""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import forge3d as f3d


def test_protected_material_palette_survives_light_camera_and_bundle(tmp_path):
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert probe["status"] == "ok" and probe["device_type"] in {"DiscreteGpu", "IntegratedGpu"}, probe
    assert not probe.get("software_fallback"), probe
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (1, 1), "m", "m")
    yy, xx = np.mgrid[:48, :48]
    heights = (5 * np.sin(xx / 10) * np.cos(yy / 10)).astype(np.float32)
    classes = np.where(xx < 24, 2, 7).astype(np.int16)
    palette = f3d.CategoricalRasterStyle({2: (80, 140, 50, 255), 7: (20, 90, 220, 255)}, {2: "forest", 7: "water"})
    material = f3d.TerrainMaterialRecipe.land_cover(palette, albedo=(170, 125, 80), relief_strength=.6)
    base = f3d.SceneRecipe(f3d.TerrainSource(data=heights, crs=grid.crs),
        f3d.OrbitCamera(distance=100, elevation_deg=55, azimuth_deg=140),
        f3d.LightingPreset("custom", sun_direction=(.8, .6, 0), settings={"cli_params": {"camera_mode": "mesh"}}),
        output=f3d.OutputSpec(width=256, height=256))
    bound = material.bind(base, grid, classes=classes)
    labelled = f3d.MapScene(recipe=bound)
    labelled.render(tmp_path / "legend.png")
    legend_pixels = np.asarray(Image.open(tmp_path / "legend.png"))
    for entry in bound.map_furniture.legend["entries"]:
        assert np.any(np.all(legend_pixels[..., :3] == entry["rgba"][:3], axis=-1))
    # No furniture in the light invariant comparison; swatches tested separately.
    bound = replace(bound, map_furniture=None)
    scene = f3d.MapScene(recipe=bound)
    scene.render(tmp_path / "first.png")
    assert scene.last_render_backend == "gpu_terrain"
    assert scene.last_render_metadata["thematic_backend"] == "native_terrain_rgba_cells"
    first = np.asarray(Image.open(tmp_path / "first.png"))
    scene_b = f3d.MapScene(recipe=replace(bound, lighting=replace(bound.lighting, sun_direction=(-.8, .1, 0), intensity=5)))
    scene_b.render(tmp_path / "second.png")
    second = np.asarray(Image.open(tmp_path / "second.png"))
    np.testing.assert_array_equal(first, second)
    counts = []
    for color in palette.palette.values():
        count = int(np.all(first[..., :3] == color[:3], axis=-1).sum())
        assert count > 0, (color, np.unique(first.reshape(-1, 4), axis=0))
        counts.append(count)
    moved = f3d.MapScene(recipe=replace(bound, camera=replace(bound.camera, azimuth_deg=200)))
    moved.render(tmp_path / "moved.png")
    assert not np.array_equal(first, np.asarray(Image.open(tmp_path / "moved.png")))
    bundle = tmp_path / "bundle.forge3d"
    scene.save_bundle(bundle)
    restored = f3d.MapScene.load_bundle(bundle)
    restored.render(tmp_path / "replay.png")
    np.testing.assert_array_equal(first, np.asarray(Image.open(tmp_path / "replay.png")))
    record = {"adapter": probe, "native_backend": scene.last_render_backend,
              "palette_pixel_counts": counts, "lighting_byte_identical": True, "bundle_replay_byte_identical": True}
    (tmp_path / "evidence.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if os.environ.get("FORGE3D_C01_ARTIFACT_DIR"):
        path = Path(os.environ["FORGE3D_C01_ARTIFACT_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "gpu-evidence.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


def test_recipe_surface_channels_execute_existing_native_maps(tmp_path):
    from forge3d.map_scene import _shared_terrain_render_context
    adapter = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert adapter["status"] == "ok" and not adapter.get("software_fallback"), adapter
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    normal_path = tmp_path / "normal.png"
    Image.fromarray(np.tile(np.array([[[218, 128, 218]]], np.uint8), (2, 2, 1))).save(normal_path)
    normal = f3d.MaterialTexture(str(normal_path), (16, 16), "linear", "normal", normal_convention="opengl")
    material = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("base", (150, 140, 130), normal=normal, roughness=.2),),
                                           presentation="physical")
    heights = np.zeros((32, 32), np.float32)
    base = f3d.SceneRecipe(f3d.TerrainSource(data=heights, crs=grid.crs),
        f3d.OrbitCamera(distance=150, azimuth_deg=140, elevation_deg=35),
        f3d.LightingPreset("custom", sun_direction=(1., .3, 0.), settings={"cli_params": {"camera_mode": "mesh:zup:north"}}),
        output=f3d.OutputSpec(width=256, height=256))
    records = {}
    def render(recipe, name):
        scene = f3d.MapScene(recipe=recipe)
        scene.render(tmp_path / (name + ".png"))
        assert scene.last_render_backend == "gpu_terrain"
        records[name] = scene.last_render_metadata
        return np.asarray(Image.open(tmp_path / (name + ".png")))
    bound = material.bind(base, grid)
    with _shared_terrain_render_context():
        first = render(bound, "mapped")
        # An in-place data asset edit must invalidate the colour/frame cache.
        Image.fromarray(np.tile(np.array([[[128, 128, 255]]], np.uint8), (2, 2, 1))).save(normal_path)
        flat = render(bound, "flat")
        assert not np.array_equal(first, flat)
        rough = render(material.with_layer("base", roughness=1).bind(base, grid), "rough")
        assert not np.array_equal(flat, rough)
        raster = render(material.bind_raster(base, np.ones_like(heights), grid, units="kg"), "raster")
        np.testing.assert_array_equal(flat, raster)
        raster_roughness = replace(material, layers=(replace(material.layers[0], normal=None, roughness=1),))
        mixed = raster_roughness.bind_raster(bound, np.ones_like(heights), grid, units="kg")
        np.testing.assert_array_equal(rough, render(mixed, "mixed"))
        restored_path = tmp_path / "surface.forge3d"
        scene = f3d.MapScene(recipe=bound)
        scene.save_bundle(restored_path)
        replay = render(f3d.MapScene.load_bundle(restored_path).recipe, "replay")
        np.testing.assert_array_equal(flat, replay)
    evidence = {"adapter": adapter, "normal_edit_changes_native_frame": True,
                "roughness_changes_native_frame": True, "terrain_raster_channel_parity": True,
                "mixed_raster_roughness_matches_native_terrain_override": True,
                "bundle_replay_byte_identical": True}
    if os.environ.get("FORGE3D_C01_ARTIFACT_DIR"):
        (Path(os.environ["FORGE3D_C01_ARTIFACT_DIR"]) / "surface-channel-evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf8")
