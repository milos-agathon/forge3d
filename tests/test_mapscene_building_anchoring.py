"""Phase 20: MapScene buildings must be anchored to the terrain, not stretched
across it, and a GeoJSON layer must not be refused as textured geometry.

Both failures were measured on a country-scale DEM: validate() reported
`buildings.mapscene_render: supported`, then render() raised
MapSceneNativeUnavailable, and with TAA off the layer changed 0 of 262,144
samples because a 12 m building was rendered as a fraction of a 45 m scene.
The tests below pin the two fixes without needing a physical GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import forge3d as f3d
from forge3d import map_scene


def _city_geojson() -> dict:
    # A 1 km campus at the centre of a 24 km DEM: in metres, not scene units.
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "campus-1",
                "properties": {"height": 40, "building:material": "concrete", "roof:shape": "flat"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [138.727, 35.345],
                            [138.731, 35.345],
                            [138.731, 35.349],
                            [138.727, 35.349],
                            [138.727, 35.345],
                        ]
                    ],
                },
            }
        ],
    }


def _country_scene(layer, width: int = 128, height: int = 128) -> f3d.MapScene:
    rng = np.random.default_rng(3)
    dem = rng.integers(200, 3800, size=(256, 256)).astype(np.float32)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=dem,
            crs="EPSG:4326",
            metadata={
                "source_id": "country-dem",
                "width": 256,
                "height": 256,
                "bounds": [138.6, 35.26, 138.83, 35.46],
                "terrain_span": 18683.0,
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=24000.0,
                               azimuth_deg=90.0, elevation_deg=45.0, fov_deg=45.0),
        lighting=f3d.LightingPreset(
            name="studio-daylight",
            settings={"screen_space": {"enabled": True, "taa_enabled": True}},
        ),
        output=f3d.OutputSpec(width=width, height=height, format="png"),
        layers=[layer],
    )


def _building_layer(tmp_path) -> f3d.MapSceneBuildingLayer:
    path = tmp_path / "campus.geojson"
    path.write_text(json.dumps(_city_geojson()), encoding="utf-8")
    return f3d.MapSceneBuildingLayer.from_geojson(path)


def test_geojson_source_is_not_a_gltf_path():
    layer = f3d.MapSceneBuildingLayer(
        layer_id="campus",
        source={"path": "/data/campus.geojson", "source_format": "geojson"},
        support_level="supported",
        features=_city_geojson()["features"],
    )
    assert map_scene._building_gltf_path(layer) is None


def test_geojson_buildings_render_with_screen_space_effects(tmp_path):
    layer = _building_layer(tmp_path)
    scene = _country_scene(layer)
    assert scene.validate().status == "ok"
    out = tmp_path / "taa.png"
    try:
        scene.render(str(out))
    except f3d.MapSceneNativeUnavailable as exc:
        pytest.fail(f"supported GeoJSON buildings were refused under TAA: {exc}")
    assert out.exists()


def test_country_scale_building_height_is_anchored_to_the_dem(tmp_path):
    layer = _building_layer(tmp_path)
    # The batch builder must expose a real-world height for a 40 m building,
    # not the old 40/45 scene fraction. 18 km span, 1.7 scene units wide.
    batches = map_scene._native_building_mesh_batches_for_layers(
        [layer], terrain_bounds=[138.6, 35.26, 138.83, 35.46], metric_span=18683.0
    )
    assert batches
    # 40 m / 18.6 km * 1.7 scene units, small but nonzero.
    assert 0.0 < batches[0]["wall_height"] < 0.01, batches[0]["wall_height"]


def test_country_scale_buildings_move_real_pixels(tmp_path):
    from PIL import Image

    def shot(layers, name):
        # Same DEM and camera as _country_scene, with or without the layer.
        import numpy as np

        rng = np.random.default_rng(3)
        dem = rng.integers(200, 3800, size=(256, 256)).astype(np.float32)
        scene = f3d.MapScene(
            terrain=f3d.TerrainSource(
                data=dem,
                crs="EPSG:4326",
                metadata={
                    "source_id": "country-dem",
                    "width": 256,
                    "height": 256,
                    "bounds": [138.6, 35.26, 138.83, 35.46],
                    "terrain_span": 18683.0,
                },
            ),
            camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=24000.0,
                                   azimuth_deg=90.0, elevation_deg=45.0, fov_deg=45.0),
            lighting=f3d.LightingPreset(
                name="studio-daylight",
                settings={"screen_space": {"enabled": True, "taa_enabled": True}},
            ),
            output=f3d.OutputSpec(width=128, height=128, format="png"),
            layers=layers,
        )
        out = tmp_path / name
        scene.render(str(out))
        return np.asarray(Image.open(out).convert("RGB"), dtype=np.int32)

    layer = _building_layer(tmp_path)
    without = shot([], "without.png")
    with_buildings = shot([layer], "with.png")
    diff = np.abs(without - with_buildings).sum(axis=2)
    changed = int((diff > 8).sum())
    # Before anchoring this was 0 of 16,384 samples. Six is physically right:
    # a 40 m building in a 24 km DEM at 128² is sub-pixel in height.
    assert changed > 0, "the anchored building still moves no pixels"
