"""Phase 20: MapScene buildings are anchored to the terrain's own frame.

Milos's review found eight ways a building could look right in a unit test and
still be wrong in a render: two different spans, a grid the helper never read,
footprints in a different CRS, a requested texture silently ignored, a path
string treated differently from a mapping, a fixed-unit roof, a lost synthetic
fallback, and a pixel check that was an xfail. Every one is a gate here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import forge3d as f3d
from forge3d import map_scene

BOUNDS = [138.6, 35.26, 138.83, 35.46]
METRIC_SPAN = 18683.0


def _city_geojson(height: float = 40.0, roof: str = "flat") -> dict:
    # A ~400 m campus inside a 24 km DEM, in degrees.
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "campus-1",
                "properties": {
                    "height": height,
                    "building:material": "concrete",
                    "roof:shape": roof,
                },
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


def _country_scene(layers, width: int = 128, height: int = 128) -> f3d.MapScene:
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
                "bounds": BOUNDS,
                "terrain_span": METRIC_SPAN,
            },
        ),
        camera=f3d.OrbitCamera(
            target=(0.0, 0.0, 0.0),
            distance=24000.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            fov_deg=45.0,
        ),
        lighting=f3d.LightingPreset(
            name="studio-daylight",
            settings={"screen_space": {"enabled": True, "taa_enabled": True}},
        ),
        output=f3d.OutputSpec(width=width, height=height, format="png"),
        layers=list(layers),
    )


def _building_layer(tmp_path, **geojson_options) -> f3d.MapSceneBuildingLayer:
    path = tmp_path / "campus.geojson"
    path.write_text(json.dumps(_city_geojson(**geojson_options)), encoding="utf-8")
    return f3d.MapSceneBuildingLayer.from_geojson(path)


# --- source-format detection (mapping and string must agree) -----------------

def test_geojson_source_mapping_is_not_a_gltf_path():
    layer = f3d.MapSceneBuildingLayer(
        layer_id="campus",
        source={"path": "/data/campus.geojson", "source_format": "geojson"},
        support_level="supported",
        features=_city_geojson()["features"],
    )
    assert map_scene._building_gltf_path(layer) is None


def test_geojson_source_string_is_not_a_gltf_path():
    layer = f3d.MapSceneBuildingLayer(
        layer_id="campus",
        source="/data/campus.geojson",
        support_level="supported",
        metadata={"source_format": "geojson"},
        features=_city_geojson()["features"],
    )
    assert map_scene._building_gltf_path(layer) is None


def test_mesh_source_string_is_a_gltf_path():
    layer = f3d.MapSceneBuildingLayer(
        layer_id="landmark", source="/data/landmark.glb", support_level="supported",
    )
    assert map_scene._building_gltf_path(layer) == "/data/landmark.glb"


@pytest.mark.parametrize("source_form", ("mapping", "string"))
def test_geojson_buildings_render_with_screen_space_effects(tmp_path, source_form):
    path = tmp_path / "campus.geojson"
    path.write_text(json.dumps(_city_geojson()), encoding="utf-8")
    if source_form == "mapping":
        layer = f3d.MapSceneBuildingLayer.from_geojson(path)
    else:
        layer = f3d.MapSceneBuildingLayer(
            layer_id="campus", source=str(path), support_level="supported",
            metadata={"source_format": "geojson"}, features=_city_geojson()["features"],
        )
    assert map_scene._building_gltf_path(layer) is None
    scene = _country_scene([layer])
    assert scene.validate().status == "ok"
    out = tmp_path / f"taa-{source_form}.png"
    try:
        scene.render(str(out))
    except f3d.MapSceneNativeUnavailable as exc:
        pytest.fail(f"{source_form} GeoJSON buildings were refused under TAA: {exc}")
    assert out.exists()


def test_requested_texture_is_reported_instead_of_drawing_a_scalar_building(tmp_path):
    from PIL import Image

    texture = tmp_path / "facade.png"
    Image.new("RGBA", (4, 4), (180, 40, 40, 255)).save(texture)
    layer = _building_layer(tmp_path)
    layer.metadata = dict(layer.metadata or {})
    # Valid texture + declared UVs: validation must not block this, yet the
    # native building pass has no UV stream, so render must say so explicitly.
    layer.metadata["textured_material"] = {
        "albedo_texture": str(texture), "uv_available": True,
    }
    scene = _country_scene([layer])
    assert scene.validate().status == "ok"
    with pytest.raises(f3d.MapSceneNativeUnavailable) as exc:
        scene.render(str(tmp_path / "textured.png"))
    assert "textured building material" in str(exc.value)


# --- one terrain frame for the renderer and the buildings --------------------

def test_renderer_and_buildings_share_the_measured_span(tmp_path):
    scene = _country_scene([_building_layer(tmp_path)])
    bounds, crs, span = map_scene._terrain_frame(scene.recipe.terrain)
    assert bounds == tuple(BOUNDS)
    assert span == METRIC_SPAN
    render_span, (_near, far) = map_scene._terrain_render_span_and_clip(scene.recipe)
    assert render_span == span
    assert far > float(scene.recipe.camera.distance), "the camera must be inside the far clip"


def test_file_backed_projected_dem_supplies_bounds_and_span(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "ten_km.tif"
    transform = from_origin(500000.0, 5200000.0, 156.25, 156.25)
    with rasterio.open(
        path, "w", driver="GTiff", width=64, height=64, count=1,
        dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((64, 64), dtype=np.float32), 1)

    terrain = f3d.TerrainSource(path=str(path), crs="EPSG:32632")
    bounds, crs, span = map_scene._terrain_frame(terrain)
    assert crs == "EPSG:32632"
    assert bounds == pytest.approx((500000.0, 5190000.0, 510000.0, 5200000.0))
    assert span == pytest.approx(10000.0)


def test_file_backed_dem_places_a_small_footprint_at_its_grid_corner(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "ten_km.tif"
    transform = from_origin(500000.0, 5200000.0, 156.25, 156.25)
    with rasterio.open(
        path, "w", driver="GTiff", width=64, height=64, count=1,
        dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((64, 64), dtype=np.float32), 1)

    bounds, crs, span = map_scene._terrain_frame(
        f3d.TerrainSource(path=str(path), crs="EPSG:32632")
    )
    # A 100 m polygon in the northeast corner of the 10 km grid.
    geojson = tmp_path / "corner.geojson"
    geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "id": "corner",
            "properties": {"height": 100.0, "roof:shape": "flat"},
            "geometry": {"type": "Polygon", "coordinates": [[
                [509900.0, 5199900.0], [510000.0, 5199900.0],
                [510000.0, 5200000.0], [509900.0, 5200000.0],
                [509900.0, 5199900.0],
            ]]},
        }],
    }), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer.from_geojson(
        geojson, metadata={"source_crs": "EPSG:32632"}
    )
    batches = map_scene._native_building_mesh_batches_for_layers(
        [layer], bounds, span, crs
    )
    assert batches
    positions = np.asarray(batches[0]["positions"], dtype=np.float64)
    center = (positions[:, [0, 2]].min(axis=0) + positions[:, [0, 2]].max(axis=0)) * 0.5
    size = np.ptp(positions[:, [0, 2]], axis=0)
    # 100 m of a 10 km span is 0.017 scene units — not a country-wide quad.
    assert size == pytest.approx((0.017, 0.017), abs=1.0e-3)
    assert center == pytest.approx((0.8415, -0.8415), abs=0.02)


def test_footprints_are_transformed_into_the_terrain_crs(tmp_path):
    from pyproj import Transformer

    lonlat = [(138.727, 35.345), (138.731, 35.345), (138.731, 35.349)]
    easting, northing = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform(*lonlat[0])
    projected = map_scene._building_points_in_terrain_crs(
        lonlat, "EPSG:4326", "EPSG:3857", "campus"
    )
    assert projected[0] == pytest.approx((easting, northing))

    layer = _building_layer(tmp_path)
    layer.metadata = dict(layer.metadata or {})
    layer.metadata["source_crs"] = "EPSG:4326"
    terrain_point = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform(138.729, 35.347)
    west, south = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform(BOUNDS[0], BOUNDS[1])
    east, north = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform(BOUNDS[2], BOUNDS[3])
    batches = map_scene._native_building_mesh_batches_for_layers(
        [layer], (west, south, east, north), METRIC_SPAN, "EPSG:3857"
    )
    assert batches
    positions = np.asarray(batches[0]["positions"], dtype=np.float64)
    center = (positions[:, [0, 2]].min(axis=0) + positions[:, [0, 2]].max(axis=0)) * 0.5
    expected_x = (terrain_point[0] - west) / (east - west) * 1.7 - 0.85
    expected_z = (1.0 - (terrain_point[1] - south) / (north - south)) * 1.7 - 0.85
    assert center[0] == pytest.approx(expected_x, abs=0.02)
    assert center[1] == pytest.approx(expected_z, abs=0.02)


# --- height and roof scale ---------------------------------------------------

def test_country_scale_building_height_is_anchored_to_the_dem(tmp_path):
    layer = _building_layer(tmp_path)
    batches = map_scene._native_building_mesh_batches_for_layers(
        [layer], BOUNDS, METRIC_SPAN, "EPSG:4326"
    )
    assert batches
    assert 0.0 < batches[0]["wall_height"] < 0.01, batches[0]["wall_height"]


def test_synthetic_terrain_keeps_the_scene_unit_minimum(tmp_path):
    layer = _building_layer(tmp_path, height=2.0)
    batches = map_scene._native_building_mesh_batches_for_layers([layer])
    assert batches
    assert batches[0]["wall_height"] == pytest.approx(0.08)


def test_gabled_roof_scales_with_the_metric_walls(tmp_path):
    layer = _building_layer(tmp_path, roof="gabled")
    metric = map_scene._native_building_mesh_batches_for_layers(
        [layer], BOUNDS, METRIC_SPAN, "EPSG:4326"
    )
    synthetic = map_scene._native_building_mesh_batches_for_layers([layer])
    assert metric and synthetic
    metric_peak = float(np.max(np.asarray(metric[0]["positions"])[:, 1]))
    synthetic_peak = float(np.max(np.asarray(synthetic[0]["positions"])[:, 1]))
    assert metric_peak == pytest.approx(metric[0]["wall_height"] * 1.25, rel=1e-3)
    assert metric_peak < 0.01, "a fixed scene-unit roof minimum leaked into the metric path"
    assert synthetic_peak >= synthetic[0]["wall_height"] + 0.05


# --- physical-pixel gate -----------------------------------------------------

def test_country_scale_buildings_move_real_pixels(tmp_path):
    from PIL import Image

    layer = _building_layer(tmp_path)
    scene = _country_scene([])
    out_without = tmp_path / "without.png"
    scene.render(str(out_without))
    without = np.asarray(Image.open(out_without).convert("RGB"), dtype=np.int32)

    assert float(without.std()) > 5.0, "the terrain itself was not visible in this frame"

    scene = _country_scene([layer])
    out_with = tmp_path / "with.png"
    scene.render(str(out_with))
    with_buildings = np.asarray(Image.open(out_with).convert("RGB"), dtype=np.int32)

    diff = np.abs(without - with_buildings).sum(axis=2)
    ys, xs = np.nonzero(diff > 8)
    assert xs.size > 0, "the anchored building still moves no pixels"
    height, width = diff.shape
    # The campus is near the frame centre; far-edge changes would mean the
    # footprint was stretched across the DEM again.
    assert 0.15 * width < float(xs.mean()) < 0.85 * width
    assert 0.15 * height < float(ys.mean()) < 0.85 * height
