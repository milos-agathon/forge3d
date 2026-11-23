"""Phase 20: MapScene buildings are anchored to the terrain's own frame.

Milos's review found eight ways a building could look right in a unit test and
still be wrong in a render: two different spans, a grid the helper never read,
footprints in a different CRS, a requested texture silently ignored, a path
string treated differently from a mapping, a fixed-unit roof, a lost synthetic
fallback, and a pixel check that was an xfail. Every one is a gate here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from _terrain_runtime import terrain_rendering_available  # noqa: E402

engine_required = pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="no terrain-safe GPU for the engine's render path",
)

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

    # Channel spread passes on a flat RGB background; spatial variation does not.
    assert float(without[..., 0].astype(np.float64).std()) > 5.0, (
        "the terrain itself was not visible in this frame"
    )

    scene = _country_scene([layer])
    out_with = tmp_path / "with.png"
    scene.render(str(out_with))
    with_buildings = np.asarray(Image.open(out_with).convert("RGB"), dtype=np.int32)

    diff = np.abs(without - with_buildings).sum(axis=2)
    ys, xs = np.nonzero(diff > 8)
    assert xs.size > 0, "the anchored building still moves no pixels"
    height, width = diff.shape
    # The campus projects to x=71..73, y=54..57 on this fixture. A full-frame
    # change whose mean sits near the centre must not satisfy the gate.
    _assert_changed_inside_projected_campus(diff)


# --- Milos's second review ---------------------------------------------------

def test_spacing_metadata_and_projected_feet_are_metres_for_both_consumers(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    # In-memory DEM: only bounds + 100 m spacing, no explicit span/grid.
    scene = _country_scene([])
    terrain = f3d.TerrainSource(
        data=np.zeros((64, 64), np.float32), crs="EPSG:32632",
        metadata={"width": 64, "height": 64, "bounds": [500000, 5200000, 510000, 5200000],
                  "resolution": (156.25, 156.25)},
    )
    assert map_scene._terrain_frame(terrain)[2] == pytest.approx(10000.0)

    # File-backed grid in US survey feet: 10,000 ft is ~3,048 m, not 1 unit.
    path = tmp_path / "feet.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=64, height=64, count=1,
        dtype="float32", crs="EPSG:2263",
        transform=from_origin(1000000.0, 200000.0, 156.25, 156.25),
    ) as dataset:
        dataset.write(np.zeros((64, 64), np.float32), 1)
    feet = map_scene._terrain_frame(f3d.TerrainSource(path=str(path), crs="EPSG:2263"))
    assert feet[2] == pytest.approx(10000.0 * 1200 / 3937, rel=0.01)


def test_preset_camera_uses_metric_span_and_far_clip_includes_relief(tmp_path):
    scene = _country_scene([])
    terrain = f3d.TerrainSource(
        data=np.zeros((256, 256), np.float32), crs="EPSG:4326",
        metadata={"width": 256, "height": 256, "bounds": BOUNDS,
                  "resolution": [(BOUNDS[2] - BOUNDS[0]) / 256, (BOUNDS[3] - BOUNDS[1]) / 256],
                  "terrain_span": METRIC_SPAN},
    )
    current = scene.recipe.camera
    preset = map_scene._camera_from_preset(current, terrain, {"radius_scale": 2.4})
    assert preset.distance == pytest.approx(METRIC_SPAN * 2.4, rel=1e-3)

    relief_scene = _country_scene([])
    params = map_scene._build_mapscene_terrain_params(
        relief_scene.recipe, relief_scene.recipe.terrain.data, (128, 128), projection_only=True
    )
    relief = float(relief_scene.recipe.terrain.data.max()) - float(relief_scene.recipe.terrain.data.min())
    assert params.clip[1] >= 24000.0 * 1.2 + relief - 1.0


@engine_required
def test_plain_string_geojson_renders_on_projected_terrain(tmp_path):
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    west, south = tr.transform(BOUNDS[0], BOUNDS[1])
    east, north = tr.transform(BOUNDS[2], BOUNDS[3])
    path = tmp_path / "campus.geojson"
    path.write_text(json.dumps(_city_geojson()), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer(
        layer_id="campus", source=str(path), support_level="supported",
        features=_city_geojson()["features"],
    )
    assert map_scene._building_gltf_path(layer) is None
    assert map_scene._building_layer_crs(layer) == "EPSG:4326"
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.random.default_rng(3).integers(200, 3800, size=(256, 256)).astype(np.float32),
            crs="EPSG:3857", metadata={"width": 256, "height": 256,
                                       "bounds": [west, south, east, north],
                                       "terrain_span": METRIC_SPAN},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=24000.0,
                               azimuth_deg=90.0, elevation_deg=45.0, fov_deg=45.0),
        lighting=f3d.LightingPreset(name="studio-daylight",
                                    settings={"screen_space": {"enabled": True, "taa_enabled": True}}),
        output=f3d.OutputSpec(128, 128, format="png"), layers=[layer],
    )
    assert scene.validate().status == "ok"
    scene.render(str(tmp_path / "plain-string.png"))
    assert (tmp_path / "plain-string.png").exists()


def test_nonfinite_crs_transform_returns_a_layer_diagnostic():
    from pyproj import Transformer

    points = [(100.0, 0.0)]
    with pytest.raises(f3d.MapSceneNativeUnavailable) as exc:
        map_scene._building_points_in_terrain_crs(
            points, "+proj=longlat +datum=WGS84",
            "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84", "campus",
        )
    assert "nonfinite" in str(exc.value)


def test_rotated_affine_uses_both_axes_and_inverse_placement(tmp_path):
    import rasterio
    from affine import Affine

    path = tmp_path / "rotated.tif"
    transform = Affine.translation(500000.0, 5200000.0) * Affine.rotation(30) * Affine.scale(156.25, -156.25)
    with rasterio.open(
        path, "w", driver="GTiff", width=64, height=64, count=1,
        dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((64, 64), np.float32), 1)

    bounds, crs, span, affine, shape = map_scene._terrain_grid_frame(
        f3d.TerrainSource(path=str(path), crs="EPSG:32632")
    )
    assert shape == (64, 64)
    assert span == pytest.approx(10000.0, rel=1e-3)
    expected_point = transform * (19.2, 19.2)  # 30% into both axes
    placed = map_scene._building_point_to_scene(expected_point, bounds, affine, shape)
    assert placed == pytest.approx((0.3 * 1.7 - 0.85, 0.3 * 1.7 - 0.85), abs=1e-3)


def test_metric_height_is_not_clamped_but_synthetic_is(tmp_path):
    path = tmp_path / "tower.geojson"
    path.write_text(json.dumps(_city_geojson(height=120.0)), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer.from_geojson(path)
    metric = map_scene._native_building_mesh_batches_for_layers(
        [layer], BOUNDS, 100.0, "EPSG:4326"
    )
    small_path = tmp_path / "small.geojson"
    small_path.write_text(json.dumps(_city_geojson(height=2.0)), encoding="utf-8")
    small = f3d.MapSceneBuildingLayer.from_geojson(small_path)
    synthetic = map_scene._native_building_mesh_batches_for_layers([small])
    assert metric[0]["wall_height"] == pytest.approx(2.04, rel=1e-3)
    assert synthetic[0]["wall_height"] == pytest.approx(0.08)


# --- Milos's third review ----------------------------------------------------

def test_span_uses_data_shape_and_geodesic_distance():
    # Degree bounds with no explicit span: Geod.inv's third value is distance.
    terrain = f3d.TerrainSource(
        data=np.zeros((100, 100), np.float32), crs="EPSG:4326",
        metadata={"bounds": BOUNDS, "resolution": (0.0009, 0.0008)},
    )
    span = map_scene._terrain_frame(terrain)[2]
    assert 20_000.0 < span < 25_000.0, span

    # Width/height absent: the array shape supplies the grid dimensions.
    metres = f3d.TerrainSource(
        data=np.zeros((100, 100), np.float32), crs="EPSG:32632",
        metadata={"bounds": (500000, 5200000, 510000, 5201000),
                  "resolution": (100.0, 100.0)},
    )
    assert map_scene._terrain_frame(metres)[2] == pytest.approx(10000.0)


def test_unequal_pixel_affine_span(tmp_path):
    import rasterio
    from affine import Affine

    path = tmp_path / "rect.tif"
    transform = Affine.translation(500000.0, 5200000.0) * Affine.rotation(30) * Affine.scale(100.0, -25.0)
    with rasterio.open(
        path, "w", driver="GTiff", width=100, height=40, count=1,
        dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((40, 100), np.float32), 1)
    span = map_scene._terrain_frame(f3d.TerrainSource(path=str(path), crs="EPSG:32632"))[2]
    assert span == pytest.approx(10000.0, rel=1e-3)


def test_far_clip_covers_exaggerated_elevation():
    scene = _country_scene([])
    settings = dict(scene.recipe.lighting.settings or {})
    settings["exaggeration"] = 4.0
    scene.recipe.lighting.settings = settings
    params = map_scene._build_mapscene_terrain_params(
        scene.recipe, scene.recipe.terrain.data, (128, 128), projection_only=True
    )
    high = float(scene.recipe.terrain.data.max())
    assert params.clip[1] >= 24000.0 * 1.2 + high * 4.0 + 0.5 * METRIC_SPAN - 1.0


@engine_required
def test_mapping_path_without_format_renders_on_projected_terrain(tmp_path):
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    west, south = tr.transform(BOUNDS[0], BOUNDS[1])
    east, north = tr.transform(BOUNDS[2], BOUNDS[3])
    path = tmp_path / "campus.geojson"
    path.write_text(json.dumps(_city_geojson()), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer(
        layer_id="campus", source={"path": str(path)}, support_level="supported",
    )
    assert map_scene._building_layer_crs(layer) == "EPSG:4326"
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.random.default_rng(3).integers(200, 3800, size=(256, 256)).astype(np.float32),
            crs="EPSG:3857", metadata={"width": 256, "height": 256,
                                       "bounds": [west, south, east, north],
                                       "terrain_span": METRIC_SPAN},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=24000.0,
                               azimuth_deg=90.0, elevation_deg=45.0, fov_deg=45.0),
        lighting=f3d.LightingPreset(name="studio-daylight",
                                    settings={"screen_space": {"enabled": True, "taa_enabled": True}}),
        output=f3d.OutputSpec(128, 128, format="png"), layers=[layer],
    )
    assert scene.validate().status == "ok"
    try:
        scene.render(str(tmp_path / "mapping-path.png"))
    except f3d.MapSceneNativeUnavailable as exc:
        pytest.fail(f"mapping path did not reach the shared pass: {exc}")


def _assert_changed_inside_projected_campus(diff: np.ndarray) -> None:
    ys, xs = np.nonzero(diff > 8)
    assert xs.size > 0, "the building moved no pixels"
    # Projected campus: x=71..73, y=54..57, with rasterization tolerance.
    assert 68 <= int(xs.min()) and int(xs.max()) <= 76, (xs.min(), xs.max())
    assert 51 <= int(ys.min()) and int(ys.max()) <= 60, (ys.min(), ys.max())


def test_displaced_footprint_fails_the_pixel_gate():
    diff = np.zeros((128, 128), dtype=np.int32)
    # Five pixels west of the projected campus, as Milos's mutation produced.
    for x, y in ((66, 54), (66, 55), (67, 54), (67, 55), (67, 56)):
        diff[y, x] = 12
    with pytest.raises(AssertionError):
        _assert_changed_inside_projected_campus(diff)


def test_geodesic_span_uses_distance_not_azimuth(monkeypatch):
    class Geod:
        def __init__(self, *args, **kwargs):
            pass

        def inv(self, *args):
            # East-west call has equal south/north arguments; the bad code
            # would take the azimuth 89.93 instead of the distance.
            if args[1] == args[3]:
                return (89.93, 180.0, 22189.44)
            return (0.0, 0.0, 12345.0)

    import pyproj

    monkeypatch.setattr(pyproj, "Geod", Geod)
    terrain = f3d.TerrainSource(
        data=np.zeros((100, 100), np.float32), crs="EPSG:4326",
        metadata={"bounds": BOUNDS, "resolution": (0.0009, 0.0008)},
    )
    assert map_scene._terrain_frame(terrain)[2] == pytest.approx(22189.44)


# --- Milos's fourth review ---------------------------------------------------

def test_inline_dem_shape_matches_explicit_dimensions(tmp_path):
    from affine import Affine

    path = tmp_path / "inline.tif"
    transform = Affine.translation(500000.0, 5200000.0) * Affine.scale(100.0, -100.0)
    import rasterio

    with rasterio.open(
        path, "w", driver="GTiff", width=100, height=100, count=1,
        dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((100, 100), np.float32), 1)

    inline = f3d.TerrainSource(
        data=np.zeros((100, 100), np.float32), crs="EPSG:32632",
        metadata={"transform": transform},
    )
    explicit = f3d.TerrainSource(
        data=np.zeros((100, 100), np.float32), crs="EPSG:32632",
        metadata={"transform": transform, "width": 100, "height": 100},
    )
    frame_inline = map_scene._terrain_grid_frame(inline)
    frame_explicit = map_scene._terrain_grid_frame(explicit)
    assert frame_inline[2] == frame_explicit[2] == pytest.approx(10000.0)
    assert frame_inline[4] == frame_explicit[4] == (100, 100)

    path_geo = tmp_path / "corner.geojson"
    path_geo.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "id": "small",
            "properties": {"height": 100.0, "roof:shape": "flat"},
            "geometry": {"type": "Polygon", "coordinates": [[
                [509900.0, 5199900.0], [510000.0, 5199900.0],
                [510000.0, 5200000.0], [509900.0, 5200000.0],
                [509900.0, 5199900.0],
            ]]},
        }],
    }), encoding="utf-8")
    layer = f3d.MapSceneBuildingLayer.from_geojson(path_geo, metadata={"source_crs": "EPSG:32632"})
    for frame in (frame_inline, frame_explicit):
        batches = map_scene._native_building_mesh_batches_for_layers(
            [layer], frame[0], frame[2], frame[1], frame[3], frame[4]
        )
        positions = np.asarray(batches[0]["positions"], dtype=np.float64)
        assert np.ptp(positions[:, [0, 2]], axis=0) == pytest.approx((0.017, 0.017), abs=1e-3)


def test_rotated_geographic_grid_measures_raster_edges(tmp_path):
    from affine import Affine

    path = tmp_path / "geo_rot.tif"
    transform = Affine.translation(0.0, 0.0) * Affine.rotation(45) * Affine.scale(0.00078, -0.00078)
    import rasterio

    with rasterio.open(
        path, "w", driver="GTiff", width=128, height=128, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((128, 128), np.float32), 1)

    span = map_scene._terrain_frame(f3d.TerrainSource(path=str(path), crs="EPSG:4326"))[2]
    assert span == pytest.approx(11095.0, rel=0.05)


@engine_required
def test_mixed_elevation_stays_inside_the_far_plane(tmp_path):
    scene = _country_scene([])
    dem = np.full((128, 128), -1800.0, dtype=np.float32)
    dem[:64, :] = 1800.0
    scene.recipe.terrain.data = dem
    settings = dict(scene.recipe.lighting.settings or {})
    settings["exaggeration"] = 4.0
    scene.recipe.lighting.settings = settings

    params = map_scene._build_mapscene_terrain_params(
        scene.recipe, dem, (128, 128), projection_only=True
    )
    required = 24000.0 * 1.2 + (1800.0 - (-1800.0)) * 4.0 + 0.5 * METRIC_SPAN
    assert params.clip[1] >= required - 1.0

    out = tmp_path / "mixed.png"
    scene.render(str(out))
    from PIL import Image

    frame = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32)
    assert float(frame[:64].std()) > 1.0, "the high half did not render"
    assert float(frame[64:].std()) > 1.0, "the low half was clipped to background"
