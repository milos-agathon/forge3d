"""C05: semantic/local routes, independent coordinates, masks and reuse."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import forge3d as f3d
from forge3d.preparation import (
    PreparationError, PreparationGrid as Grid, PreparationSource as Source,
    PreparationRecipe as Recipe, PreparedScene, prepare_scene,
)


def grid(width=4, height=4, x=100_000, y=440_000, spacing=100):
    return Grid("EPSG:28992", width, height, (x, spacing, 0, y, 0, -spacing))


def raster(path, values, source_grid, *, nodata=None):
    values = np.asarray(values)
    bands = values[None, ...] if values.ndim == 2 else values
    with rasterio.open(path, "w", driver="GTiff", width=source_grid.width, height=source_grid.height,
                       count=bands.shape[0], dtype=values.dtype, crs=source_grid.crs,
                       transform=source_grid.transform, nodata=nodata) as ds:
        ds.write(bands)
    return path


def elevation(tmp_path, source_grid=None, values=None):
    source_grid = source_grid or grid()
    values = np.ones((source_grid.height, source_grid.width), dtype=np.float64) if values is None else values
    return Source("dem", raster(tmp_path / "dem.tif", values, source_grid), "elevation",
                  units="m", vertical_reference="NAP", metadata={"product": "controlled fixture", "edition": "one"})


def test_prepare_shifted_category_and_count_semantics(tmp_path):
    target = grid()
    dem = elevation(tmp_path)
    shifted = grid(x=100_100)
    classes = np.tile(np.arange(4, dtype=np.int16), (4, 1))
    classes[1, 2] = -99
    thematic = Source("classes", raster(tmp_path / "classes.tif", classes, shifted, nodata=-99), "categorical")
    counts = np.arange(16, dtype=np.int64).reshape(4, 4) + 2**53
    np.save(tmp_path / "counts.npy", counts)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    np.save(tmp_path / "disclosure.npy", mask)
    population = Source("population", tmp_path / "counts.npy", "count", grid=shifted,
                        units="people/cell", mask_path=tmp_path / "disclosure.npy")
    prepared = prepare_scene(Recipe(target, (dem, thematic, population), "NAP"))
    category = prepared.layers["classes"]
    assert category.data.dtype == classes.dtype
    assert not category.valid_mask[:, 0].any()
    np.testing.assert_array_equal(category.data[0, 1:], [0, 1, 2])
    assert not category.valid_mask[1, 3]
    assert category.valid_mask[0, 1]  # class zero is data, not nodata
    count = prepared.layers["population"]
    np.testing.assert_array_equal(count.data, counts)
    np.testing.assert_array_equal(count.valid_mask, mask)
    assert count.grid.same_as(shifted)
    assert count.metadata["operations"][0]["operation"] == "retain_native_count_cells"
    assert prepared.report["render_verified"] is False


def test_float64_continuous_values_masks_and_warp(tmp_path):
    target = grid()
    source_grid = grid(x=100_100)
    values = 1e12 + np.arange(16, dtype=np.float64).reshape(4, 4) / 8
    source = Source("measurement", raster(tmp_path / "measure.tif", values, source_grid),
                    "continuous", units="m", resampling="nearest")
    prepared = prepare_scene(Recipe(target, (elevation(tmp_path), source), "NAP"))
    layer = prepared.layers["measurement"]
    assert layer.data.dtype == np.float64
    np.testing.assert_array_equal(layer.data[:, 1:], values[:, :3])
    assert not layer.valid_mask[:, 0].any()


def test_raster_mask_does_not_leak_into_bilinear(tmp_path):
    src = grid(width=2, height=2)
    target = grid(width=4, height=4, spacing=50)
    data = np.array([[10, -9999], [10, 10]], dtype=np.float64)
    dem = replace(elevation(tmp_path, src, data), nodata=-9999)
    prepared = prepare_scene(Recipe(target, (dem,), "NAP"))
    layer = prepared.layers["dem"]
    np.testing.assert_allclose(layer.data[layer.valid_mask], 10)
    assert not layer.valid_mask.all()


def test_independent_geographic_control_and_shift_detection(tmp_path):
    # Published EPSG:3857 spherical equation; not computed via preparation.
    target = Grid("EPSG:3857", 4, 4, (0, 50_000, 0, 200_000, 0, -50_000))
    src = Grid("EPSG:4326", 4, 4, (0, .5, 0, 2, 0, -.5))
    dem = replace(elevation(tmp_path, src), vertical_reference="ellipsoidal")
    expected_x = 6378137 * np.deg2rad(.25)
    expected_y = 6378137 * np.log(np.tan(np.pi / 4 + np.deg2rad(1.75) / 2))
    # Metre tolerance is MENSURA's existing 1 mm transform conformance contract.
    checkpoint = {"source_id": "dem", "source_pixel": [0, 0],
                  "expected_target_xy": [expected_x, expected_y], "tolerance": .001}
    recipe = Recipe(target, (dem,), "ellipsoidal", (checkpoint,))
    prepared = prepare_scene(recipe)
    assert prepared.report["checkpoints"][0]["residual"] <= .001
    translated = replace(src, geotransform=(.5, .5, 0, 2, 0, -.5))
    raster(Path(dem.path), np.ones((4, 4), dtype=np.float32), translated)
    with pytest.raises(PreparationError, match="checkpoint_failed"):
        prepare_scene(recipe)


@pytest.mark.parametrize("datum", ["NAP", "LN02", "chart_datum"])
def test_unsupported_vertical_conversions_refuse(tmp_path, datum):
    dem = replace(elevation(tmp_path), vertical_reference=datum)
    with pytest.raises(PreparationError, match="unsupported_vertical_transform") as error:
        prepare_scene(Recipe(grid(), (dem,), "ellipsoidal"))
    assert "externally" in error.value.diagnostic.remediation


def test_height_unit_conversion_preserves_vertical_datum(tmp_path):
    dem = replace(elevation(tmp_path, values=np.full((4, 4), -10., dtype=np.float64)), units="ft")
    layer = prepare_scene(Recipe(grid(), (dem,), "NAP")).layers["dem"]
    np.testing.assert_allclose(layer.data, -3.048)
    assert layer.metadata["vertical_reference"] == "NAP"
    assert layer.metadata["units"] == "m"


@pytest.mark.parametrize("kwargs,code", [
    ({"vertical_reference": None}, "missing_vertical_reference"),
    ({"units": None}, "missing_units"),
    ({"units": "furlong"}, "unsupported_height_units"),
])
def test_actionable_missing_semantics(tmp_path, kwargs, code):
    with pytest.raises(PreparationError, match=code) as error:
        replace(elevation(tmp_path), **kwargs)
    assert error.value.diagnostic.layer_id == "dem"
    assert error.value.diagnostic.remediation


@pytest.mark.parametrize("kind,method,code", [
    ("categorical", "bilinear", "category_resampling_refused"),
    ("count", "nearest", "count_resampling_refused"),
])
def test_discrete_resampling_cannot_be_requested(tmp_path, kind, method, code):
    values = np.ones((4, 4), dtype=np.int32)
    source = Source("other", raster(tmp_path / "other.tif", values, grid()), kind,
                    units="people/cell", resampling=method)
    with pytest.raises(PreparationError, match=code):
        prepare_scene(Recipe(grid(), (elevation(tmp_path), source), "NAP"))


def test_vectors_observations_keep_properties_and_use_xy(tmp_path):
    target = Grid("EPSG:3857", 4, 4, (0, 50_000, 0, 200_000, 0, -50_000))
    path = tmp_path / "observations.geojson"
    props = {"value": 5, "date": "2024-01-01", "withheld": False}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "id": "one",
        "bbox": [0, 0, 1, 1], "properties": props, "geometry": {"type": "Point", "coordinates": [1, 1]}}]}))
    observation = Source("observations", path, "observation", crs="EPSG:4326", units="m")
    layer = prepare_scene(Recipe(target, (elevation(tmp_path, target), observation), "NAP")).layers["observations"]
    feature = layer.features[0]
    assert feature["properties"] == props
    assert feature["id"] == "one"
    assert "bbox" not in feature
    assert feature["geometry"]["coordinates"][0] == pytest.approx(111319.49079327357)


@pytest.mark.parametrize("kind", ["MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"])
def test_geojson_multi_geometries_transform(tmp_path, kind):
    from forge3d.preparation import _transform_geometry_checked
    coordinates = {"MultiPoint": [[1, 1]], "MultiLineString": [[[1, 1], [2, 2]]],
                   "MultiPolygon": [[[[1, 1], [2, 2], [2, 1], [1, 1]]]]}
    geometry = ({"type": kind, "geometries": [{"type": "Point", "coordinates": [1, 1]}]}
                if kind == "GeometryCollection" else {"type": kind, "coordinates": coordinates[kind]})
    transformed = _transform_geometry_checked(geometry, "EPSG:4326", "EPSG:3857")
    assert transformed["type"] == kind
    assert "111319" in json.dumps(transformed)


def test_snapshot_reuse_integrity_recipe_revision_and_scene_bundle(tmp_path):
    dem = elevation(tmp_path)
    recipe = Recipe(grid(), (dem,), "NAP")
    recipe_path = tmp_path / "recipe.json"
    recipe.save(recipe_path)
    prepared = prepare_scene(Recipe.load(recipe_path))
    directory = prepared.save(tmp_path / "prepared")
    loaded = PreparedScene.load(directory)
    np.testing.assert_array_equal(loaded.layers["dem"].data, prepared.layers["dem"].data)
    assert loaded.report == prepared.report
    assert loaded.coverage_preview(tmp_path / "coverage.html").is_file()
    scene = loaded.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(width=64, height=64))
    assert scene.recipe.terrain.metadata["preparation"] == prepared.report
    scene.save_bundle(tmp_path / "scene.bundle")
    restored = f3d.MapScene.load_bundle(tmp_path / "scene.bundle")
    assert restored.recipe.terrain.metadata["preparation"] == prepared.report
    # A new edition is a recipe revision, not a change to workflow internals.
    raster(Path(dem.path), np.full((4, 4), 3, dtype=np.float64), grid())
    revised = prepare_scene(Recipe.load(recipe_path))
    assert revised.layers["dem"].metadata["identity"]["sha256"] != prepared.layers["dem"].metadata["identity"]["sha256"]
    # Snapshot still reuses the original assets even after source revision.
    np.testing.assert_array_equal(PreparedScene.load(directory).layers["dem"].data, 1)
    data_file = next(directory.glob("*.data.npy"))
    data_file.write_bytes(b"corrupt")
    with pytest.raises(PreparationError, match="prepared_asset_integrity"):
        PreparedScene.load(directory)


def test_fixed_styles_preserve_category_zero_and_transparency(tmp_path):
    classes = np.array([[0, 1, -1, 0]] * 4, dtype=np.int16)
    source = Source("classes", raster(tmp_path / "classes.tif", classes, grid(), nodata=-1), "categorical")
    prepared = prepare_scene(Recipe(grid(), (elevation(tmp_path), source), "NAP"))
    prepared.save(tmp_path / "prepared")
    scene = prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(width=64, height=64),
                              styles={"classes": {"classes": {0: [10, 20, 30, 255], 1: [50, 60, 70, 255]}}})
    overlay = scene.recipe.layers[0]
    with rasterio.open(overlay.path) as ds:
        rgba = ds.read()
    np.testing.assert_array_equal(rgba[:, 0, 0], [10, 20, 30, 255])
    np.testing.assert_array_equal(rgba[:, 0, 2], [0, 0, 0, 0])
    assert overlay.metadata["source_id"] == "classes"
    # Four-column RGBA GeoTIFFs must not confuse band and image axes.
    from forge3d.map_scene import _load_native_raster_overlay
    loaded = _load_native_raster_overlay(overlay, target_grid=grid().to_dict())
    np.testing.assert_array_equal(loaded[0, 0], [10, 20, 30, 255])
    np.testing.assert_array_equal(loaded[0, 2], [0, 0, 0, 0])


def test_missing_crs_and_nonoverlap_refuse(tmp_path):
    dem = elevation(tmp_path)
    far = grid(x=1_000_000)
    with pytest.raises(PreparationError, match="no_coverage"):
        prepare_scene(Recipe(far, (dem,), "NAP"))
    with rasterio.open(tmp_path / "unlocated.tif", "w", driver="GTiff", width=4, height=4,
                       count=1, dtype="float32", transform=from_origin(1, 2, 1, 1)) as ds:
        ds.write(np.ones((1, 4, 4), dtype=np.float32))
    with pytest.raises(PreparationError, match="missing_crs"):
        prepare_scene(Recipe(grid(), (replace(dem, path=tmp_path / "unlocated.tif"),), "NAP"))


def test_public_exports():
    for name in ("PreparationGrid", "PreparationSource", "PreparationRecipe", "PreparedScene", "prepare_scene",
                 "LandscapeLayout", "export_landscape", "verify_landscape_import"):
        assert name in f3d.__all__
        assert getattr(f3d, name) is not None


def test_egm96_route_uses_published_height_anchor_and_inverse(tmp_path):
    anchors = Path("tests/data/egm96_test_values.txt").read_text().splitlines()
    lat, lon, n_ref, _ = next(line.split() for line in anchors if line.strip() and not line.startswith("#"))
    lat, lon, n_ref = float(lat), float(lon), float(n_ref)
    source_grid = Grid("EPSG:4326", 1, 1, (lon - .005, .01, 0, lat + .005, 0, -.01))
    dem = replace(elevation(tmp_path, source_grid, np.array([[100.]], dtype=np.float64)), vertical_reference="orthometric_egm96")
    prepared = prepare_scene(Recipe(source_grid, (dem,), "ellipsoidal"))
    # Existing EGM96 degree-120 contract: <0.5 m versus NGA degree-360 anchors.
    assert abs(prepared.layers["dem"].data[0, 0] - (100 + n_ref)) < .5
    reverse = replace(dem, path=raster(tmp_path / "ellipsoid.tif", prepared.layers["dem"].data, source_grid),
                      vertical_reference="ellipsoidal")
    restored = prepare_scene(Recipe(source_grid, (reverse,), "orthometric_egm96"))
    np.testing.assert_allclose(restored.layers["dem"].data, 100, rtol=0, atol=1e-6)


def test_warped_byte_imagery_displays_and_reuses_style_settings(tmp_path):
    rgb = np.full((3, 4, 4), 23, dtype=np.uint8)
    source = Source("photo", raster(tmp_path / "rgb.tif", rgb, grid(x=100_100)), "imagery")
    recipe = Recipe(grid(), (elevation(tmp_path), source), "NAP", display_styles={})
    prepared = prepare_scene(recipe)
    assert prepared.layers["photo"].data.dtype == np.uint8
    prepared.save(tmp_path / "prepared")
    scene = PreparedScene.load(tmp_path / "prepared").to_scene(
        lighting=f3d.LightingPreset(), output=f3d.OutputSpec(width=64, height=64))
    with rasterio.open(scene.recipe.layers[0].path) as ds:
        display = ds.read()
    assert not display[3, :, 0].any()
    np.testing.assert_array_equal(display[:3, :, 1:], 23)


def test_effective_style_recipe_and_relative_sources_survive_snapshot(tmp_path):
    dem = elevation(tmp_path)
    classes = Source("classes", raster(tmp_path / "classes.tif", np.zeros((4, 4), dtype=np.uint8), grid()), "categorical")
    styles = {"classes": {"classes": {"0": [10, 20, 30, 255]}}}
    recipe = Recipe(grid(), (replace(dem, path="dem.tif"), classes), "NAP", display_styles=styles)
    prepared = prepare_scene(recipe, base_dir=tmp_path)
    prepared.save(tmp_path / "prepared")
    loaded = PreparedScene.load(tmp_path / "prepared")
    assert loaded.recipe.display_styles == styles
    executable = Recipe.load(tmp_path / "prepared" / "recipe.json")
    rerun = prepare_scene(executable)
    assert rerun.report == prepared.report
    loaded.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(width=64, height=64))


@pytest.mark.parametrize("route", ["https://example.org/dem.tif", "/vsicurl/dem.tif", "unhandled.nc"])
def test_unsupported_routes_fail_before_opening(tmp_path, route):
    recipe = Recipe(grid(), (replace(elevation(tmp_path), path=route),), "NAP")
    with pytest.raises(PreparationError, match="unsupported_source_route"):
        prepare_scene(recipe)


def test_viewer_assets_register_count_display_and_geographic_vectors(tmp_path):
    target = grid()
    count_grid = grid(x=100_100)
    values = np.ones((4, 4), dtype=np.int64) * 2**54
    np.save(tmp_path / "counts.npy", values)
    counts = Source("counts", tmp_path / "counts.npy", "count", grid=count_grid, units="people/cell")
    vector_path = tmp_path / "point.geojson"
    xy = target.xy(1, 2)
    vector_path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"value": 2}, "geometry": {"type": "Point", "coordinates": xy}}]}))
    vector = Source("point", vector_path, "vector", crs=target.crs)
    styles = {"counts": {"range": [0, 2**55], "colors": [[0, 0, 0, 0], [255, 0, 0, 255]]},
              "point": {"color": [0, 0, 255, 255]}}
    prepared = prepare_scene(Recipe(target, (elevation(tmp_path), counts, vector), "NAP", display_styles=styles))
    assets = prepared.viewer_assets(tmp_path / "viewer")
    from PIL import Image
    with rasterio.open(assets["terrain_path"]) as terrain:
        assert terrain.crs == rasterio.crs.CRS.from_user_input(target.crs)
        assert terrain.transform.to_gdal() == target.geotransform
        assert terrain.tags()["vertical_reference"] == "NAP"
    image = np.array(Image.open(assets["overlays"][0]["path"]))
    assert not image[:, 0, 3].any()
    assert np.all(image[:, 1:, 3] > 0)
    point = np.array(Image.open(assets["overlays"][1]["path"]))
    np.testing.assert_array_equal(point[2, 1], [0, 0, 255, 255])
    assert assets["viewer_render_verified"] is False
    np.testing.assert_array_equal(prepared.layers["counts"].data, values)
    prepared.save(tmp_path / "prepared")
    scene = prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(64,64))
    layer = scene.recipe.layers[1]
    from forge3d.map_scene import _vector_point_to_pixel
    assert _vector_point_to_pixel(xy, layer, 65, 65) == (24, 40)
    assert layer.features[0]["geometry"]["coordinates"] == list(xy)


def test_encoded_measurement_nodata_is_checked_before_scale_offset(tmp_path):
    # Raw 1 -> decoded -9999 is valid; raw -9999 is missing.
    values = np.ones((4,4), dtype=np.int16)
    values[0,0] = -9999
    path = raster(tmp_path / "encoded.tif", values, grid())
    with rasterio.open(path, "r+") as ds:
        ds.scales = (2.,)
        ds.offsets = (-10001.,)
        ds.set_band_unit(1, "metre")
    source = Source("measure", path, "continuous", units="m", nodata=-9999)
    layer = prepare_scene(Recipe(grid(), (elevation(tmp_path),source), "NAP")).layers["measure"]
    assert not layer.valid_mask[0,0]
    assert layer.valid_mask[0,1]
    assert layer.data[0,1] == -9999


@pytest.mark.parametrize("mismatch,code", [("units","conflicting_units"),("datum","conflicting_vertical_reference")])
def test_conflicting_file_semantics_are_not_silently_overridden(tmp_path,mismatch,code):
    dem = elevation(tmp_path)
    with rasterio.open(dem.path, "r+") as ds:
        if mismatch == "units":
            ds.set_band_unit(1,"ft")
        else:
            ds.update_tags(vertical_reference="LN02")
    with pytest.raises(PreparationError,match=code):
        prepare_scene(Recipe(grid(),(dem,),"NAP"))


def test_compound_vertical_crs_refuses_implicit_warp():
    with pytest.raises(PreparationError,match="compound_crs_requires_preparation"):
        Grid("EPSG:4326+3855",2,2,(0,1,0,2,0,-1))


@pytest.mark.parametrize("crs", ["EPSG:4979", "EPSG:4978"])
def test_three_dimensional_crs_refuses_discarding_an_axis(crs):
    with pytest.raises(PreparationError, match="unsupported_grid_crs"):
        Grid(crs, 2, 2, (0, 1, 0, 2, 0, -1))


def test_recipe_canonicalizes_numeric_negative_zero_without_changing_text(tmp_path):
    source_grid = Grid("EPSG:28992", 2, 2, (0, 1, -0.0, 2, -0.0, -1))
    source = Source("terrain", tmp_path / "height.npy", "elevation", grid=source_grid,
                    units="m", vertical_reference="NAP",
                    metadata={"zero": -0.0, "nested": [-0.0], "literal": "-0.0"})
    recipe = Recipe(source_grid, (source,), "NAP")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    recipe.save(first)
    decoded = json.loads(first.read_text(encoding="utf-8"))
    assert not np.signbit(decoded["target_grid"]["geotransform"][2])
    assert not np.signbit(decoded["sources"][0]["metadata"]["zero"])
    assert not np.signbit(decoded["sources"][0]["metadata"]["nested"][0])
    assert decoded["sources"][0]["metadata"]["literal"] == "-0.0"
    Recipe.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("value,dtype", [(2**53+1,"int64"), (2**63+1,"uint64"), (-(2**53)-1,"int64")])
def test_large_category_ids_refuse_before_float_scratch(tmp_path,value,dtype):
    np.save(tmp_path / "classes.npy",np.full((4,4),value,dtype=dtype))
    source=Source("classes",tmp_path / "classes.npy","categorical",grid=grid(x=100_100))
    with pytest.raises(PreparationError,match="category_precision"):
        prepare_scene(Recipe(grid(),(elevation(tmp_path),source),"NAP"))


@pytest.mark.parametrize("mutation", ["target", "source_typo", "root_typo", "tolerance", "checkpoint_typo", "xy", "root_type"])
def test_malformed_recipes_have_structured_diagnostics(tmp_path, mutation):
    value = Recipe(grid(), (elevation(tmp_path),), "NAP", checkpoints=[
        {"source_id": "dem", "source_pixel": [0, 0], "expected_target_xy": grid().xy(0, 0), "tolerance": 0}]).to_dict()
    if mutation == "target": del value["target_grid"]
    elif mutation == "source_typo": value["sources"][0]["vertcal_reference"] = "NAP"
    elif mutation == "root_typo": value["target_gird"] = value.pop("target_grid")
    elif mutation == "tolerance": del value["checkpoints"][0]["tolerance"]
    elif mutation == "checkpoint_typo": value["checkpoints"][0]["tolernce"] = 0
    elif mutation == "xy": value["checkpoints"][0]["source_pixel"] = [0]
    else: value = []
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PreparationError) as caught:
        Recipe.load(path)
    assert caught.value.diagnostic.code == "invalid_recipe"
    assert caught.value.diagnostic.remediation


def test_invalid_json_is_a_recipe_diagnostic(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"sources":', encoding="utf-8")
    with pytest.raises(PreparationError, match="invalid_recipe"):
        Recipe.load(path)


def test_oblique_scene_rasters_drape_exactly_like_registered_viewer_textures(tmp_path):
    from forge3d._map_scene_rasters import _overlay_in_terrain_uv
    from forge3d._png import load_png_rgba
    # Native count cells are coarser than the target grid, so the drape reprojects.
    counts = Source("counts", raster(tmp_path / "counts.tif", np.array([[1, 7], [0, 9]], np.int32), grid(2, 2, spacing=200)),
                    "count", units="people/cell")
    classes = Source("classes", raster(tmp_path / "classes.tif", np.array([[1, 2, 1, 2]] * 4, np.uint8), grid()), "categorical")
    prepared = prepare_scene(Recipe(grid(), (elevation(tmp_path), counts, classes), "NAP", display_styles={
        "counts": {"range": [0, 10], "colors": [[0, 0, 0, 255], [255, 255, 255, 255]]},
        "classes": {"classes": {"1": [200, 30, 30, 255], "2": [30, 30, 200, 128]}}}))
    prepared.save(tmp_path / "snapshot")
    scene = prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(64, 64),
                              camera=f3d.OrbitCamera(elevation_deg=45))
    assets = prepared.viewer_assets(tmp_path / "viewer")
    rasters = [layer for layer in scene.recipe.layers if isinstance(layer, f3d.RasterOverlay)]
    assert [layer.layer_id for layer in rasters] == [o["source_id"] for o in assets["overlays"]] == ["counts", "classes"]
    for layer, overlay in zip(rasters, assets["overlays"]):
        draped = _overlay_in_terrain_uv(layer, scene.recipe, (4, 4))
        np.testing.assert_array_equal(draped, load_png_rgba(overlay["path"]))


def test_log1p_style_reuses_recipe_for_previously_unseen_counts(tmp_path):
    from forge3d.preparation import _display_rgba
    path = raster(tmp_path / "counts.tif", np.full((4, 4), 5, np.int32), grid())
    source = Source("population", path, "count", units="people/cell")
    style = {"range": [0, 2010], "scale": "log1p", "colors": [[246, 226, 178, 255], [146, 30, 56, 255]]}
    recipe = Recipe(grid(), (elevation(tmp_path), source), "NAP", display_styles={"population": style})
    recipe.save(tmp_path / "recipe.json")
    raster(path, np.full((4, 4), 17, np.int32), grid())  # a new published value; no lookup edit
    prepared = prepare_scene(Recipe.load(tmp_path / "recipe.json"))
    rgba = _display_rgba(prepared.layers["population"], style)
    expected = np.rint(np.array(style["colors"][0]) + np.log1p(17) / np.log1p(2010) *
                       (np.array(style["colors"][1]) - style["colors"][0])).astype(np.uint8)
    np.testing.assert_array_equal(rgba[0, 0], expected)
    assert np.all(prepared.layers["population"].data == 17)


def test_population_state_survives_snapshot_and_viewer_assets(tmp_path):
    from PIL import Image
    counts = np.array([[0, -99997, 15, 0]] * 4, np.int32)
    states = np.array([[0, 1, 2, 0]] * 4, np.uint8)
    count_path = raster(tmp_path / "counts.tif", counts, grid())
    with rasterio.open(count_path, "r+") as ds: ds.write_mask((states == 2).astype(np.uint8) * 255)
    state = Source("population_state", raster(tmp_path / "state.tif", states, grid()), "categorical")
    population = Source("population", count_path, "count", units="people/cell")
    styles = {"population_state": {"classes": {"0": [193, 192, 179, 255], "1": [125, 88, 148, 255], "2": [0, 0, 0, 0]}},
              "population": {"range": [0, 2010], "scale": "log1p", "colors": [[246, 226, 178, 255], [146, 30, 56, 255]]}}
    prepared = prepare_scene(Recipe(grid(), (elevation(tmp_path), state, population), "NAP", display_styles=styles))
    prepared.save(tmp_path / "snapshot")
    restored = PreparedScene.load(tmp_path / "snapshot")
    np.testing.assert_array_equal(restored.layers["population_state"].data, states)
    np.testing.assert_array_equal(restored.layers["population"].valid_mask, states == 2)
    assets = restored.viewer_assets(tmp_path / "viewer")
    rgba = np.array(Image.open(assets["overlays"][0]["path"]))
    np.testing.assert_array_equal(rgba[0, 0], styles["population_state"]["classes"]["0"])
    np.testing.assert_array_equal(rgba[0, 1], styles["population_state"]["classes"]["1"])


@pytest.mark.parametrize("kind", ["vector", "observation"])
def test_pixel_checkpoint_requires_raster_at_recipe_load(tmp_path, kind):
    source = Source("features", tmp_path / "features.geojson", kind, crs=grid().crs,
                    units="m" if kind == "observation" else None)
    with pytest.raises(PreparationError, match="invalid_recipe"):
        Recipe(grid(), (elevation(tmp_path), source), "NAP", checkpoints=[{
            "source_id": "features", "source_pixel": [0, 0], "expected_target_xy": [0, 0], "tolerance": 0}])


@pytest.mark.parametrize("style", [
    {"range": [0, 10], "colour": [[0, 0, 0, 255], [255, 255, 255, 255]]},
    {"range": [0, 10], "colors": [[0, 0, 0, 255], [255, 255, 255, 255]], "sacle": "log1p"},
    {"range": [0, 10], "colors": [[0, 0, 0, 255], [255, 255, 255, 255]], "scale": "log"},
    {"range": [10, 0], "colors": [[0, 0, 0, 255], [255, 255, 255, 255]]},
    {"classes": {"wrong": [0, 0, 0, 255]}},
    {"classes": {"1": [0, 0, 256, 255]}},
])
def test_display_style_errors_are_diagnosed_when_loading_recipe(tmp_path, style):
    recipe = Recipe(grid(), (elevation(tmp_path),), "NAP").to_dict()
    recipe["display_styles"] = {"dem": style}
    with pytest.raises(PreparationError, match="invalid_recipe"):
        Recipe.from_dict(recipe)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_recipe_metadata_is_rejected_before_io(tmp_path, constant):
    value = Recipe(grid(), (elevation(tmp_path),), "NAP").to_dict()
    value["sources"][0]["metadata"]["bad"] = "NONFINITE"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value).replace('"NONFINITE"', constant), encoding="utf-8")
    with pytest.raises(PreparationError, match="invalid_recipe"):
        Recipe.load(path)
    value["sources"][0]["metadata"]["bad"] = float(constant.replace("NaN", "nan"))
    with pytest.raises(PreparationError, match="invalid_recipe"):
        Recipe.from_dict(value)


def test_snapshot_publish_is_atomic_and_retryable(tmp_path, monkeypatch):
    prepared = prepare_scene(Recipe(grid(), (elevation(tmp_path),), "NAP"))
    directory = tmp_path / "saved"
    before = set(tmp_path.iterdir())
    with monkeypatch.context() as patch:
        patch.setattr(np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("simulated disk write failure")))
        with pytest.raises(PreparationError, match="snapshot_write_failed"):
            prepared.save(directory)
    assert set(tmp_path.iterdir()) == before
    assert prepared.root is None
    prepared.save(directory)
    assert PreparedScene.load(directory).report == prepared.report
    recorded = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(PreparationError, match="output_exists"):
        prepared.save(directory)
    assert recorded == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_invalid_snapshot_metadata_does_not_publish_partial_folder(tmp_path):
    prepared = prepare_scene(Recipe(grid(), (elevation(tmp_path),), "NAP"))
    prepared.report["bad"] = float("nan")
    with pytest.raises(PreparationError, match="invalid_recipe"):
        prepared.save(tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad.*"))


@pytest.mark.parametrize("explicit", [False, True])
def test_scene_display_assets_do_not_mutate_snapshot(tmp_path, explicit):
    source = Source("classes", raster(tmp_path / "classes.tif", np.ones((4, 4), np.uint8), grid()), "categorical")
    prepared = prepare_scene(Recipe(grid(), (elevation(tmp_path), source), "NAP",
                                    display_styles={"classes": {"classes": {"1": [255, 0, 0, 255]}}}))
    root = prepared.save(tmp_path / "snapshot")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assets = tmp_path / "custom-assets" if explicit else root.with_name("snapshot.scene-assets")
    scene = prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(64, 64),
                              **({"assets_dir": assets} if explicit else {}))
    assert Path(scene.recipe.layers[0].path).parent == assets
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert PreparedScene.load(root).report == prepared.report
    with pytest.raises(PreparationError, match="invalid_scene_assets_directory"):
        prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(64, 64), assets_dir=root / "display")
