"""D02 hand-computed units, masks, colors, bins, legends and recipe proof."""
import json

import numpy as np
import pytest

import forge3d as f3d
from forge3d.raster_style import raster_style_from_dict

PALETTE = (
    ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)),
    ((255, 255, 0, 255), (0, 255, 255, 255), (255, 0, 255, 255)),
    ((128, 0, 0, 255), (0, 128, 0, 255), (0, 0, 128, 255)),
)


def bivariate(**kwargs):
    args = dict(x_bins=[1, 2], y_bins=[10, 20], palette=PALETTE,
                x_label="Population", y_label="Temperature", x_units="people/km²", y_units="°C")
    return f3d.BivariateRasterStyle(**(args | kwargs))


def scene(terrain, layers=()):
    return f3d.MapScene(terrain=terrain, layers=layers,
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64))


def test_population_heights_mask_shade_units():
    style = f3d.RasterHeightSurfaceStyle(2, "people/km²", nodata=-1,
                                       low_color=(0, 0, 0, 255), high_color=(200, 100, 50, 255))
    data = np.array([[0, 10, 20], [-1, np.nan, 15]], np.float32)
    result = style.apply(data, valid_mask=np.array([[1, 1, 1], [1, 1, 0]], bool))
    np.testing.assert_allclose(result.heights, [[0, 20, 40], [np.nan, np.nan, np.nan]], equal_nan=True)
    np.testing.assert_array_equal(result.valid_mask, [[1, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(result.rgba[0], [[0, 0, 0, 255], [100, 50, 25, 255], [200, 100, 50, 255]])
    assert not result.rgba[1].any()
    assert result.legend["source_units"] == "people/km²"
    assert result.legend["height_units"] == "m"
    assert result.legend["height_range"] == [0, 40]
    np.testing.assert_array_equal(style.apply(np.ones((2, 2), np.float32)).rgba[0, 0], style.low_color)


def test_categorical_exact_ids_colors_and_legend():
    style = f3d.CategoricalRasterStyle({7: PALETTE[0][2], 0: PALETTE[0][0], 2: PALETTE[0][1]},
                                     {0: "bare", 2: "forest", 7: "water"}, nodata=-1, units="class")
    result = style.apply(np.array([[0, 2, 7], [-1, np.nan, 2]], np.float32))
    np.testing.assert_array_equal(result.rgba[0], PALETTE[0])
    np.testing.assert_array_equal(result.classes, [[1, 2, 3], [0, 0, 2]])
    assert result.legend["entries"] == [
        {"class_id": 0, "label": "bare", "rgba": list(PALETTE[0][0])},
        {"class_id": 2, "label": "forest", "rgba": list(PALETTE[0][1])},
        {"class_id": 7, "label": "water", "rgba": list(PALETTE[0][2])}]
    assert result.legend["nodata_rgba"] == [0, 0, 0, 0]
    assert result.legend["units"] == "class"
    np.testing.assert_array_equal(style.apply(np.array([[0, 2, 7]], np.int16)).rgba, [PALETTE[0]])
    for data in [np.array([[3]], np.float32), np.array([[2.5]], np.float32)]:
        with pytest.raises(ValueError, match="invalid_argument"):
            style.apply(data)


@pytest.mark.parametrize("right,xc,yc", [
    (False, [1, 2, 3, 3], [1, 2, 3, 3]),
    (True, [1, 1, 2, 3], [1, 1, 2, 3]),
])
def test_bivariate_boundary_bins_and_axes(right, xc, yc):
    style = bivariate(right=right)
    result = style.apply(np.array([[0, 1, 2, 3, np.nan]], np.float32),
                         np.array([[0, 10, 20, 30, 0]], np.float32))
    np.testing.assert_array_equal(result.x_classes, [xc + [0]])
    np.testing.assert_array_equal(result.y_classes, [yc + [0]])
    np.testing.assert_array_equal(result.rgba[0, :4], [PALETTE[y - 1][x - 1] for x, y in zip(xc, yc)])
    assert not result.rgba[0, 4].any()
    assert result.legend["x_axis"] == {"label": "Population", "units": "people/km²", "bins": [1., 2.], "labels": ["low", "middle", "high"]}
    assert result.legend["y_axis"]["label"] == "Temperature"
    assert result.legend["right_inclusive"] is right


def test_bivariate_all_nine_cells_and_joint_nodata():
    x = np.array([[0, 1, 2]] * 3, np.float32)
    y = np.array([[0] * 3, [10] * 3, [20] * 3], np.float32)
    result = bivariate().apply(x, y)
    np.testing.assert_array_equal(result.rgba, PALETTE)
    np.testing.assert_array_equal(result.classes, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.ones((3, 3), bool)
    mask[0, 1] = False
    y[2, 0] = -99
    result = bivariate(y_nodata=-99).apply(x, y, valid_mask=mask)
    assert not result.valid_mask[0, 1] and not result.valid_mask[2, 0]
    assert not result.rgba[0, 1].any() and not result.rgba[2, 0].any()


@pytest.mark.parametrize("factory", [
    lambda: f3d.RasterHeightSurfaceStyle(0, "people"),
    lambda: f3d.RasterHeightSurfaceStyle(float("nan"), "people"),
    lambda: f3d.RasterHeightSurfaceStyle(1, ""),
    lambda: f3d.CategoricalRasterStyle({1: (256, 0, 0, 255)}),
    lambda: f3d.CategoricalRasterStyle({1: PALETTE[0][0]}, {2: "bad"}),
    lambda: bivariate(x_bins=[1, 1]),
    lambda: bivariate(palette=[[(0, 0, 0, 255)]]),
    lambda: bivariate(x_labels=["one"]),
    lambda: bivariate(palette=None),
    lambda: bivariate(palette=[None, None, None]),
    lambda: bivariate(x_bins=None),
    lambda: bivariate(y_labels=None),
])
def test_invalid_style_diagnoses(factory):
    with pytest.raises(ValueError, match="invalid_argument|shape_mismatch"):
        factory()


@pytest.mark.parametrize("data", [np.array([[-2]], np.float32), np.array([[1e38]], np.float32)])
def test_invalid_population_domain_or_overflow(data):
    with pytest.raises(ValueError, match="invalid_argument"):
        f3d.RasterHeightSurfaceStyle(100, "people").apply(data)


def test_invalid_bivariate_shape_empty_and_bins():
    with pytest.raises(ValueError, match="shape_mismatch"):
        bivariate().apply(np.zeros((2, 2), np.float32), np.zeros((3, 2), np.float32))
    with pytest.raises(ValueError, match="empty_raster"):
        bivariate().apply(np.array([[np.nan, 0]], np.float32), np.array([[0, np.nan]], np.float32))
    for bins in ([1, 1], [2, 1], [1, float("inf")], [1]):
        args = bivariate().to_dict()
        args.pop("kind")
        args["x_bins"] = bins
        with pytest.raises(ValueError, match="invalid_argument"):
            f3d.BivariateRasterStyle(**args)


@pytest.mark.parametrize("kind", ["population", "categorical", "bivariate"])
def test_style_bundle_canonical_parameters_results_legends(tmp_path, kind):
    data = np.array([[0., 1., 2.], [2., np.nan, 0.]], np.float32)
    pop = f3d.RasterHeightSurfaceStyle(2, "people", nodata=-99)
    terrain = f3d.TerrainSource(data=data, style=pop, crs="EPSG:3857", metadata={"source_id": "fixture"})
    layers = []
    if kind != "population":
        style = bivariate() if kind == "bivariate" else f3d.CategoricalRasterStyle({i: PALETTE[0][i] for i in range(3)})
        layers = [f3d.RasterOverlay("thematic", data=data, style=style, secondary_data=data * 10 if kind == "bivariate" else None, crs="EPSG:3857")]
    original = scene(terrain, layers)
    bundle1 = tmp_path / "one.forge3d"
    original.save_bundle(bundle1)
    loaded = f3d.MapScene.load_bundle(bundle1)
    assert loaded.recipe.to_dict() == original.recipe.to_dict()
    bundle2 = tmp_path / "two.forge3d"
    loaded.save_bundle(bundle2)
    for path in bundle1.rglob("*.json"):
        assert path.read_bytes() == (bundle2 / path.relative_to(bundle1)).read_bytes(), path
    before = pop.apply(data)
    after = loaded.recipe.terrain.style.apply(loaded.recipe.terrain.data)
    np.testing.assert_allclose(before.heights, after.heights, equal_nan=True)
    assert before.legend == after.legend
    if layers:
        from forge3d.map_scene import _styled_raster_result
        a, b = _styled_raster_result(layers[0]), _styled_raster_result(loaded.recipe.layers[0])
        np.testing.assert_array_equal(a.classes, b.classes)
        np.testing.assert_array_equal(a.rgba, b.rgba)
        assert a.legend == b.legend
    json.dumps(original.recipe.to_dict(), allow_nan=False)


def test_style_serialization_and_public_exports():
    from forge3d import style
    for obj in (f3d.RasterHeightSurfaceStyle(2, "people"), bivariate(), f3d.CategoricalRasterStyle({1: PALETTE[0][0]})):
        assert getattr(style, type(obj).__name__) is type(obj)
        assert type(obj).__name__ in f3d.__all__
        assert raster_style_from_dict(json.loads(json.dumps(obj.to_dict()))).to_dict() == obj.to_dict()
    assert callable(f3d.OverlayLayer.from_raster_rgba)


def test_single_band_population_bundle_is_byte_identical(tmp_path):
    source = np.array([[[0, 10, 20], [20, np.nan, 0]]], np.float32)
    style = f3d.RasterHeightSurfaceStyle(2, "people/km²")
    terrain = f3d.TerrainSource(data=source, style=style, crs="EPSG:3857", metadata={"source_id": "single-band"})
    assert terrain.data.shape == (2, 3)
    assert terrain.to_dict()["data"]["shape"] == [2, 3]
    original = scene(terrain)
    first = tmp_path / "one.forge3d"
    second = tmp_path / "two.forge3d"
    original.save_bundle(first)
    restored = f3d.MapScene.load_bundle(first)
    restored.save_bundle(second)
    files = {path.relative_to(first) for path in first.rglob("*") if path.is_file()}
    assert files == {path.relative_to(second) for path in second.rglob("*") if path.is_file()}
    for path in files:
        assert (first / path).read_bytes() == (second / path).read_bytes(), path
    before = style.apply(source)
    after = restored.recipe.terrain.style.apply(restored.recipe.terrain.data)
    np.testing.assert_allclose(before.heights, after.heights, equal_nan=True)
    np.testing.assert_array_equal(before.valid_mask, after.valid_mask)
    np.testing.assert_array_equal(before.rgba, after.rgba)
    assert before.legend == after.legend


def test_native_raster_constructor_rejects_invalid_inputs_before_gpu():
    with pytest.raises(ValueError, match="shape"):
        f3d.OverlayLayer.from_raster_rgba(np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(ValueError, match="strength"):
        f3d.OverlayLayer.from_raster_rgba(np.zeros((2, 2, 4), np.uint8), strength=float("nan"))


def test_local_geotiff_nodata_preserved(tmp_path):
    source = np.array([[0, 10], [-99, 20]], np.float32)
    path = tmp_path / "population.tif"
    f3d.gis.write_raster(path, source, crs="EPSG:3857", transform=(0, 1, 0, 2, 0, -1), nodata=-99)
    style = f3d.RasterHeightSurfaceStyle(2, "people/km²")
    result = style.apply(path)
    np.testing.assert_allclose(result.heights, [[0, 20], [np.nan, 40]], equal_nan=True)
    terrain = f3d.TerrainSource(path=path, style=style)
    from forge3d.map_scene import _load_native_heightmap
    np.testing.assert_allclose(_load_native_heightmap(terrain), result.heights, equal_nan=True)


def test_style_decode_rejects_nonfinite_parameters_and_canonicalizes_zero():
    from forge3d.raster_style import raster_values_from_dict
    with pytest.raises(ValueError, match="invalid_argument"):
        raster_values_from_dict({"dtype": "float32", "values": [[float("nan")]]})
    encoded = f3d.TerrainSource(data=np.array([[-0.0, np.nan]], np.float32),
                              style=f3d.RasterHeightSurfaceStyle(1, "people")).to_dict()
    assert encoded["values"]["values"] == [[0, None]]
    assert "NaN" not in json.dumps(encoded, allow_nan=False)


def test_mapscene_invalid_style_blocks_explicitly(tmp_path):
    terrain = f3d.TerrainSource(data=np.ones((2, 2), np.float32), style=f3d.RasterHeightSurfaceStyle(1, "people"), metadata={"source_id": "fixture"})
    bad = f3d.RasterOverlay("bad", data=np.full((2, 2), 5, np.float32), style=f3d.CategoricalRasterStyle({1: PALETTE[0][0]}))
    s = scene(terrain, [bad])
    report = s.validate()
    assert any(d.code == "invalid_raster_style" and d.layer_id == "bad" for d in report.diagnostics)
    assert report.render_blocked(s.render_policy)
    with pytest.raises(Exception, match="invalid_raster_style|diagnostic"):
        s.render(str(tmp_path / "blocked.png"))
    assert not (tmp_path / "blocked.png").exists()
