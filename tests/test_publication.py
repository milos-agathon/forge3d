"""C06 dependency/geometry contracts. Fake frames below are unit-test fixtures.

Actual native execution, cache receipts and CBS revisions are exercised by
examples/c06_population_publication.py, never inferred from these fixtures.
"""
from dataclasses import replace
import hashlib
import os

import numpy as np
import pytest

from forge3d import gis
from forge3d._png import save_png
from forge3d.map_plate import BBox
from forge3d.map_scene import MapScene, MapSceneNativeUnavailable
from forge3d.publication import PublicationSource, PublicationExtent, PublicationRecipe

pytestmark = pytest.mark.usefixtures("pro_license")


@pytest.fixture
def publication(tmp_path):
    path = tmp_path / "2024.tif"
    values = np.arange(600, dtype=np.float32).reshape(20, 30)
    values[0, 0] = -9999
    gis.write_raster(path, values, crs="EPSG:32631", nodata=-9999,
                     transform=(100, 0, 600000, 0, -100, 5820000))
    source = PublicationSource(path, "Fixture", "2024", "v2", "https://example.test/2024",
                               "people/source cell", "No published count")
    bounds = BBox(600000, 5818000, 603000, 5820000, "EPSG:32631")
    recipe = PublicationRecipe(source, [100, 300], [(240, 230, 200, 255),
                                (220, 130, 70, 255), (110, 30, 50, 255)], bounds)
    page = PublicationExtent("city", "Fixture city", BBox(600000, 5819000, 601000, 5820000, bounds.crs))
    return recipe, page, values


@pytest.fixture
def fake_render(monkeypatch):
    observed = type("Observed", (list,), {"gain": 1.0})()  # gain < 1 mimics a lit frame.
    def render(scene, path, *, cache=None):
        assert scene.recipe.lighting.settings["unlit_albedo"] is True and not scene.recipe.output.aovs
        rgba = scene.recipe.layers[0].style.apply(scene.recipe.layers[0].data).rgba
        yy = np.arange(scene.recipe.output.height) * rgba.shape[0] // scene.recipe.output.height
        xx = np.arange(scene.recipe.output.width) * rgba.shape[1] // scene.recipe.output.width
        frame = rgba[yy[:, None], xx].copy()
        frame[..., :3] = np.rint(frame[..., :3] * observed.gain)
        save_png(path, frame)
        scene.last_render_backend = "unit-test-fixture"
        scene.last_render_metadata = {"cache_argument": str(cache)}
        observed.append(scene)
    monkeypatch.setattr(MapScene, "render", render)
    return observed


def ids(receipt):
    return {key: value["id"] for key, value in receipt["dependencies"].items()}


def test_revisions_use_current_inputs_and_preserve_unaffected_dependencies(publication, fake_render, tmp_path):
    recipe, page, values = publication
    def run(name):
        return recipe.render_series([page], tmp_path / name, cache=tmp_path / "cache")[0]
    first = run("first")
    same = run("same")
    assert ids(first) == ids(same)
    assert first["output"]["pixel_sha256"] == same["output"]["pixel_sha256"]
    assert len(fake_render) == 4  # Every call still enters MapScene's cache authority.
    recipe.breaks = [50, 250]
    changed = run("breaks")
    for key in ("classification", "legend", "map", "locator"):
        assert ids(first)[key] != ids(changed)[key]
    for key in ("source", "caption", "scale"):
        assert ids(first)[key] == ids(changed)[key]
    assert first["output"]["pixel_sha256"] != changed["output"]["pixel_sha256"]
    recipe.breaks = [100, 300]
    page = replace(page, bounds=BBox(600100, 5819000, 601100, 5820000, "EPSG:32631"))
    moved = run("extent")
    assert ids(first)["map"] != ids(moved)["map"]
    assert ids(first)["locator"] != ids(moved)["locator"]
    assert ids(first)["legend"] == ids(moved)["legend"]
    assert first["footprint"]["coordinates"] != moved["footprint"]["coordinates"]
    newpath = tmp_path / "2023.tif"
    gis.write_raster(newpath, values + 20, crs="EPSG:32631", nodata=-9979,
                     transform=(100, 0, 600000, 0, -100, 5820000))
    recipe.source = replace(recipe.source, path=newpath, year="2023", edition="v1",
                            uri="https://example.test/2023")
    previous = run("year")
    assert "2023" in previous["caption"] and "v1" in previous["caption"]
    for key in ("source", "classification", "caption", "map", "locator"):
        assert ids(moved)[key] != ids(previous)[key]
    # Same path/size/mtime edits must also invalidate the actual map.
    data = values + 50
    previous_stat = newpath.stat()
    gis.write_raster(newpath, data, crs="EPSG:32631", nodata=-9949,
                     transform=(100, 0, 600000, 0, -100, 5820000), overwrite=True)
    os.utime(newpath, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    assert newpath.stat().st_size == previous_stat.st_size
    assert newpath.stat().st_mtime_ns == previous_stat.st_mtime_ns
    edited = run("edited")
    assert ids(previous)["map"] != ids(edited)["map"]
    assert previous["source"]["sha256"] != edited["source"]["sha256"]


def test_native_classifier_closure_palette_units_missing(publication):
    recipe, _, _ = publication
    values = np.array([[0, 99, 100, 300, -9999]], np.float32)
    valid = values != -9999
    for right, expected in ((False, [[1, 1, 2, 3, 0]]), (True, [[1, 1, 1, 2, 0]])):
        recipe.right = right
        classes, style, result, table = recipe._classify(values, valid)
        np.testing.assert_array_equal(classes, expected)
        assert result.legend == style.legend()
        assert result.legend["units"] == recipe.source.units
        assert result.legend["entries"][0]["label"] == recipe.source.missing_label
        assert len(result.legend["entries"]) == 4
        for entry in result.legend["entries"]:
            np.testing.assert_array_equal(result.rgba[classes == entry["class_id"]],
                np.tile(entry["rgba"], (np.count_nonzero(classes == entry["class_id"]), 1)))
        assert sum(row["count"] for row in table) == values.size


def test_registration_and_scale_independent_of_recipe(publication, fake_render, tmp_path):
    from pyproj import Transformer, Geod
    recipe, page, _ = publication
    result = recipe.render_series([page], tmp_path)[0]
    # Independent source affine -> locator pixels, without recipe projection helpers.
    xy = np.array(result["footprint"]["coordinates"])
    expected = (xy - [600000, 5820000]) / [3000, -2000] * result["locator_viewport"]
    np.testing.assert_array_equal(expected, result["footprint"]["locator_pixels"])
    transform = Transformer.from_crs(32631, 4326, always_xy=True)
    lon, lat = transform.transform([600000, 601000], [5819500, 5819500])
    distance = Geod(ellps="WGS84").inv(*[lon[0], lat[0], lon[1], lat[1]])[2]
    # Existing MENSURA/PROJ agreement is measured; round to the publication
    # pixel's ground footprint, rather than inventing a geographic tolerance.
    actual = result["scale"]["meters_per_pixel"] * result["main_viewport"][0]
    assert round(actual / result["scale"]["meters_per_pixel"]) == round(distance / result["scale"]["meters_per_pixel"])
    label, unit = result["scale"]["bar_label"].split()
    labelled_metres = float(label) * {"m": 1, "km": 1000}[unit]
    independent_mpp = distance / result["main_viewport"][0]
    assert abs(labelled_metres - result["scale"]["bar_pixel_span"] * independent_mpp) < independent_mpp


@pytest.mark.parametrize("case", ["camera", "unaligned", "outside", "crs", "locator", "palette", "bins", "empty", "duplicate"])
def test_unsupported_cases_are_explicit(publication, fake_render, tmp_path, case):
    recipe, page, _ = publication
    pages = [page]
    if case == "camera": recipe.camera_mode = "mesh:zup"
    if case == "unaligned": pages = [replace(page, bounds=replace(page.bounds, west=600001))]
    if case == "outside": pages = [replace(page, bounds=replace(page.bounds, west=599900))]
    if case == "crs": pages = [replace(page, bounds=replace(page.bounds, crs="EPSG:4326"))]
    if case == "locator": recipe.locator_bounds = page.bounds; pages = [replace(page, bounds=replace(page.bounds, east=601100))]
    if case == "palette": recipe.palette = [(0, 0, 0, 255)]
    if case == "bins": recipe.breaks = [300, 100]
    if case == "empty": pages = []
    if case == "duplicate": pages = [page, page]
    with pytest.raises((ValueError, TypeError)):
        recipe.render_series(pages, tmp_path / case)
    assert not fake_render


def test_unavailable_native_does_not_return_old_output(publication, monkeypatch, tmp_path):
    recipe, page, _ = publication
    old = tmp_path / "city.png"
    old.write_bytes(b"old output is not a valid result")
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    def unavailable(*args, **kwargs):
        raise MapSceneNativeUnavailable({"layer": "publication", "reason": "test native absence",
                                         "required_native": "physical renderer"})
    monkeypatch.setattr(MapScene, "render", unavailable)
    with pytest.raises(MapSceneNativeUnavailable, match="test native absence"):
        recipe.render_series([page], tmp_path)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == digest


def test_palette_units_and_missing_key_are_linked(publication, fake_render, tmp_path):
    recipe, page, _ = publication
    base = recipe.render_series([page], tmp_path / "base")[0]
    recipe.palette = [(10, 20, 30, 255), *recipe.palette[1:]]
    palette = recipe.render_series([page], tmp_path / "palette")[0]
    assert ids(base)["map"] != ids(palette)["map"]
    assert palette["legend"]["entries"][1]["rgba"] == [10, 20, 30, 255]
    recipe.source = replace(recipe.source, units="revised units", missing_label="Absent or suppressed")
    semantic = recipe.render_series([page], tmp_path / "semantic")[0]
    assert semantic["legend"]["units"] == "revised units"
    assert semantic["legend"]["entries"][0]["label"] == "Absent or suppressed"
    assert ids(palette)["legend"] != ids(semantic)["legend"]
    assert ids(palette)["map"] != ids(semantic)["map"]


@pytest.mark.parametrize("case", ["geographic", "feet", "unsupported_crs", "rotated", "multiband", "empty_data"])
def test_unsupported_source_registration(publication, fake_render, tmp_path, monkeypatch, case):
    recipe, page, values = publication
    crs, transform = "EPSG:32631", (100, 0, 600000, 0, -100, 5820000)
    if case == "geographic": crs = "EPSG:4326"
    if case == "feet": crs = "EPSG:2263"
    if case == "unsupported_crs": crs = "EPSG:28992"
    if case == "rotated": transform = (100, 1, 600000, 0, -100, 5820000)
    if case == "multiband": values = np.stack([values, values])
    if case == "empty_data": values = np.full_like(values, -9999)
    # The native writer itself rejects these CRS codes. Exercise the reader
    # boundary with equivalent metadata from an externally authored GeoTIFF.
    written_crs = "EPSG:32631" if case in ("feet", "unsupported_crs") else crs
    gis.write_raster(recipe.source.path, values, crs=written_crs, transform=transform,
                     nodata=-9999, overwrite=True)
    if case in ("feet", "unsupported_crs"):
        original = gis.read_raster
        def read(*args, **kwargs):
            result = original(*args, **kwargs)
            result["info"]["crs_authority"] = {"name": "EPSG", "code": crs.split(":")[1]}
            return result
        monkeypatch.setattr(gis, "read_raster", read)
    with pytest.raises(ValueError):
        recipe.render_series([page], tmp_path / case)
    assert not fake_render


def test_immutable_categorical_style_can_be_snapshotted(publication):
    import copy
    recipe, _, values = publication
    _, style, _, _ = recipe._classify(values, values >= 0)
    assert copy.deepcopy(style) is style
    with pytest.raises(TypeError):
        style.palette[0] = (0, 0, 0, 255)
    with pytest.raises(TypeError):
        style.labels[0] = "changed"


def test_legend_retains_precise_breaks_and_text_never_silently_clips(publication, fake_render, tmp_path):
    recipe, page, _ = publication
    recipe.breaks = [1.0000001, 1.0000002]
    _, _, result, _ = recipe._classify(np.array([[0, 2]], np.float64), np.ones((1, 2), bool))
    assert "1.0000001" in result.legend["entries"][2]["label"]
    assert "1.0000002" in result.legend["entries"][2]["label"]
    with pytest.raises(ValueError, match="title cannot fit"):
        recipe.render_series([replace(page, title="Long title " * 200)], tmp_path / "long")
    assert not fake_render


def _shown_and_expected(result, values, breaks):
    """Saved page pixels at cell centres vs an independent NumPy classification."""
    from PIL import Image
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA")).astype(int)
    x0, y0, x1, y1 = result["map_rect"]
    rows, cols = values.shape
    ys = y0 + ((np.arange(rows) + .5) * (y1 - y0) / rows).astype(int)
    xs = x0 + ((np.arange(cols) + .5) * (x1 - x0) / cols).astype(int)
    classes = np.where(values != -9999, np.searchsorted(breaks, values, side="right") + 1, 0)
    palette = {e["class_id"]: e["rgba"][:3] for e in result["legend"]["entries"]}
    expected = np.array([palette[k] for k in range(len(palette))])[classes]
    bx0, by0, bx1, by1 = result["scale"]["bar_rect"]
    keep = ~((ys[:, None] >= by0) & (ys[:, None] < by1) & (xs[None] >= bx0) & (xs[None] < bx1))
    return image[np.ix_(ys, xs)][..., :3][keep], expected[keep], set(np.unique(classes).tolist())


def test_displayed_map_colours_are_the_legend_colours(publication, fake_render, tmp_path):
    recipe, page, values = publication
    whole = replace(page, bounds=recipe.locator_bounds)  # Values 0-599 plus the missing cell.
    result = recipe.render_series([whole], tmp_path / "ok")[0]
    assert result["main_render"]["max_legend_deviation_levels"] <= 1
    shown, expected, classes = _shown_and_expected(result, values, recipe.breaks)
    assert classes == {0, 1, 2, 3}  # Missing and every interval are exercised.
    assert np.abs(shown - expected).max() <= 1


def test_shaded_frame_or_absent_albedo_is_rejected(publication, fake_render, tmp_path):
    from forge3d.map_scene import OutputSpec, _unlit_albedo_rgba
    recipe, page, _ = publication
    fake_render.gain = 0.46  # The measured lit/tone-mapped darkening of the beauty frame.
    with pytest.raises(ValueError, match="differ from the legend"):
        recipe.render_series([page], tmp_path / "shaded")
    assert not (tmp_path / "shaded" / "city.png").exists()
    with pytest.raises(MapSceneNativeUnavailable, match="albedo"):
        _unlit_albedo_rgba(None, OutputSpec(width=4, height=4))


@pytest.mark.parametrize("field", ["missing_color", "palette"])
def test_non_opaque_keys_are_rejected_not_flattened(publication, fake_render, tmp_path, field):
    recipe, page, _ = publication
    if field == "missing_color":
        recipe.missing_color = (0, 0, 0, 0)
    else:
        recipe.palette = [(240, 230, 200, 128), *recipe.palette[1:]]
    with pytest.raises(ValueError, match="fully opaque"):
        recipe.render_series([page], tmp_path / field)
    assert not fake_render


def test_cell_edges_tolerate_only_representational_rounding(tmp_path):
    path = tmp_path / "fine.tif"
    gis.write_raster(path, np.arange(100, dtype=np.float32).reshape(10, 10), crs="EPSG:32631", nodata=-9999,
                     transform=(0.1, 0, 600000.1, 0, -0.1, 5820000.1))
    source = PublicationSource(path, "Fine", "2024", "v", "https://example.test", "people", "missing")
    bounds = BBox(600000.1, 5819999.1, 600001.1, 5820000.1, "EPSG:32631")
    recipe = PublicationRecipe(source, [10], [(1, 2, 3, 255), (4, 5, 6, 255)], bounds)
    values, _, grid, resolution, _ = recipe._read()
    for k in range(1, 10):  # Several of these edges divide to 0.9999999997..., 3.0000000004...
        west = grid.west + k * resolution[0]
        rows, cols = recipe._window(replace(bounds, west=west), grid, resolution, values.shape)
        assert (rows.start, rows.stop, cols.start, cols.stop) == (0, 10, k, 10)
    for offset in (0.5, 0.01, 1e-6):  # In cells: small, but far beyond rounding.
        with pytest.raises(ValueError, match="cell edges"):
            recipe._window(replace(bounds, west=grid.west + offset * resolution[0]), grid, resolution, values.shape)
    with pytest.raises(ValueError, match="too fine"):
        recipe._window(bounds, grid, (1e-10, 1e-10), values.shape)


def test_scale_bar_is_drawn_on_the_map_not_the_margin(publication, fake_render, tmp_path):
    from PIL import Image
    recipe, page, _ = publication
    narrow = replace(page, bounds=replace(page.bounds, east=600600))  # 6 x 10 cells, centred narrow map.
    result = recipe.render_series([narrow], tmp_path / "narrow")[0]
    x0, y0, x1, y1 = result["map_rect"]
    assert result["main_viewport"] == [x1 - x0, y1 - y0] == [558, 930]
    bx0, by0, bx1, by1 = result["scale"]["bar_rect"]
    assert x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA"))
    # The bar's black frame is inside the map; the margin left of it stays background.
    assert np.any(np.all(image[by0:by1, bx0:bx1, :3] == 0, axis=2))
    assert np.all(image[y0:y1, x0 - 30:x0, :3] == 255)
    with pytest.raises(ValueError, match="scale bar cannot fit"):
        recipe.render_series([replace(page, bounds=replace(page.bounds, east=600100))], tmp_path / "sliver")


@pytest.mark.parametrize("width", [60, 300, 479, 480, 1000])
def test_short_scale_label_matches_independently_measured_bar(publication, fake_render, tmp_path, width):
    from PIL import Image
    from pyproj import Geod, Transformer
    recipe, page, values = publication
    cell = width / 10
    bounds = BBox(600000, 5820000 - width, 600000 + width, 5820000, "EPSG:32631")
    gis.write_raster(recipe.source.path, values[:10, :10], crs=bounds.crs, nodata=-9999,
                     transform=(cell, 0, bounds.west, 0, -cell, bounds.north), overwrite=True)
    recipe.locator_bounds = bounds
    page = replace(page, bounds=bounds)
    receipt = recipe.render_series([page], tmp_path / "short")[0]
    transform = Transformer.from_crs(bounds.crs, 4326, always_xy=True)
    lon, lat = transform.transform([bounds.west, bounds.east], [bounds.center_lat] * 2)
    ground_width = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])[2]
    mpp = ground_width / receipt["main_viewport"][0]
    image = np.asarray(Image.open(receipt["output"]["path"]).convert("RGBA"))
    x0, y0, x1, y1 = receipt["scale"]["bar_rect"]
    # Detect the actual border in the exported image, not a second recipe span.
    row = np.flatnonzero(np.all(image[y0 + 12, x0:x1, :3] == 0, axis=1))
    drawn_metres = (row[-1] - row[0] + 1) * mpp
    number, units = receipt["scale"]["bar_label"].split()
    assert units == "m"
    assert abs(float(number) - drawn_metres) <= mpp


def test_wrong_scale_label_cannot_replace_a_page(publication, fake_render, tmp_path, monkeypatch):
    from forge3d.scale_bar import ScaleBar
    recipe, page, _ = publication
    old = tmp_path / "city.png"
    old.write_bytes(b"prior page")
    original = ScaleBar.render_geometry
    def wrong_label(self, **kwargs):
        image, _, anchor = original(self, **kwargs)
        return image, "0.0 km", anchor
    monkeypatch.setattr(ScaleBar, "render_geometry", wrong_label)
    with pytest.raises(ValueError, match="scale bar label"):
        recipe.render_series([page], tmp_path)
    assert old.read_bytes() == b"prior page"


@pytest.mark.parametrize("which", ["missing", "classes"])
@pytest.mark.parametrize("difference", [0, 1, 2])
def test_legend_keys_must_exceed_frame_colour_tolerance(publication, which, difference):
    recipe, _, values = publication
    first = recipe.palette[0]
    close = (first[0] + difference, *first[1:])
    if which == "missing":
        recipe.missing_color = close
    else:
        recipe.palette = [first, close, recipe.palette[2]]
    if difference <= 1:
        with pytest.raises(ValueError, match="distinguishable"):
            recipe._classify(values, values >= 0)
    else:
        recipe._classify(values, values >= 0)


def test_casefolded_page_ids_fail_before_render_or_replacement(publication, fake_render, tmp_path):
    recipe, page, _ = publication
    old = tmp_path / "City.png"
    old.write_bytes(b"prior page")
    with pytest.raises(ValueError, match="distinct page IDs"):
        recipe.render_series([replace(page, page_id="City"), replace(page, page_id="city")], tmp_path)
    assert not fake_render
    assert old.read_bytes() == b"prior page"


def _set_gdal_metadata(path, items):
    """Write actual embedded GeoTIFF metadata, preserving geographic TIFF tags."""
    from PIL import Image
    with Image.open(path) as source:
        pixels = source.copy()
        tags = dict(source.tag_v2)
    tags[42112] = "<GDALMetadata>" + items + "</GDALMetadata>"
    pixels.save(path, tiffinfo=tags)


@pytest.mark.parametrize(("scale", "offset"), [("0.1", "0"), ("1", "5"), ("nan", "0")])
def test_encoded_values_rejected_before_classification(publication, fake_render, tmp_path, scale, offset):
    recipe, page, _ = publication
    _set_gdal_metadata(recipe.source.path,
        f'<Item name="SCALE" sample="0" role="scale">{scale}</Item>'
        f'<Item name="OFFSET" sample="0" role="offset">{offset}</Item>')
    with pytest.raises(ValueError, match="scale/offset"):
        recipe.render_series([page], tmp_path / "encoded")
    assert not fake_render


@pytest.mark.parametrize("year", [None, "2024", "1999", ""])
def test_observation_year_tag_must_match_declaration(publication, fake_render, tmp_path, year):
    recipe, page, _ = publication
    items = '<Item name="SCALE" sample="0" role="scale">1</Item><Item name="OFFSET" sample="0" role="offset">0</Item>'
    if year is not None:
        items += f'<Item name="observation_year">{year}</Item>'
    _set_gdal_metadata(recipe.source.path, items)
    if year in (None, "2024"):
        values, _, _, _, description = recipe._read()
        assert values.shape == (20, 30)
    else:
        with pytest.raises(ValueError, match="observation_year"):
            recipe.render_series([page], tmp_path / "year")
    assert not fake_render


def test_external_metadata_sidecar_is_explicitly_unsupported(publication):
    recipe, _, _ = publication
    from pathlib import Path
    Path(str(recipe.source.path) + ".aux.xml").write_text('<PAMDataset/>', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        recipe._read()
