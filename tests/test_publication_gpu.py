"""C06 reuse and colour proof on real native frames. Opt-in physical GPU lane."""
import os

import numpy as np
from PIL import Image
import pytest

import forge3d as f3d
from forge3d import gis
from forge3d.map_plate import BBox
from forge3d.publication import PublicationExtent, PublicationRecipe, PublicationSource

pytestmark = [pytest.mark.usefixtures("pro_license"),
              pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
                                 reason="native publication reuse requires a physical GPU")]
TRANSFORM = (100, 0, 600000, 0, -100, 5820000)
PALETTE = [(240, 230, 200, 255), (220, 130, 70, 255), (110, 30, 50, 255)]
BOUNDS = BBox(600000, 5818000, 603000, 5820000, "EPSG:32631")
PAGE = PublicationExtent("city", "City", BBox(600000, 5818000, 601000, 5820000, "EPSG:32631"))


@pytest.fixture
def recipe(tmp_path):
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert probe["status"] == "ok" and not probe.get("software_fallback", False), probe
    values = np.arange(600, dtype=np.float32).reshape(20, 30)
    values[0, 0] = -9999
    path = tmp_path / "source.tif"
    gis.write_raster(path, values, crs="EPSG:32631", nodata=-9999, transform=TRANSFORM)
    source = PublicationSource(path, "Fixture", "2024", "v2", "https://example.test", "people", "missing")
    return PublicationRecipe(source, [100, 300], PALETTE, BOUNDS)


def _hits(receipt):
    return receipt["main_render"]["metadata"]["cache_hit"], receipt["locator_render"]["metadata"]["cache_hit"]


def _assert_cells_show_legend(receipt, values, breaks):
    """Saved page pixels vs NumPy classification, independent of the recipe."""
    image = np.asarray(Image.open(receipt["output"]["path"]).convert("RGBA")).astype(int)
    x0, y0, x1, y1 = receipt["map_rect"]
    rows, cols = values.shape
    ys = y0 + ((np.arange(rows) + .5) * (y1 - y0) / rows).astype(int)
    xs = x0 + ((np.arange(cols) + .5) * (x1 - x0) / cols).astype(int)
    classes = np.where(values != -9999, np.searchsorted(breaks, values, side="right") + 1, 0)
    palette = {entry["class_id"]: entry["rgba"] for entry in receipt["legend"]["entries"]}
    expected = np.array([palette[k] for k in range(len(palette))])[classes]
    bx0, by0, bx1, by1 = receipt["scale"]["bar_rect"]
    keep = ~((ys[:, None] >= by0) & (ys[:, None] < by1) & (xs[None] >= bx0) & (xs[None] < bx1))
    assert np.abs(image[np.ix_(ys, xs)] - expected)[keep].max() <= 1


def test_unchanged_inputs_reuse_frames_and_each_revision_misses(recipe, tmp_path):
    cache = tmp_path / "cache"
    values = np.arange(600, dtype=np.float32).reshape(20, 30)
    values[0, 0] = -9999
    def run(name, active=recipe, page=PAGE):
        return active.render_series([page], tmp_path / name, cache=cache)[0]

    first = run("first")
    assert _hits(first) == (False, False)
    assert first["main_render"]["metadata"]["frame_color_source"] == "native_albedo_srgb"
    _assert_cells_show_legend(first, values[:20, :10], recipe.breaks)

    same = run("same")
    assert _hits(same) == (True, True)
    assert same["output"]["pixel_sha256"] == first["output"]["pixel_sha256"]
    assert same["main_render"]["max_legend_deviation_levels"] == first["main_render"]["max_legend_deviation_levels"]

    # Same path, size and mtime: only the bytes identify the change.
    source = recipe.source.path
    before = source.stat()
    edited = values[::-1].copy()
    gis.write_raster(source, edited, crs="EPSG:32631", nodata=-9999, transform=TRANSFORM, overwrite=True)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (source.stat().st_size, source.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    changed = run("edited")
    assert _hits(changed) == (False, False)
    assert changed["output"]["pixel_sha256"] != first["output"]["pixel_sha256"]
    _assert_cells_show_legend(changed, edited[:20, :10], recipe.breaks)

    recipe.breaks = [50, 250]
    revised = run("breaks")
    assert _hits(revised) == (False, False)
    _assert_cells_show_legend(revised, edited[:20, :10], recipe.breaks)

    moved = PublicationExtent("city", "City", BBox(600100, 5818000, 601100, 5820000, "EPSG:32631"))
    assert _hits(run("extent", page=moved)) == (False, True)  # The locator's scene is unchanged.


def test_lit_and_unlit_frames_never_share_a_cache_entry(recipe, tmp_path):
    cache = tmp_path / "cache"
    recipe.render_series([PAGE], tmp_path / "first", cache=cache)
    values, valid, grid, _, description = recipe._read()
    classes, style, _, _ = recipe._classify(values, valid)
    lit = recipe._scene(classes, style, grid, (300, 200), description)
    unlit = recipe._scene(classes, style, grid, (300, 200), description)
    lit.recipe.lighting.settings = {"cli_params": {"camera_mode": "screen"}}
    unlit.render(str(tmp_path / "unlit.png"), cache=cache)
    lit.render(str(tmp_path / "lit.png"), cache=cache)
    assert not lit.last_render_metadata["cache_hit"]
    assert np.mean(np.asarray(Image.open(tmp_path / "lit.png"))[..., :3]) < np.mean(
        np.asarray(Image.open(tmp_path / "unlit.png"))[..., :3])


def test_unlit_frame_holds_only_opaque_legend_colours(recipe, tmp_path):
    values, valid, grid, _, description = recipe._read()
    classes, style, _, _ = recipe._classify(values, valid)
    scene = recipe._scene(classes, style, grid, (300, 200), description)
    scene.render(str(tmp_path / "frame.png"))
    assert scene.last_render_backend == "gpu_terrain"
    frame = np.asarray(Image.open(tmp_path / "frame.png").convert("RGBA"))
    assert (frame[..., 3] == 255).all()
    assert {tuple(colour) for colour in np.unique(frame.reshape(-1, 4), axis=0)} <= {
        tuple(rgba) for rgba in style.palette.values()}


def test_corrupt_cached_frame_is_never_published(recipe, tmp_path):
    cache = tmp_path / "cache"
    first = recipe.render_series([PAGE], tmp_path / "first", cache=cache)[0]
    files = [path for path in cache.rglob("*") if path.is_file() and path.stat().st_size > 1024]
    assert files
    for path in files:
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(data)
    try:
        again = recipe.render_series([PAGE], tmp_path / "again", cache=cache)[0]
    except (RuntimeError, ValueError, OSError):
        return  # Refusing corrupt storage is valid; publishing from it is not.
    assert again["output"]["pixel_sha256"] == first["output"]["pixel_sha256"]


@pytest.mark.parametrize("width", [60, 300])
def test_native_short_map_scale_matches_independent_geodesic(recipe, tmp_path, width):
    from dataclasses import replace
    from pyproj import Geod, Transformer
    cell = width / 10
    bounds = BBox(600000, 5820000 - width, 600000 + width, 5820000, "EPSG:32631")
    gis.write_raster(recipe.source.path, np.arange(100, dtype=np.float32).reshape(10, 10),
                     crs=bounds.crs, transform=(cell, 0, bounds.west, 0, -cell, bounds.north), overwrite=True)
    recipe.locator_bounds = bounds
    page = replace(PAGE, bounds=bounds)
    result = recipe.render_series([page], tmp_path / "short")[0]
    assert result["main_render"]["backend"] == "gpu_terrain"
    transform = Transformer.from_crs(bounds.crs, 4326, always_xy=True)
    lon, lat = transform.transform([bounds.west, bounds.east], [bounds.center_lat] * 2)
    mpp = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])[2] / result["main_viewport"][0]
    image = np.asarray(Image.open(result["output"]["path"]).convert("RGBA"))
    x0, y0, x1, y1 = result["scale"]["bar_rect"]
    border = np.flatnonzero(np.all(image[y0 + 12, x0:x1, :3] == 0, axis=1))
    number, unit = result["scale"]["bar_label"].split()
    assert unit == "m"
    assert abs(float(number) - (border[-1] - border[0] + 1) * mpp) <= mpp
