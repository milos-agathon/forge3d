from dataclasses import replace
import json
from math import radians, tan
import os

import numpy as np
from PIL import Image
import pytest

import forge3d as f3d
from forge3d import map_scene as ms
from forge3d.map_plate import BBox
from test_publication import _set_gdal_metadata


RAW = np.array([
    [0, 20, 50, 50, 100, 100],
    [0, 20, 50, 50, 100, 100],
    [25, 25, -99997, -99997, 200, 200],
    [25, 25, -99997, -99997, 200, 200],
    [50, 50, 200, 200, -9999, -9999],
    [50, 50, 200, 200, -9999, -9999],
], dtype=np.int32)
PALETTE = ((255, 248, 233, 255), (244, 175, 100, 255), (201, 90, 70, 255),
           (111, 44, 57, 255), (120, 113, 158, 255), (210, 218, 220, 255))
BOUNDS = BBox(618000, 5483000, 618600, 5483600, "EPSG:32631")


@pytest.fixture
def binding(tmp_path):
    return _prepared_binding(tmp_path, RAW)


def _prepared_binding(tmp_path, raw):
    rows, cols = raw.shape
    bounds = replace(BOUNDS, east=BOUNDS.west + cols * 100, north=BOUNDS.south + rows * 100)
    path = tmp_path / f"counts-{rows}.tif"
    f3d.gis.write_raster(path, raw, crs=bounds.crs, nodata=-9999,
                         transform=(100, 0, bounds.west, 0, -100, bounds.north))
    _set_gdal_metadata(path, '<Item name="observation_year">2024</Item>'
                             '<Item name="measurement_field">residents</Item>'
                             '<Item name="measurement_units">people</Item>')
    source = f3d.PublicationSource(path, "Counts", "2024", "Fixture", "https://example.test",
                                   "people", "No record", {"measurement_field": "residents"})
    extent = f3d.PublicationExtent("test", "Test city", bounds)
    return f3d.ValueFaithfulRasterBinding(source, extent, "count", .5, (0, 50, 100, 200),
                                          PALETTE, -9999, -99997)


def test_measurement_units_states_and_canonical_bundle(binding, tmp_path):
    from forge3d.value_faithful import ValueFaithfulRasterBinding

    assert f3d.ValueFaithfulRasterBinding is ValueFaithfulRasterBinding
    assert "ValueFaithfulRasterBinding" in f3d.__all__
    saved = json.dumps(binding.to_dict(), allow_nan=False)
    assert json.dumps(ValueFaithfulRasterBinding.from_dict(json.loads(saved)).to_dict(),
                      allow_nan=False) == saved
    count = binding._resolve()
    density = replace(binding, measurement_type="density", height_scale=.005,
                      colour_edges=(0, 5000, 10000, 20000))._resolve()
    np.testing.assert_array_equal(count.values, np.where(RAW >= 0, RAW, np.nan))
    np.testing.assert_array_equal(density.values, np.where(RAW >= 0, RAW * 100, np.nan))
    np.testing.assert_array_equal(count.classes, density.classes)
    assert count.summary == density.summary == {
        "known_count_sum": 2540, "known_cells": 28, "known_zero_cells": 2,
        "withheld_cells": 4, "missing_cells": 4, "total_cells": 36,
        "known_fraction": 28 / 36, "cell_area_m2": 10000.0,
        "known_area_m2": 280000.0, "extent_area_m2": 360000.0,
    }
    assert count.classes[0].tolist() == [1, 2, 3, 3, 4, 4]
    assert (count.classes[2, 2], count.classes[-1, -1]) == (5, 6)
    scene = count.scene(f3d.OrbitCamera(distance=1200), (320, 320))
    scene.save_bundle(tmp_path / "one.forge3d")
    restored = f3d.MapScene.load_bundle(tmp_path / "one.forge3d")
    restored.save_bundle(tmp_path / "two.forge3d")
    for path in (tmp_path / "one.forge3d").rglob("*.json"):
        assert path.read_bytes() == (tmp_path / "two.forge3d" / path.relative_to(tmp_path / "one.forge3d")).read_bytes()
    assert restored.recipe.to_dict() == scene.recipe.to_dict()


def test_rejects_invalid_domain_units_fields_and_grids(binding):
    with pytest.raises(ValueError, match="height_scale"):
        replace(binding, height_scale=0)
    with pytest.raises(ValueError, match="colour domain"):
        replace(binding, colour_edges=(0, 50, 100, 199))._resolve()
    with pytest.raises(ValueError, match="measurement_units"):
        replace(binding, source=replace(binding.source, units="households"))._resolve()
    with pytest.raises(ValueError, match="measurement_field"):
        replace(binding, source=replace(binding.source, provenance={"measurement_field": "other"}))._resolve()
    with pytest.raises(ValueError, match="nodata"):
        replace(binding, missing_code=-8)._resolve()
    for bounds, error in (
        (replace(BOUNDS, crs="EPSG:28992"), "source CRS"),
        (replace(BOUNDS, west=BOUNDS.west + 1), "cell edges"),
        (replace(BOUNDS, west=BOUNDS.west - 100), "outside"),
        (replace(BOUNDS, west=BOUNDS.west + 100), "square extent"),
    ):
        with pytest.raises(ValueError, match=error):
            replace(binding, extent=replace(binding.extent, bounds=bounds))._resolve()


def test_mesh_rejects_cells_finer_than_native_centre_registration(tmp_path):
    # More than one cell per native mesh interval cannot guarantee a plateau
    # at each independent centre, even when every fixture value is the same.
    binding = _prepared_binding(tmp_path, np.zeros((512, 512), dtype=np.int32))
    with pytest.raises(ValueError, match="preserve their centre heights"):
        binding._resolve()


def test_camera_rejects_far_plane_and_partial_viewport_clipping(binding):
    active = binding._resolve()
    size = (480, 480)
    camera = f3d.OrbitCamera(distance=1200, elevation_deg=0, fov_deg=45)
    assert active._check_camera(active.scene(camera, size), size) == 0
    # A south-side view puts grid north straight ahead. An east-side view
    # points grid north to the right of the view centre.
    for azimuth, rotation in ((-90, 0), (0, 90)):
        oblique = replace(camera, elevation_deg=25, azimuth_deg=azimuth, near=800, far=1600)
        assert round(active._check_camera(active.scene(oblique, size), size)) == rotation
    for clipped in (replace(camera, distance=20_000), replace(camera, far=1150),
                    replace(camera, fov_deg=20), replace(camera, near=1150)):
        with pytest.raises(ValueError, match="Camera clips"):
            active._check_camera(active.scene(clipped, size), size)


def _check_nadir_depth(depth, pixels, raw, scale, camera, *, indices=None):
    rows, cols = raw.shape
    height, width = pixels.shape[:2]
    focal = height / 2 / tan(radians(camera.fov_deg / 2))
    # The clip planes bracket the declared relief. This avoids the loss of
    # precision from inverting perspective depth with the general 0.1 m near
    # plane at kilometre distances. The AOV stores normalized depth in f16.
    near, far = camera.near, camera.far
    for row, col in np.ndindex(raw.shape) if indices is None else indices:
        value = raw[row, col]
        view_depth = camera.distance - max(0, value) * scale
        x = int(width / 2 + (col + .5 - cols / 2) * 100 * focal / view_depth)
        y = int(height / 2 + (row + .5 - rows / 2) * 100 * focal / view_depth)
        if value < 0:
            np.testing.assert_array_equal(pixels[y, x], [0, 0, 0, 255])
            assert not np.any(depth[y, x])
            continue
        key = 0 if value == 0 else np.searchsorted([50, 100], value, side="right") + 1
        assert np.max(np.abs(pixels[y, x].astype(int) - PALETTE[key])) <= 1
        expected = (view_depth - near) / (far - near)
        actual = float(np.ravel(depth[y, x])[0])
        assert abs(actual - expected) <= float(np.spacing(np.float16(expected))), (row, col, actual, expected)


@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
                    reason="run in the physical publication GPU gate")
def test_native_consumes_height_and_masks_and_matches_linked_keys(binding, tmp_path, monkeypatch, pro_license):
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert probe["status"] == "ok" and probe["device_type"].lower() in ("discretegpu", "integratedgpu")
    assert not probe.get("software_fallback", False)
    observed = []
    render = ms._render_terrain_renderer_result

    def capture(recipe, heightmap, **kwargs):
        result = render(recipe, heightmap, **kwargs)
        if "depth" in recipe.output.aovs:
            frame = result.aov_frame
            observed.append((heightmap.copy(), kwargs.get("nodata_height_below"),
                             np.asarray(frame.albedo()).copy(), np.asarray(frame.depth()).copy()))
        return result

    monkeypatch.setattr(ms, "_render_terrain_renderer_result", capture)
    # Straight down at a known distance, so projected cell centres follow
    # an independent pinhole calculation. AOV depth measures real rasterized geometry.
    camera = f3d.OrbitCamera(target=(0, 0, 0), distance=1200, elevation_deg=0,
                             fov_deg=45, near=1000, far=1300)
    active = binding._resolve()
    scene = active.scene(camera, (480, 480), aovs=("albedo", "depth"))
    scene.render(str(tmp_path / "mesh.png"))
    submitted, threshold, albedo, depth = observed[-1]
    assert submitted.shape[0] % RAW.shape[0] == 0
    repeats = submitted.shape[0] // RAW.shape[0]
    counts = RAW.repeat(repeats, axis=0).repeat(repeats, axis=1)
    np.testing.assert_array_equal(submitted >= threshold, counts >= 0)
    np.testing.assert_array_equal(submitted[counts >= 0], counts[counts >= 0] * .5)
    assert scene.last_render_backend == "gpu_terrain"
    assert scene.last_render_metadata["thematic_backend"] == "native_terrain_rgba_cells"
    mesh = np.asarray(Image.open(tmp_path / "mesh.png").convert("RGBA"))
    _check_nadir_depth(depth, mesh, RAW, .5, camera)
    np.testing.assert_array_equal(albedo[np.all(mesh[..., :3] == 0, axis=-1)], 0)
    revised = replace(binding, height_scale=.25)._resolve().scene(camera, (480, 480), aovs=("depth",))
    revised.render(str(tmp_path / "shorter.png"))
    shorter = np.asarray(Image.open(tmp_path / "shorter.png").convert("RGBA"))
    _check_nadir_depth(observed[-1][-1], shorter, RAW, .25, camera)

    page, = binding.render_series({"top": camera}, tmp_path)
    assert page["legend"]["height_transform"]["factor"] == .5
    assert page["legend"]["height_reference"] == [
        {"value": 0, "height_m": 0}, {"value": 100, "height_m": 50}, {"value": 200, "height_m": 100},
    ]
    assert "Height key (people / cell)  0 = 0 m, 100 = 50 m, 200 = 100 m." in page["caption"]
    assert page["grid_north_rotation_deg"] == 0
    assert "In this 0.6 km square, published counts sum to 2,540" in page["caption"]
    x0, y0, x1, y1 = page["inset_rect"]
    image = np.asarray(Image.open(page["output"]["path"]).convert("RGBA"))
    map_left, map_top, *_ = page["map_rect"]
    assert np.any(np.max(image[:map_top, map_left:map_left + 64, :3], axis=-1) == 0)
    xs = (x0 + (np.arange(6) + .5) * (x1 - x0) / 6).astype(int)
    ys = (y0 + (np.arange(6) + .5) * (y1 - y0) / 6).astype(int)
    ids = np.searchsorted([50, 100], RAW, side="right") + 1
    ids[RAW == 0] = 0
    ids[RAW == -99997] = 4
    ids[RAW == -9999] = 5
    assert np.abs(image[np.ix_(ys, xs)].astype(int) - np.asarray(PALETTE)[ids]).max() <= 1
    for colour in PALETTE:
        assert np.any(np.all(image[:y0, x0:x1] == colour, axis=-1))

    # Isolated contrasting cells exercise the same native mesh and source
    # spacing as the Amsterdam window, including centres near its edges.
    large = np.zeros((80, 80), dtype=np.int32)
    controls = {(r, c): 200 for r, c in ((0, 0), (1, 76), (39, 39), (51, 68), (78, 0), (79, 79))}
    controls.update({(38, 39): -99997, (50, 68): -9999, (30, 30): 50, (37, 39): 0})
    for cell, value in controls.items():
        large[cell] = value
    # Leave room in the CENSOR projection-matrix range. In a perspective
    # matrix the translation magnitude is near * far / (far - near).
    large_camera = replace(camera, distance=12_000, near=9500, far=12_200)
    large_scene = _prepared_binding(tmp_path, large)._resolve().scene(large_camera, (1200, 1200), aovs=("depth",))
    large_scene.render(str(tmp_path / "city-size.png"))
    large_image = np.asarray(Image.open(tmp_path / "city-size.png").convert("RGBA"))
    _check_nadir_depth(observed[-1][-1], large_image, large, .5, large_camera, indices=controls)
