from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("geopandas")

from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "population_ghsl" / "iberia_builtup_cover_3d.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("iberia_builtup_cover_3d", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_population_raster_is_written_on_dem_grid(tmp_path):
    module = _load_example_module()
    pop_path = tmp_path / "population_4326.tif"
    dem_path = tmp_path / "dem_3035.tif"
    out_path = tmp_path / "population_on_dem.tif"

    lonlat_bounds = (-10.0, 35.0, 4.0, 44.0)
    with rasterio.open(
        pop_path,
        "w",
        driver="GTiff",
        width=14,
        height=9,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(*lonlat_bounds, 14, 9),
        nodata=0.0,
    ) as dst:
        dst.write(np.ones((9, 14), dtype=np.float32), 1)

    dem_bounds = transform_bounds("EPSG:4326", "EPSG:3035", *lonlat_bounds, densify_pts=21)
    dem_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    dem_data[0, 0] = -9999.0
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_bounds(*dem_bounds, 4, 3),
        nodata=-9999.0,
    ) as dst:
        dst.write(dem_data, 1)

    module._write_population_on_dem_grid(
        pop_path,
        box(*lonlat_bounds),
        dem_path,
        out_path,
        force=True,
    )

    with rasterio.open(dem_path) as dem, rasterio.open(out_path) as pop:
        assert pop.crs == dem.crs
        assert pop.transform.almost_equals(dem.transform)
        assert (pop.width, pop.height) == (dem.width, dem.height)
        data = pop.read(1)
        assert data[0, 0] == pytest.approx(pop.nodata)
        assert np.any(data[1:, 1:] > 0.0)


def test_population_rgba_shades_yellow_by_dem_relief(monkeypatch):
    module = _load_example_module()
    population = np.array([[0.0, 0.09], [0.1, 5.0]], dtype=np.float32)
    dem = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    relief = np.array([[0.46, 0.46], [0.46, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        module.base_viewer,
        "_height_shade_from_dem",
        lambda heightmap: relief,
    )

    rgba = module._population_rgba(population, dem, valid)
    terrain_rgb = module._terrain_base_rgb(dem, valid)
    active = valid & np.isfinite(population) & (population >= 0.1)

    assert rgba[0, 0, 3] == 255
    assert rgba[0, 1, 3] == 0
    np.testing.assert_array_equal(rgba[~active, :3], terrain_rgb[~active])
    low_relief = tuple(rgba[1, 0, :3])
    high_relief = tuple(rgba[1, 1, :3])
    assert low_relief != high_relief
    assert low_relief[0] < high_relief[0]
    assert low_relief[1] < high_relief[1]
    assert low_relief == (128, 106, 0)
    assert high_relief == (255, 211, 1)
    assert tuple(rgba[0, 0, :3]) != (255, 255, 255)


def test_terrain_base_rgb_bakes_reference_height_shade(monkeypatch):
    module = _load_example_module()
    dem = np.array(
        [
            [100.0, 150.0, 200.0],
            [250.0, 300.0, 350.0],
            [400.0, 450.0, 500.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(dem, dtype=bool)
    monkeypatch.setattr(
        module.base_viewer,
        "_height_shade_from_dem",
        lambda heightmap: np.zeros_like(heightmap, dtype=np.float32),
    )

    rgb = module._terrain_base_rgb(dem, valid)

    values = dem[valid].astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    norm = np.clip((dem - low) / (high - low), 0.0, 1.0)
    scaled = norm * (len(module.TERRAIN_PALETTE) - 1)
    idx = np.clip(np.floor(scaled).astype(np.int16), 0, len(module.TERRAIN_PALETTE) - 2)
    frac = scaled - idx
    expected = module.TERRAIN_PALETTE[idx] * (1.0 - frac[:, :, None]) + module.TERRAIN_PALETTE[idx + 1] * frac[:, :, None]
    expected *= module.TERRAIN_SHADE_FLOOR
    expected_rgb = np.round(np.clip(expected, 0.0, 255.0)).astype(np.uint8)
    np.testing.assert_array_equal(rgb, expected_rgb)


def test_render_preserves_reference_height_shade_colors(tmp_path, monkeypatch):
    module = _load_example_module()
    dem_path = tmp_path / "dem.tif"
    overlay_path = tmp_path / "overlay.png"
    snapshot_path = tmp_path / "snapshot.png"
    overlay_path.write_bytes(b"not inspected by render test")
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_bounds(0.0, 0.0, 4.0, 3.0, 4, 3),
    ) as dst:
        dst.write(np.ones((3, 4), dtype=np.float32), 1)

    class FakeViewer:
        def __init__(self):
            self.commands = []
            self.overlay_calls = []
            self.snapshots = []

        def send_ipc(self, command):
            self.commands.append(command)

        def load_overlay(self, name, path, **kwargs):
            self.overlay_calls.append((name, path, kwargs))

        def snapshot(self, path, **kwargs):
            self.snapshots.append((path, kwargs))

    class FakeViewerContext:
        def __init__(self, viewer):
            self.viewer = viewer

        def __enter__(self):
            return self.viewer

        def __exit__(self, exc_type, exc, tb):
            return False

    viewer = FakeViewer()
    monkeypatch.setattr(module.f3d, "open_viewer_async", lambda **kwargs: FakeViewerContext(viewer))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "_compose_snapshot", lambda raw_path, output_path: None)

    module._render(snapshot_path, dem_path, overlay_path)

    assert viewer.overlay_calls
    assert viewer.overlay_calls[0][2]["preserve_colors"] is True
    preserve_commands = [
        command for command in viewer.commands
        if command.get("cmd") == "set_overlay_preserve_colors"
    ]
    assert preserve_commands == [{"cmd": "set_overlay_preserve_colors", "preserve_colors": True}]
