from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from PIL import Image
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "bosnia_terrain_landcover_viewer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bosnia_terrain_landcover_viewer", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    forge3d_stub = types.ModuleType("forge3d")
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = False
    previous_forge3d = sys.modules.get("forge3d")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
        added_examples_dir = True
    sys.modules["forge3d"] = forge3d_stub
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_overlay_cache_path_tracks_landcover_opacity(tmp_path: Path) -> None:
    module = load_module()

    overlay_path = module._overlay_cache_path(tmp_path)

    assert overlay_path.parent == tmp_path
    assert module.LANDCOVER_OVERLAY_CACHE_KEY in overlay_path.name
    assert "display-v" in overlay_path.name
    assert overlay_path.name.startswith("bih_landcover_overlay_")
    assert overlay_path.suffix == ".png"


def test_build_overlay_uses_display_palette_and_reports_present_classes(tmp_path: Path) -> None:
    module = load_module()
    classes_path = tmp_path / "classes.tif"
    dem_path = tmp_path / "dem.tif"
    overlay_path = tmp_path / "overlay.png"
    classes = np.array([[1, 0], [5, 11]], dtype=np.uint8)
    dem = np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)

    with rasterio.open(
        classes_path,
        "w",
        driver="GTiff",
        width=classes.shape[1],
        height=classes.shape[0],
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=0,
    ) as dst:
        dst.write(classes, 1)

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=dem.shape[1],
        height=dem.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)

    built_path, present_classes = module._build_overlay(classes_path, dem_path, overlay_path, force=False)
    rgba = np.asarray(Image.open(built_path).convert("RGBA"), dtype=np.uint8)
    expected_rgba = module._classes_to_rgba(classes, dem)

    assert built_path == overlay_path
    assert present_classes == [1, 5, 11]
    assert tuple(int(v) for v in rgba[0, 0]) == tuple(int(v) for v in expected_rgba[0, 0])
    assert tuple(int(v) for v in rgba[0, 1]) == (0, 0, 0, 0)
    assert tuple(int(v) for v in rgba[1, 0]) == tuple(int(v) for v in expected_rgba[1, 0])
    assert tuple(int(v) for v in rgba[1, 1]) == tuple(int(v) for v in expected_rgba[1, 1])


def test_landcover_alpha_uses_actual_transparency_scale() -> None:
    module = load_module()

    assert module._landcover_alpha(0.0) == 0
    assert module._landcover_alpha(0.5) == 128
    assert module._landcover_alpha(1.0) == 255


def test_hillshade_uses_geographic_east_west_direction() -> None:
    module = load_module()
    east_rising_heightmap = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))

    shade_from_east = module._hillshade(
        east_rising_heightmap,
        azimuth_deg=90.0,
        elevation_deg=30.0,
        z_factor=1.0,
    )
    shade_from_west = module._hillshade(
        east_rising_heightmap,
        azimuth_deg=270.0,
        elevation_deg=30.0,
        z_factor=1.0,
    )

    assert float(shade_from_west.mean()) > float(shade_from_east.mean())


def test_shadow_offset_tracks_sun_direction() -> None:
    module = load_module()

    dx, dy = module._shadow_offset_from_sun(
        (1000, 800),
        azimuth_deg=314.0,
        elevation_deg=28.0,
        distance_ratio=0.02,
    )

    assert dx > 0
    assert dy > 0
