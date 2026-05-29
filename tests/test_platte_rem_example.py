from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio.enums")
pytest.importorskip("rasterio.io")
pytest.importorskip("rasterio.transform")
pytest.importorskip("rasterio.warp")
pytest.importorskip("rasterio.windows")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
PYTHON_DIR = ROOT / "python"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import platte_rem_forge3d as mod


def test_expand_bounds_to_square_returns_equal_sides() -> None:
    bounds = (100.0, 200.0, 130.0, 220.0)
    square = mod._expand_bounds_to_square(bounds, pad_fraction=0.0)
    width = square[2] - square[0]
    height = square[3] - square[1]
    assert abs(width - height) < 1e-6


def test_aggregate_mean_uses_block_means() -> None:
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = mod._aggregate_mean(data, factor=2)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_filter_paths_to_main_corridor_drops_off_axis_lines() -> None:
    center = (0.0, 0.0)
    on_axis = np.array([[-10.0, -10.0], [10.0, 10.0]], dtype=np.float64)
    near_axis = np.array([[-12.0, -6.0], [8.0, 14.0]], dtype=np.float64)
    off_axis = np.array([[-10.0, 40.0], [10.0, 60.0]], dtype=np.float64)
    kept = mod._filter_paths_to_main_corridor(
        [on_axis, near_axis, off_axis],
        center_xy=center,
        axis_deg=45.0,
        half_band_m=12.0,
    )
    assert len(kept) == 2
    assert any(np.array_equal(path, on_axis) for path in kept)
    assert any(np.array_equal(path, near_axis) for path in kept)


def test_idw_surface_preserves_constant_field() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )
    values = np.full(4, 7.5, dtype=np.float32)
    gx, gy = np.meshgrid(np.linspace(0.0, 10.0, 5), np.linspace(0.0, 10.0, 5))
    surface = mod._idw_surface(points, values, gx, gy, power=2.0)
    np.testing.assert_allclose(surface, 7.5, rtol=1e-5, atol=1e-5)


def test_build_base_plate_resizes_and_returns_uint8_rgb() -> None:
    rem = np.linspace(0.0, 12.0, 16, dtype=np.float32).reshape(4, 4)
    plate = mod._build_base_plate(rem, output_size=(8, 8))
    assert plate.shape == (8, 8, 3)
    assert plate.dtype == np.uint8
