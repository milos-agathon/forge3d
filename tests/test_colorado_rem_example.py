from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio.features")
pytest.importorskip("rasterio.io")
pytest.importorskip("rasterio.transform")
pytest.importorskip("rasterio.warp")
pytest.importorskip("rasterio.windows")
pytest.importorskip("shapely.geometry")

from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
PYTHON_DIR = ROOT / "python"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import colorado_rem_forge3d as mod


def test_estimate_neighbor_count_increases_with_sinuosity() -> None:
    straight = mod._estimate_neighbor_count(interp_pts=1000, river_pixel_count=240, sinuosity=1.0)
    sinuous = mod._estimate_neighbor_count(interp_pts=1000, river_pixel_count=240, sinuosity=3.0)
    assert straight >= 5
    assert sinuous > straight
    assert sinuous <= 100


def test_idw_chunk_exact_preserves_constant_field() -> None:
    river_coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    river_values = np.full(4, 7.5, dtype=np.float32)
    query_coords = np.array(
        [
            [2.0, 2.0],
            [5.0, 5.0],
            [8.0, 3.0],
        ],
        dtype=np.float32,
    )
    interpolated = mod._idw_chunk_exact(
        river_coords,
        river_values,
        query_coords,
        k=4,
        power=1.0,
    )
    np.testing.assert_allclose(interpolated, 7.5, rtol=1e-5, atol=1e-5)


def test_sample_centerline_points_returns_points_and_endpoints() -> None:
    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (15.0, 5.0)]),
    ]
    points, endpoints = mod._sample_centerline_points(lines, interp_pts=12)
    assert len(points) >= 4
    assert all(isinstance(point, Point) for point in points)
    assert endpoints.shape == (4, 2)


def test_iter_line_parts_flattens_nested_linework() -> None:
    nested = GeometryCollection(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            MultiLineString(
                [
                    [(1.0, 0.0), (2.0, 0.0)],
                    [(2.0, 0.0), (3.0, 0.0)],
                ]
            ),
        ]
    )
    parts = mod._iter_line_parts(nested)
    assert len(parts) == 3
    assert all(isinstance(part, LineString) for part in parts)
