"""G-002 Later COG and slippy tile helper tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS COG/tile tests require the compiled _forge3d extension",
)


def _codes(result) -> set[str]:
    return {warning["code"] for warning in result.get("warnings", [])}


def test_read_cog_local_base_resolution(tmp_path: Path):
    path = tmp_path / "local_cog_like.tif"
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(1.0, 0.0, -2.0, 0.0, -1.0, 3.0),
    )

    result = gis.read_cog(path)

    np.testing.assert_array_equal(result["array"], data.reshape(1, 3, 4))
    assert result["overview"] is None
    assert result["window"] is None
    assert result["is_cog_like"] is False
    assert result["tile_info"]["tiling"] in {"striped", "tiled"}
    assert result["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}


def test_read_cog_local_window(tmp_path: Path):
    path = tmp_path / "window.tif"
    data = np.arange(20, dtype=np.uint8).reshape(4, 5)
    gis.write_raster(path, data, transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0))

    result = gis.read_cog(path, window=(1, 1, 3, 2))

    np.testing.assert_array_equal(result["array"], data[1:3, 1:4].reshape(1, 2, 3))
    assert result["window"] == (1, 1, 3, 2)
    assert result["info"]["width"] == 3
    assert result["info"]["height"] == 2
    assert result["info"]["transform"] == pytest.approx((2.0, 0.0, 12.0, 0.0, -2.0, 18.0))


def test_read_cog_rejects_remote_without_cog_streaming():
    with pytest.raises(RuntimeError, match="backend_unavailable.*cog_streaming"):
        gis.read_cog("https://example.com/data.tif")


def test_read_cog_rejects_overview_for_local_reader(tmp_path: Path):
    path = tmp_path / "overview.tif"
    gis.write_raster(path, np.ones((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="metadata_unavailable|invalid_argument"):
        gis.read_cog(path, overview=1)


def test_slippy_tile_index_known_zoom_schema_and_sorting():
    result = gis.slippy_tile_index((-1.0, -1.0, 1.0, 1.0), 1)

    assert result["zoom"] == 1
    assert result["crs"] == "EPSG:4326"
    assert result["antimeridian_split"] is False
    assert [(tile["z"], tile["x"], tile["y"]) for tile in result["tiles"]] == sorted(
        (tile["z"], tile["x"], tile["y"]) for tile in result["tiles"]
    )
    assert {tile["z"] for tile in result["tiles"]} == {1}
    assert {tile["x"] for tile in result["tiles"]} <= {0, 1}
    assert {tile["y"] for tile in result["tiles"]} <= {0, 1}
    assert all("bounds_wgs84" in tile for tile in result["tiles"])
    assert all("bounds_web_mercator" in tile for tile in result["tiles"])


def test_slippy_tile_index_clamps_web_mercator_latitude():
    result = gis.slippy_tile_index((-10.0, 84.0, 10.0, 89.0), 2)

    assert "invalid_bounds" in _codes(result)
    assert result["bounds_wgs84"][3] == pytest.approx(85.05112878)
    assert result["tile_count"] > 0


def test_slippy_tile_index_antimeridian_split():
    result = gis.slippy_tile_index((170.0, -10.0, -170.0, 10.0), 2)

    assert result["antimeridian_split"] is True
    assert result["tile_count"] > 0
    assert all(0 <= tile["x"] < 4 for tile in result["tiles"])


@pytest.mark.parametrize("zoom", [-1, 25, 1.5])
def test_slippy_tile_index_rejects_invalid_zoom(zoom):
    with pytest.raises((TypeError, ValueError), match="invalid_argument|argument"):
        gis.slippy_tile_index((-1.0, -1.0, 1.0, 1.0), zoom)


def test_slippy_tile_index_rejects_impossible_latitude():
    with pytest.raises(ValueError, match="invalid_bounds"):
        gis.slippy_tile_index((-1.0, -91.0, 1.0, 1.0), 1)
