"""G-002 Later COG and slippy tile helper tests: local reads, basic remote
fetch/decode, overview validation, and slippy tile indexing. Range-streaming
tests live in ``test_gis_cog_range.py`` (striped) and
``test_gis_cog_range_tiled.py`` (tiled); shared HTTP servers and TIFF fixtures
live in ``_cog_http_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _cog_http_fixtures import _codes, _serve_bytes

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS COG/tile tests require the compiled _forge3d extension",
)


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


def test_read_cog_remote_fetches_and_decodes(tmp_path: Path):
    path = tmp_path / "remote_source.tif"
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    gis.write_raster(
        path,
        data,
        crs="EPSG:4326",
        transform=(1.0, 0.0, -2.0, 0.0, -1.0, 3.0),
    )
    local = gis.read_cog(path)

    server, url = _serve_bytes(path.read_bytes(), content_type="image/tiff")
    try:
        remote = gis.read_cog(url)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    assert remote["info"]["width"] == 4
    assert remote["info"]["height"] == 3


def test_read_cog_remote_unreachable_host_raises():
    # Remote COG reads are wired through the gis-remote fetch/cache path now that
    # the feature ships; an unreachable host must raise (never a silent success).
    with pytest.raises(RuntimeError):
        gis.read_cog("http://127.0.0.1:1/data.tif")


def test_read_cog_remote_overview_rejected_before_fetch():
    # overview is validated before any download: the unsupported overview must be
    # rejected (metadata_unavailable) without ever contacting the host, so the
    # error is the overview rejection, not a network error from the dead port.
    with pytest.raises(ValueError, match="metadata_unavailable|invalid_argument"):
        gis.read_cog("http://127.0.0.1:1/data.tif", overview=1)


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
    from forge3d._forge3d import clear_native_degradations, native_degradations

    clear_native_degradations()
    result = gis.slippy_tile_index((-10.0, 84.0, 10.0, 89.0), 2)

    assert "invalid_bounds" in _codes(result)
    assert result["bounds_wgs84"][3] == pytest.approx(85.05112878)
    assert result["tile_count"] > 0
    assert native_degradations() == [
        {
            "kind": "input_clamped",
            "name": "web_mercator_latitude",
            "consequence": "latitude bounds were clamped to the Web Mercator valid range",
        }
    ]
    clear_native_degradations()


def test_slippy_tile_index_does_not_record_normal_or_refused_bounds():
    from forge3d._forge3d import clear_native_degradations, native_degradations

    clear_native_degradations()
    normal = gis.slippy_tile_index((-10.0, -20.0, 10.0, 20.0), 2)
    assert "invalid_bounds" not in _codes(normal)
    assert native_degradations() == []

    # Antimeridian bounds are handled as an explicit split, not latitude clamp.
    split = gis.slippy_tile_index((170.0, -10.0, -170.0, 10.0), 2)
    assert split["antimeridian_split"] is True
    assert native_degradations() == []

    with pytest.raises(ValueError, match="antimeridian_bounds_unsupported"):
        gis.web_mercator_bounds((170.0, -10.0, -170.0, 10.0), "EPSG:4326")
    assert native_degradations() == []


def test_clamp_visibility_assertion_rejects_a_silent_implementation():
    """Negative control: a warning without the CENSOR record must fail red."""

    def assert_visible(result, degradations):
        assert "invalid_bounds" in _codes(result)
        assert any(
            item["kind"] == "input_clamped" and item["name"] == "web_mercator_latitude"
            for item in degradations
        ), "Web Mercator clamp warning was emitted without a CENSOR degradation record"

    silent_result = {
        "warnings": [
            {
                "code": "invalid_bounds",
                "message": "latitude was clamped to the Web Mercator valid range",
                "field": "bounds",
            }
        ]
    }
    with pytest.raises(AssertionError, match="without a CENSOR degradation record"):
        assert_visible(silent_result, [])


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
