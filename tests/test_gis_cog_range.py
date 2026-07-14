"""Remote COG range streaming over striped TIFFs: windowed reads, fallback
behavior for servers that ignore or mangle Range, and the range/cache
diagnostic codes. Tiled-COG range tests live in ``test_gis_cog_range_tiled.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _cog_http_fixtures import (
    _clear_cog_cache,
    _codes,
    _serve_bad_range,
    _serve_no_range,
    _serve_range,
    _tiled_uint16_fixture,
    _warning_message,
)

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS COG/tile tests require the compiled _forge3d extension",
)


def test_read_cog_remote_windowed_read_streams_only_needed_bytes(tmp_path: Path):
    # Multi-strip fixture (~5 strips of 512 rows each): a windowed remote read must
    # fetch only the overlapping strip(s) via HTTP range requests, not the full file.
    path = tmp_path / "multistrip.tif"
    data = (np.arange(2400 * 512, dtype=np.float32) % 1000.0).reshape(2400, 512)
    gis.write_raster(
        path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2400.0)
    )
    body = path.read_bytes()
    total = len(body)

    window = (0, 0, 512, 20)  # col_off, row_off, width, height -> rows 0..20 (strip 0)
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_range(body)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window
    assert remote["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    # Range streaming: only strip 0 (+ the IFD) is fetched, far less than the whole file.
    assert served["bytes"] < total // 2, (
        f"windowed remote read transferred {served['bytes']} of {total} bytes"
    )


@pytest.mark.parametrize(
    "window",
    [
        (0, 0, 512, 20),  # first strip, full width
        (100, 500, 300, 40),  # spans strip 0 and strip 1, column offset
        (50, 1030, 200, 30),  # middle strip (strip 2), column + row offset
        (0, 2380, 512, 20),  # last (partial) strip
    ],
)
def test_read_cog_remote_windowed_range_matches_local(tmp_path: Path, window):
    # A range-streamed window must be byte-identical to the local windowed read
    # across multi-strip spans, middle strips, column offsets, and the last strip.
    path = tmp_path / "ms.tif"
    data = (np.arange(2400 * 512, dtype=np.float32) % 997.0).reshape(2400, 512)
    gis.write_raster(
        path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2400.0)
    )
    local = gis.read_cog(path, window=window)

    server, url, _served = _serve_range(path.read_bytes())
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_read_cog_remote_windowed_range_dtypes_and_bands(tmp_path: Path, dtype):
    # Multi-band + non-float dtypes must assemble correctly through the strip path.
    path = tmp_path / f"mb_{np.dtype(dtype).name}.tif"
    band0 = (np.arange(3000 * 400) % 200).reshape(3000, 400).astype(dtype)
    band1 = ((np.arange(3000 * 400) * 3) % 200).reshape(3000, 400).astype(dtype)
    gis.write_raster(
        path,
        np.stack([band0, band1]),  # (bands, height, width)
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 3000.0),
    )
    window = (10, 900, 120, 40)  # spans a strip boundary, offset in both axes
    local = gis.read_cog(path, window=window)

    server, url, _served = _serve_range(path.read_bytes())
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["array"].shape == (2, 40, 120)


def test_read_cog_remote_windowed_falls_back_when_server_ignores_range(tmp_path: Path):
    # A server that ignores Range (always 200 full body) must NOT corrupt the read:
    # the range path rejects the non-206 response and read_cog falls back to a full
    # fetch, still returning the correct window.
    path = tmp_path / "ms.tif"
    data = (np.arange(2400 * 512, dtype=np.float32) % 991.0).reshape(2400, 512)
    gis.write_raster(
        path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2400.0)
    )
    window = (0, 0, 512, 20)
    local = gis.read_cog(path, window=window)

    server, url = _serve_no_range(path.read_bytes())
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window


@pytest.mark.parametrize("unit", ["Bytes", "BYTES"])
def test_read_cog_remote_case_varied_content_range_unit_stays_on_range_path(
    tmp_path: Path, unit
):
    # RFC 9110 §14.1: range-unit names are case-insensitive. A valid 206 whose
    # Content-Range says `Bytes ...` must be accepted by the range reader — not
    # rejected as invalid_range_response and downgraded to a full-object GET.
    path = tmp_path / "striped.tif"
    data = (np.arange(64 * 80, dtype=np.uint16).reshape(80, 64) % 4001).astype(np.uint16)
    gis.write_raster(path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 80.0))
    window = (5, 5, 20, 20)
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_range(path.read_bytes(), unit=unit)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window
    assert "range_read" in _codes(remote)
    assert "range_fallback" not in _codes(remote)
    assert served["full_requests"] == 0, (
        f"case-varied {unit!r} unit must not trigger a full-body GET: {served}"
    )


@pytest.mark.parametrize(
    "mode",
    [
        "missing_content_range",
        "wrong_start",
        "wrong_end",
        "wrong_total",
        "truncated_body",
        "oversized_body",
        "missing_space",
    ],
)
def test_read_cog_remote_invalid_206_rejected_then_falls_back(tmp_path: Path, mode):
    # An invalid 206 partial response must never enter the byte cache or decoder:
    # the range reader rejects it and read_cog falls back to the server's valid full
    # GET, still returning the correct window and reporting invalid_range_response.
    path = tmp_path / "striped.tif"
    data = (np.arange(64 * 80, dtype=np.uint16).reshape(80, 64) % 4000).astype(np.uint16)
    gis.write_raster(path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 80.0))
    body = path.read_bytes()
    window = (5, 5, 20, 20)
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_bad_range(body, mode)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window
    assert "range_fallback" in _codes(remote)
    assert "invalid_range_response" in _warning_message(remote, "range_fallback")
    assert served["full_requests"] == 1, f"expected exactly one full GET, got {served}"


def test_read_cog_remote_invalid_window_raises_without_full_fetch(tmp_path: Path):
    # An out-of-bounds window must raise the existing validation error after range
    # metadata establishes the dimensions — never triggering a full-body download.
    path = tmp_path / "striped.tif"
    data = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 3000).astype(np.uint16)
    gis.write_raster(path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 64.0))
    body = path.read_bytes()

    server, url, served = _serve_range(body)
    try:
        with pytest.raises(ValueError, match="window must be inside|invalid"):
            gis.read_cog(url, window=(1000, 1000, 10, 10))
    finally:
        server.shutdown()

    assert served["full_requests"] == 0, f"validation must not trigger a full GET: {served}"
    assert served["range_requests"] > 0, "dimensions must be read via range requests first"


def test_read_cog_remote_invalid_window_on_unsupported_layout_raises_without_full_fetch(
    tmp_path: Path,
):
    # Regression: window validation must run BEFORE the layout-support checks. A
    # 2-band grayscale TIFF is an unsupported layout for the windowed decoder, so
    # with the old order an out-of-bounds window hit BackendUnavailable first and
    # fell back to a FULL-OBJECT download before the invalid window ever raised.
    path = tmp_path / "two_band.tif"
    band = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 3000).astype(np.uint16)
    gis.write_raster(
        path,
        np.stack([band, band + 1]),
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 64.0),
    )
    body = path.read_bytes()

    server, url, served = _serve_range(body)
    try:
        with pytest.raises(ValueError, match="window must be inside|invalid"):
            gis.read_cog(url, window=(1000, 1000, 10, 10))
    finally:
        server.shutdown()

    assert served["full_requests"] == 0, f"validation must not trigger a full GET: {served}"
    assert served["range_requests"] > 0, "dimensions must be read via range requests first"


def test_read_cog_remote_range_read_diagnostics(tmp_path: Path):
    # A successful range read reports code `range_read` and NEVER a cache code.
    path = tmp_path / "tiled_small.tif"
    body = _tiled_uint16_fixture(path, size=256)
    window = (10, 10, 40, 40)

    server, url, served = _serve_range(body)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    codes = _codes(remote)
    assert "range_read" in codes
    assert "cache_miss" not in codes and "cache_hit" not in codes
    assert "HTTP range" in _warning_message(remote, "range_read")
    assert served["full_requests"] == 0


def test_read_cog_remote_range_fallback_cache_miss_diagnostics(tmp_path: Path):
    # A range failure reaching an uncached full fetch reports BOTH a range_fallback
    # (with a stable reason) and a distinct cache_miss.
    _clear_cog_cache()
    path = tmp_path / "striped.tif"
    data = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 3000).astype(np.uint16)
    gis.write_raster(path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 64.0))
    body = path.read_bytes()

    server, url = _serve_no_range(body)
    try:
        remote = gis.read_cog(url, window=(5, 5, 20, 20))
    finally:
        server.shutdown()

    codes = _codes(remote)
    assert "range_fallback" in codes
    assert "cache_miss" in codes
    assert "range_not_supported" in _warning_message(remote, "range_fallback")


def test_read_cog_remote_range_fallback_cache_hit_diagnostics(tmp_path: Path):
    # Priming the cache with a full read, then a windowed read whose range attempt
    # fails, yields BOTH a range_fallback and a cache_hit (distinct codes).
    _clear_cog_cache()
    path = tmp_path / "striped.tif"
    data = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 3000).astype(np.uint16)
    gis.write_raster(path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 64.0))
    body = path.read_bytes()

    server, url = _serve_no_range(body)
    try:
        gis.read_cog(url)  # full fetch, populates the per-URL cache (cache_miss)
        remote = gis.read_cog(url, window=(5, 5, 20, 20))  # range fails -> cache hit
    finally:
        server.shutdown()

    codes = _codes(remote)
    assert "range_fallback" in codes
    assert "cache_hit" in codes
    assert "range_not_supported" in _warning_message(remote, "range_fallback")
