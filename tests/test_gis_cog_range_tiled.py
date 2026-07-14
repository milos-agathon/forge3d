"""Remote COG range streaming over TILED TIFFs: per-tile fetches, transfer
proportionality, multiband/dtype fidelity, and fatal-header propagation. The
tiled fixture is hand-rolled in ``_cog_http_fixtures`` (no GDAL/rasterio).
Striped range tests live in ``test_gis_cog_range.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _cog_http_fixtures import (
    _codes,
    _serve_range,
    _tiled_uint16_fixture,
    _write_tiled_tiff,
)

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS COG/tile tests require the compiled _forge3d extension",
)


def test_read_cog_remote_tiled_window_inside_one_tile_streams_less_than_half(tmp_path: Path):
    # A window fully inside a single 64x64 tile of a large tiled COG must stream
    # only the intersecting tile (+ header/IFD), byte-identical to the local read,
    # with no full-body GET and far less than half the file transferred.
    path = tmp_path / "tiled_u16.tif"
    body = _tiled_uint16_fixture(path)
    total = len(body)
    window = (100, 100, 40, 40)  # inside tile (row 1, col 1)
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_range(body)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window
    assert remote["info"]["crs_authority"] == {"name": "EPSG", "code": "4326"}
    # Lock that the fixture actually exercised the TILED path — a striped
    # substitute would otherwise satisfy every byte/request assertion here.
    assert remote["info"]["tiling"] == "tiled"
    assert served["full_requests"] == 0, f"unexpected full-body GET: {served}"
    assert served["range_requests"] > 0
    assert served["bytes"] < total // 10, (
        f"tiled windowed read transferred {served['bytes']} of {total} bytes"
    )
    assert "range_read" in _codes(remote)


def test_read_cog_remote_tiled_window_spans_tile_boundaries(tmp_path: Path):
    # A window spanning several tile rows/columns must assemble byte-identically to
    # the local read, still fetching only intersecting tiles (no full-body GET).
    path = tmp_path / "tiled_u16.tif"
    body = _tiled_uint16_fixture(path)
    window = (50, 50, 140, 140)  # crosses tile boundaries in both axes
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_range(body)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["window"] == window
    assert remote["info"]["tiling"] == "tiled"
    assert served["full_requests"] == 0, f"unexpected full-body GET: {served}"
    assert "range_read" in _codes(remote)


def _intersecting_tile_payload(window, *, tile: int = 64, bytes_per_pixel: int = 2) -> int:
    """Total uncompressed payload bytes of the fixture tiles a window touches."""
    col_off, row_off, width, height = window
    tiles_x = (col_off + width + tile - 1) // tile - col_off // tile
    tiles_y = (row_off + height + tile - 1) // tile - row_off // tile
    return tiles_x * tiles_y * tile * tile * bytes_per_pixel


def test_read_cog_remote_tiled_large_window_transfer_is_proportional(tmp_path: Path):
    # Regression: the range cursor used to issue a fresh UNALIGNED 64 KiB fetch per
    # tile seek that could never hit the exact-(offset, length) byte cache, so this
    # exact window transferred 156% of the file. Transport is now block-aligned and
    # cached: traffic must stay proportional to the intersecting tile payloads
    # (block-granularity rounding + header/IFD allowance), far below the file size.
    path = tmp_path / "tiled_u16.tif"
    body = _tiled_uint16_fixture(path)
    total = len(body)
    window = (300, 300, 400, 400)  # 7x7 tiles of the 16x16 tile grid
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_range(body)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["info"]["tiling"] == "tiled"
    assert served["full_requests"] == 0, f"unexpected full-body GET: {served}"
    assert "range_read" in _codes(remote)
    payload = _intersecting_tile_payload(window)
    assert served["bytes"] < 2 * payload, (
        f"transferred {served['bytes']} bytes for {payload} bytes of intersecting tiles"
    )
    assert served["bytes"] < total // 2, (
        f"large tiled window transferred {served['bytes']} of {total} bytes"
    )


def test_read_cog_remote_corrupt_header_raises_without_full_fetch():
    # A served object that is not a TIFF at all is a FATAL decode error — a full
    # fetch of the same bytes would fail identically — so it must propagate and
    # never take the fallback/full-download/cache path.
    body = b"not-a-tiff" + bytes(100_000)

    server, url, served = _serve_range(body)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            gis.read_cog(url, window=(0, 0, 4, 4))
    finally:
        server.shutdown()

    assert served["full_requests"] == 0, (
        f"corrupt header must not trigger a full-body GET: {served}"
    )


@pytest.mark.parametrize(
    "dtype, bands",
    [(np.uint8, 3), (np.uint16, 3), (np.int16, 1)],
)
def test_read_cog_remote_tiled_multiband_dtypes_match_local(tmp_path: Path, dtype, bands):
    # Multi-band (RGB) and non-float dtypes must assemble correctly through the
    # tiled range path, byte-identical to the local read, with no full-body GET.
    height, width = 300, 300
    base = np.arange(height * width, dtype=np.int64).reshape(height, width)
    if bands == 3:
        array = np.stack([((base * (k + 1)) % 250).astype(dtype) for k in range(3)])
    else:
        array = ((base % 500) - 250).astype(dtype)
    path = tmp_path / f"tiled_{np.dtype(dtype).name}_{bands}b.tif"
    _write_tiled_tiff(path, array, transform=(1.0, 0.0, 0.0, 0.0, -1.0, float(height)), tile=64)
    body = path.read_bytes()
    window = (40, 40, 120, 100)  # col_off, row_off, width, height -> spans tiles
    local = gis.read_cog(path, window=window)

    server, url, served = _serve_range(body)
    try:
        remote = gis.read_cog(url, window=window)
    finally:
        server.shutdown()

    np.testing.assert_array_equal(remote["array"], local["array"])
    assert remote["array"].shape == (bands, 100, 120)
    assert remote["info"]["tiling"] == "tiled"
    assert served["full_requests"] == 0, f"unexpected full-body GET: {served}"
    assert "range_read" in _codes(remote)
