"""G-002 Later COG and slippy tile helper tests."""

from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


def _serve_bytes(body: bytes, *, content_type: str = "image/tiff"):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

    server = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/data.tif"


def _serve_range(body: bytes):
    """Serve `body` over HTTP with Range support, counting total bytes served."""
    served = {"bytes": 0, "requests": 0}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-type", "image/tiff")
            self.send_header("content-length", str(len(body)))
            self.send_header("accept-ranges", "bytes")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            served["requests"] += 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start_s, end_s = rng[len("bytes=") :].split("-")
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                end = min(end, len(body) - 1)
                chunk = body[start : end + 1]
                self.send_response(206)
                self.send_header("content-type", "image/tiff")
                self.send_header("content-range", f"bytes {start}-{end}/{len(body)}")
                self.send_header("content-length", str(len(chunk)))
                self.send_header("connection", "close")
                self.end_headers()
                served["bytes"] += len(chunk)
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("content-type", "image/tiff")
                self.send_header("content-length", str(len(body)))
                self.send_header("connection", "close")
                self.end_headers()
                served["bytes"] += len(body)
                self.wfile.write(body)

        def log_message(self, *_args):
            return

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/data.tif", served


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
