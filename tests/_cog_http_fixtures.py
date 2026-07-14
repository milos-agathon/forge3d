"""Shared HTTP servers and TIFF fixtures for the remote-COG range-read tests.

Used by ``test_gis_cog_tiles.py``, ``test_gis_cog_range.py``, and
``test_gis_cog_range_tiled.py``. Everything here is fixture/transport plumbing
only — no assertions live in this module.
"""

from __future__ import annotations

import http.server
import socketserver
import struct
import threading
from pathlib import Path

import numpy as np


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


def _serve_range(body: bytes, *, unit: str = "bytes"):
    """Serve `body` over HTTP with Range support, counting bytes and requests.

    `served` tracks total bytes, total GETs, and — separately — how many GETs
    carried a Range header (`range_requests`) vs. how many fetched the whole body
    (`full_requests`), so tests can assert a windowed read never fell back to a
    full-body download.

    `unit` is the range-unit name echoed in the Content-Range header. RFC 9110
    §14.1 makes the name case-insensitive, so tests pass e.g. ``unit="Bytes"``
    to lock that a case-varied valid response stays on the range path.
    """
    served = {"bytes": 0, "requests": 0, "range_requests": 0, "full_requests": 0}

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
                served["range_requests"] += 1
                start_s, end_s = rng[len("bytes=") :].split("-")
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                end = min(end, len(body) - 1)
                chunk = body[start : end + 1]
                self.send_response(206)
                self.send_header("content-type", "image/tiff")
                self.send_header("content-range", f"{unit} {start}-{end}/{len(body)}")
                self.send_header("content-length", str(len(chunk)))
                self.send_header("connection", "close")
                self.end_headers()
                served["bytes"] += len(chunk)
                self.wfile.write(chunk)
            else:
                served["full_requests"] += 1
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


def _serve_no_range(body: bytes):
    """Answer HEAD, but IGNORE the Range header and always return 200 full body."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-type", "image/tiff")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/data.tif"


def _serve_bad_range(body: bytes, mode: str):
    """Serve a *malformed* 206 for Range GETs (per `mode`) but a valid 200 full body.

    Lets a windowed read attempt the range path, hit an invalid partial response,
    and fall back to the ordinary full GET which still decodes correctly.
    """
    served = {"range_requests": 0, "full_requests": 0}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-type", "image/tiff")
            self.send_header("content-length", str(len(body)))
            self.send_header("accept-ranges", "bytes")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                served["range_requests"] += 1
                start_s, end_s = rng[len("bytes=") :].split("-")
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                end = min(end, len(body) - 1)
                chunk = body[start : end + 1]
                self.send_response(206)
                self.send_header("content-type", "image/tiff")
                if mode == "missing_content_range":
                    pass
                elif mode == "wrong_start":
                    self.send_header("content-range", f"bytes {start + 1}-{end}/{len(body)}")
                elif mode == "wrong_end":
                    self.send_header("content-range", f"bytes {start}-{end + 1}/{len(body)}")
                elif mode == "wrong_total":
                    self.send_header("content-range", f"bytes {start}-{end}/{len(body) + 1}")
                elif mode == "truncated_body":
                    self.send_header("content-range", f"bytes {start}-{end}/{len(body)}")
                    chunk = chunk[:-1]
                elif mode == "oversized_body":
                    self.send_header("content-range", f"bytes {start}-{end}/{len(body)}")
                    chunk = chunk + b"\x00"
                elif mode == "missing_space":
                    # Violates the RFC 9110 `range-unit SP range-resp` grammar.
                    self.send_header("content-range", f"bytes{start}-{end}/{len(body)}")
                else:  # pragma: no cover - guard against typos in parametrization
                    raise AssertionError(f"unknown mode {mode!r}")
                self.send_header("content-length", str(len(chunk)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(chunk)
            else:
                served["full_requests"] += 1
                self.send_response(200)
                self.send_header("content-type", "image/tiff")
                self.send_header("content-length", str(len(body)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_args):
            return

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/data.tif", served


_TIFF_DTYPES = {
    np.dtype(np.uint8): (8, 1),
    np.dtype(np.uint16): (16, 1),
    np.dtype(np.int16): (16, 2),
    np.dtype(np.uint32): (32, 1),
    np.dtype(np.int32): (32, 2),
    np.dtype(np.float32): (32, 3),
    np.dtype(np.float64): (64, 3),
}


def _write_tiled_tiff(path: Path, array: np.ndarray, *, transform, tile: int = 64) -> None:
    """Write a minimal, uncompressed, chunky-planar tiled GeoTIFF (EPSG:4326).

    Hand-rolled with ``struct`` (no GDAL/rasterio) the same way ``_write_png`` is
    in ``test_gis_domain.py`` — the in-repo TIFF encoder only emits striped files,
    so this is the smallest way to exercise the tiled range-read path. Supports 1
    band (grayscale) or 3 bands (RGB), across every dtype the reader supports, and
    pads edge tiles so images need not be a multiple of the tile size.
    """
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None, :, :]
    bands, height, width = a.shape
    assert bands in (1, 3), "tiled fixture supports 1 or 3 bands"
    dt = np.dtype(a.dtype)
    bits, sample_format = _TIFF_DTYPES[dt]
    photometric = 1 if bands == 1 else 2  # 1=BlackIsZero (gray), 2=RGB
    le = dt.newbyteorder("<")
    tile_w = tile_h = tile
    tiles_across = (width + tile_w - 1) // tile_w
    tiles_down = (height + tile_h - 1) // tile_h
    num_tiles = tiles_across * tiles_down
    tile_byte_len = tile_w * tile_h * bands * (bits // 8)

    chunky = np.transpose(a, (1, 2, 0))  # (h, w, bands), band-interleaved-by-pixel
    blocks = []
    for ty in range(tiles_down):
        for tx in range(tiles_across):
            block = np.zeros((tile_h, tile_w, bands), dtype=le)
            y0, x0 = ty * tile_h, tx * tile_w
            vh = min(tile_h, height - y0)
            vw = min(tile_w, width - x0)
            block[:vh, :vw, :] = chunky[y0 : y0 + vh, x0 : x0 + vw, :].astype(le)
            blocks.append(block.tobytes())
    tile_data = b"".join(blocks)
    data_end = 8 + len(tile_data)
    tile_offsets = [8 + i * tile_byte_len for i in range(num_tiles)]

    # GeoKeyDirectory for EPSG:4326 (matches the in-repo GeoTIFF writer's keys).
    geokeys = [1, 1, 0, 4, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2054, 0, 1, 9102]
    sx, sy = abs(transform[0]), abs(transform[4])
    tx, ty = transform[2], transform[5]
    # (tag, tiff_type, values); tiff_type 3=SHORT, 4=LONG, 12=DOUBLE. Ascending tags.
    entries = [
        (256, 4, [width]),
        (257, 4, [height]),
        (258, 3, [bits] * bands),
        (259, 3, [1]),  # Compression = none
        (262, 3, [photometric]),
        (277, 3, [bands]),
        (284, 3, [1]),  # PlanarConfiguration = chunky
        (322, 4, [tile_w]),
        (323, 4, [tile_h]),
        (324, 4, tile_offsets),
        (325, 4, [tile_byte_len] * num_tiles),
        (339, 3, [sample_format] * bands),
        (33550, 12, [sx, sy, 0.0]),
        (33922, 12, [0.0, 0.0, 0.0, tx, ty, 0.0]),
        (34735, 3, geokeys),
    ]

    def _pack(tiff_type, values):
        fmt = {3: "H", 4: "I", 12: "d"}[tiff_type]
        return struct.pack("<" + fmt * len(values), *values)

    pool = bytearray()
    ifd_entries = []
    for tag, tiff_type, values in entries:
        payload = _pack(tiff_type, values)
        if len(payload) <= 4:  # value fits inline (little-endian, right-padded)
            value_field = payload + b"\x00" * (4 - len(payload))
        else:
            value_field = struct.pack("<I", data_end + len(pool))
            pool.extend(payload)
        ifd_entries.append((tag, tiff_type, len(values), value_field))

    ifd_offset = data_end + len(pool)
    ifd = bytearray(struct.pack("<H", len(ifd_entries)))
    for tag, tiff_type, count, value_field in ifd_entries:
        ifd.extend(struct.pack("<HHI", tag, tiff_type, count))
        ifd.extend(value_field)
    ifd.extend(struct.pack("<I", 0))  # no next IFD

    out = struct.pack("<2sHI", b"II", 42, ifd_offset) + tile_data + bytes(pool) + bytes(ifd)
    path.write_bytes(out)


def _tiled_uint16_fixture(path: Path, *, size: int = 1024, tile: int = 64) -> bytes:
    data = (np.arange(size * size, dtype=np.uint16).reshape(size, size) % 9000).astype(np.uint16)
    _write_tiled_tiff(path, data, transform=(1.0, 0.0, 0.0, 0.0, -1.0, float(size)), tile=tile)
    return path.read_bytes()


def _clear_cog_cache() -> None:
    """Remove the shared remote-COG byte cache so a fetch is a deterministic miss.

    read_cog caches full remote fetches under a per-URL key in the OS temp dir;
    ephemeral test ports can be reused across a session, so cache-code assertions
    must start from a known-empty cache.
    """
    import shutil
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "forge3d" / "cog_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)


def _codes(result) -> set[str]:
    return {warning["code"] for warning in result.get("warnings", [])}


def _warning_message(result, code: str) -> str:
    return next(w["message"] for w in result["warnings"] if w["code"] == code)
