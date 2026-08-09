"""Production behavior tests for ORBIS Task 6."""

from __future__ import annotations

import struct
import multiprocessing
import gc
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest


def _write_quadrant_cog(
    path: Path,
    *,
    georeferenced: bool = True,
    projected: bool = False,
    rotated_utm: bool = False,
    height_scale: float = 1.0,
) -> Path:
    """Classic TIFF with a 4x4 tiled image and a distinct 2x2 overview."""
    full_tiles = [value * height_scale for value in (10.0, 20.0, 30.0, 40.0)]
    full_data = b"".join(struct.pack("<4f", *([value] * 4)) for value in full_tiles)
    overview_data = struct.pack(
        "<4f", *(value * height_scale for value in (100.0, 200.0, 300.0, 400.0))
    )
    data = full_data + overview_data
    full_tile_offsets = [8 + 16 * index for index in range(4)]
    overview_tile_offset = 8 + len(full_data)

    external = bytearray()
    external_base = 8 + len(data)

    def ext(fmt: str, values: list[float] | list[int]) -> int:
        offset = external_base + len(external)
        external.extend(struct.pack(f"<{len(values)}{fmt}", *values))
        return offset

    full_offsets_off = ext("I", full_tile_offsets)
    full_counts_off = ext("I", [16] * 4)
    # This regional dataset is exactly global equirectangular tile (lod=1,
    # x=0,y=0): lon [-180,0], lat [0,90]. Its four TIFF tiles align with the
    # four lod=2 children, which makes wrong dataset-normalized addressing
    # immediately visible.
    transform_off = None
    if rotated_utm:
        # A rotated EPSG:32633 affine. UTM-to-WGS84 makes the projected
        # parallelogram edges curve; extrema therefore need dense edge points.
        transform_off = ext(
            "d",
            [
                362_390.78798225545, -79_800.2352165645, 0.0, 130_441.1032495692,
                69_887.79778504319, -247_238.102358239, 0.0, 6_127_056.708830688,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        geokeys_off = ext(
            "H", [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32633]
        )
    elif projected:
        # EPSG:3857 footprint of global tile lod=2,x=2,y=1:
        # lon [0,90], lat [0,45]. The non-linear latitude transform ensures
        # production code really maps through the CRS rather than treating
        # projected model coordinates as degrees.
        mercator_x_90 = 10018754.171394622
        mercator_y_45 = 5621521.486192066
        scale_off = ext("d", [mercator_x_90 / 4.0, mercator_y_45 / 4.0, 0.0])
        tiepoint_off = ext("d", [0.0, 0.0, 0.0, 0.0, mercator_y_45, 0.0])
        geokeys_off = ext(
            "H", [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857]
        )
    else:
        scale_off = ext("d", [45.0, 22.5, 0.0])
        tiepoint_off = ext("d", [0.0, 0.0, 0.0, -180.0, 90.0, 0.0])
        geokeys_off = ext(
            "H", [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]
        )

    common_full = [
        (256, 3, 1, 4), (257, 3, 1, 4), (258, 3, 1, 32),
        (259, 3, 1, 1), (262, 3, 1, 1), (277, 3, 1, 1),
        (322, 3, 1, 2), (323, 3, 1, 2),
        (324, 4, 4, full_offsets_off), (325, 4, 4, full_counts_off),
        (339, 3, 1, 3),
    ]
    if georeferenced:
        if transform_off is not None:
            common_full += [(34264, 12, 16, transform_off)]
        else:
            common_full += [
                (33550, 12, 3, scale_off),
                (33922, 12, 6, tiepoint_off),
            ]
        common_full += [(34735, 3, 16, geokeys_off)]
    overview_entries = [
        (256, 3, 1, 2), (257, 3, 1, 2), (258, 3, 1, 32),
        (259, 3, 1, 1), (262, 3, 1, 1), (277, 3, 1, 1),
        (322, 3, 1, 2), (323, 3, 1, 2),
        # A single LONG value is stored inline in a classic TIFF IFD entry.
        (324, 4, 1, overview_tile_offset), (325, 4, 1, 16),
        (339, 3, 1, 3),
    ]
    first_ifd_offset = external_base + len(external)
    first_ifd_size = 2 + 12 * len(common_full) + 4
    second_ifd_offset = first_ifd_offset + first_ifd_size

    def ifd(entries: list[tuple[int, int, int, int]], next_offset: int) -> bytes:
        encoded = struct.pack("<H", len(entries))
        encoded += b"".join(struct.pack("<HHII", *entry) for entry in entries)
        return encoded + struct.pack("<I", next_offset)

    path.write_bytes(
        struct.pack("<2sHI", b"II", 42, first_ifd_offset)
        + data
        + external
        + ifd(common_full, second_ifd_offset)
        + ifd(overview_entries, 0)
    )
    return path


def _native():
    return pytest.importorskip("forge3d._forge3d")


def _run_throttled_range_server(payload: bytes, ready, stop) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            start_text, end_text = self.headers["Range"].removeprefix("bytes=").split("-")
            start = int(start_text)
            end = min(int(end_text), len(payload) - 1)
            time.sleep(0.01)
            body = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.1
    ready.send(server.server_port)
    ready.close()
    while not stop.is_set():
        server.handle_request()
    server.server_close()


@contextmanager
def _throttled_range_server(payload: bytes):
    # The PyO3 constructor holds Python's GIL while it blocks. Serve HTTP in a
    # distinct process so the test measures the Rust executor, not Python
    # thread scheduling under the GIL.
    context = multiprocessing.get_context("spawn")
    parent_ready, child_ready = context.Pipe(duplex=False)
    stop = context.Event()
    process = context.Process(
        target=_run_throttled_range_server, args=(payload, child_ready, stop), daemon=True
    )
    process.start()
    port = parent_ready.recv()
    try:
        yield f"http://127.0.0.1:{port}/terrain.tif"
    finally:
        stop.set()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


def test_geographic_root_leaf_and_overview_selection(tmp_path: Path) -> None:
    native = _native()
    source = _write_quadrant_cog(tmp_path / "quadrants.tif")
    dataset = native.CogDataset(source.as_uri(), cache_size_mb=1)
    assert dataset.bounds == pytest.approx((-180.0, 0.0, 0.0, 90.0))

    # The whole-earth tile is only partially covered and must not stretch this
    # regional source over the planet.
    whole_earth = np.asarray(dataset.read_height_tile(0, 0, 0, 9, 9))
    whole_earth_coverage = np.asarray(dataset.read_height_tile_coverage(0, 0, 0, 9, 9))
    assert tuple(whole_earth[[0, 0, 4, 4], [0, 4, 0, 4]]) == (10, 20, 30, 40)
    assert np.all(whole_earth[[0, 4, 8, 8], [8, 8, 0, 8]] == 0)
    assert tuple(whole_earth_coverage[[0, 4, 0, 8], [0, 4, 8, 8]]) == (255, 255, 0, 0)

    root_full = np.asarray(dataset.read_height_tile(1, 0, 0, 4, 4))
    assert tuple(root_full[[0, 0, 3, 3], [0, 3, 0, 3]]) == (10, 20, 30, 40)
    for x, y, expected in [(0, 0, 10), (1, 0, 20), (0, 1, 30), (1, 1, 40)]:
        leaf = np.asarray(dataset.read_height_tile(2, x, y, 2, 2))
        assert np.all(leaf == expected)
        assert np.all(dataset.read_height_tile_coverage(2, x, y, 2, 2) == 255)

    # A 2x2 root needs only the 2x2 IFD; its deliberately distinct values prove
    # that overview selection changed with requested sample density.
    root_overview = np.asarray(dataset.read_height_tile(1, 0, 0, 2, 2))
    assert tuple(root_overview.ravel()) == (100, 200, 300, 400)
    stats = dataset.stats()
    assert stats["cache_hits"] >= 4
    assert stats["memory_used_bytes"] <= stats["memory_budget_bytes"]


def test_projected_cog_maps_global_tile_through_source_crs(tmp_path: Path) -> None:
    native = _native()
    source = _write_quadrant_cog(tmp_path / "web-mercator.tif", projected=True)
    dataset = native.CogDataset(source.as_uri(), cache_size_mb=1)
    assert dataset.bounds == pytest.approx((0.0, 0.0, 90.0, 45.0), abs=1e-6)

    covered = np.asarray(dataset.read_height_tile(2, 2, 1, 4, 4))
    assert tuple(covered[[0, 0, 3, 3], [0, 3, 0, 3]]) == (10, 20, 30, 40)
    outside = np.asarray(dataset.read_height_tile(2, 0, 0, 4, 4))
    assert np.all(outside == 0)
    assert np.all(dataset.read_height_tile_coverage(2, 0, 0, 4, 4) == 0)


def test_rotated_utm_bounds_retain_densified_edge_extrema(tmp_path: Path) -> None:
    native = _native()
    from forge3d import gis

    source = _write_quadrant_cog(tmp_path / "rotated-utm.tif", rotated_utm=True)
    dataset = native.CogDataset(source.as_uri(), cache_size_mb=1)
    inverse = gis.create_crs_transformer(32633, 4326)

    def projected(pixel_x: float, pixel_y: float) -> tuple[float, float]:
        return (
            130_441.1032495692 + 362_390.78798225545 * pixel_x - 79_800.2352165645 * pixel_y,
            6_127_056.708830688 + 69_887.79778504319 * pixel_x - 247_238.102358239 * pixel_y,
        )

    dense = []
    corners = []
    for index in range(33):
        t = 4.0 * index / 32.0
        for pixel in ((t, 0.0), (t, 4.0), (0.0, t), (4.0, t)):
            dense.append(inverse.transform_point(*projected(*pixel)))
    for pixel in ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)):
        corners.append(inverse.transform_point(*projected(*pixel)))
    dense_bounds = (
        min(point[0] for point in dense), min(point[1] for point in dense),
        max(point[0] for point in dense), max(point[1] for point in dense),
    )
    corner_bounds = (
        min(point[0] for point in corners), min(point[1] for point in corners),
        max(point[0] for point in corners), max(point[1] for point in corners),
    )
    assert max(abs(a - b) for a, b in zip(dense_bounds, corner_bounds)) > 0.01
    assert dataset.bounds == pytest.approx(dense_bounds, abs=1e-10)


def test_throttled_remote_cog_runtime_keeps_io_driven(tmp_path: Path) -> None:
    native = _native()
    source = _write_quadrant_cog(tmp_path / "remote.tif")
    with _throttled_range_server(source.read_bytes()) as url:
        dataset = native.CogDataset(url, cache_size_mb=1)
        covered = np.asarray(dataset.read_height_tile(1, 0, 0, 4, 4))
        renderer = native.TerrainRenderer(native.Session(window=False))
        renderer.enable_height_streaming_cog(
            dataset,
            terrain_extent_m=10_000.0,
            ring_count=1,
            ring_resolution=8,
            lod=1,
            tile_resolution=8,
            max_in_flight=2,
            pool_size=1,
            coarse_prefill=False,
            max_resident_bytes=1024 * 1024,
        )
        del dataset
        gc.collect()
        stats = renderer.height_streaming_stats()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stats = renderer.stream_height_tiles_globe(
                (6_372_000.0, 0.0, 0.0), max_uploads=2
            )
            if stats["tiles_uploaded"]:
                break
            time.sleep(0.005)
        renderer.disable_height_streaming()
    assert tuple(covered[[0, 0, 3, 3], [0, 3, 0, 3]]) == (10, 20, 30, 40)
    assert stats["tiles_uploaded"] >= 1, stats


def test_missing_geotiff_crs_is_an_explicit_open_error(tmp_path: Path) -> None:
    native = _native()
    source = _write_quadrant_cog(tmp_path / "no-crs.tif", georeferenced=False)
    with pytest.raises(Exception, match="GeoTIFF|CRS|georeference"):
        native.CogDataset(source.as_uri(), cache_size_mb=1)
