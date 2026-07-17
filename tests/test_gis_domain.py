"""G-002 Later domain helper tests."""

from __future__ import annotations

import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from _loopback import bind_loopback_or_skip

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS domain tests require the compiled _forge3d extension",
)


def _codes(result) -> set[str]:
    return {warning["code"] for warning in result.get("warnings", [])}


def _memory_info(array, *, transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0), crs=True):
    arr = np.asarray(array)
    bands, height, width = (1, arr.shape[0], arr.shape[1]) if arr.ndim == 2 else arr.shape
    return {
        "path": "",
        "driver": "memory",
        "width": width,
        "height": height,
        "band_count": bands,
        "dtype_per_band": [arr.dtype.name] * bands,
        "crs_wkt": None,
        "crs_authority": {"name": "EPSG", "code": "4326"} if crs else None,
        "transform": transform,
        "bounds": (0.0, 0.0, float(width), float(height)) if transform else None,
        "resolution": (abs(transform[0]), abs(transform[4])) if transform else None,
        "nodata_per_band": [None] * bands,
        "warnings": [],
    }


def _fc(features):
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }


def _feature(properties, geometry):
    return {"type": "Feature", "properties": properties, "geometry": geometry}


def _write_png(path: Path, rgb: np.ndarray):
    height, width, channels = rgb.shape
    assert channels == 3
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_load_context_vectors_layers_and_missing_layer():
    roads = _fc([_feature({"kind": "road"}, {"type": "LineString", "coordinates": [[0, 0], [1, 1]]})])
    water = _fc([_feature({"kind": "water"}, {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]})])

    result = gis.load_context_vectors({"roads": roads, "water": water}, layers=["roads"])

    assert set(result["layers"]) == {"roads"}
    assert result["layers"]["roads"]["feature_count"] == 1
    assert result["operation"]["output_count"] == 1
    with pytest.raises(ValueError, match="missing_layer"):
        gis.load_context_vectors({"roads": roads}, layers=["buildings"])


def test_load_building_footprints_geojson_cityjson_and_height_extraction(tmp_path: Path):
    building = _feature(
        {"height": "12 ft", "building:levels": "3"},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
    )
    geojson_result = gis.load_building_footprints(_fc([building]))
    heights = gis.extract_building_heights(geojson_result)

    assert geojson_result["feature_count"] == 1
    assert heights["heights_m"][0] == pytest.approx(3.6576)
    assert heights["attributes"][0] == "height"

    projected = gis.load_building_footprints(_fc([building]), dst_crs="EPSG:3857")
    assert projected["crs"]["authority"] == {"name": "EPSG", "code": "3857"}
    assert projected["bounds"][2] > 100_000

    cityjson = {
        "type": "CityJSON",
        "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]],
        "CityObjects": {
            "b1": {
                "type": "Building",
                "attributes": {"measuredHeight": 8},
                "geometry": [{"type": "MultiSurface", "boundaries": [[[[0, 1, 2, 3]]]]}],
            }
        },
    }
    cityjson_result = gis.load_building_footprints(cityjson)
    assert cityjson_result["feature_count"] == 1
    assert cityjson_result["features"][0]["properties"]["measuredHeight"] == 8
    with pytest.raises(ValueError, match="missing_crs"):
        gis.load_building_footprints(cityjson, dst_crs="EPSG:3857")

    gpkg = tmp_path / "buildings.gpkg"
    gpkg.write_bytes(b"not really gpkg")
    with pytest.raises(RuntimeError, match="backend_unavailable.*gdal-vector"):
        gis.load_building_footprints(gpkg)


def test_prepare_dem_decode_terrarium_and_derivatives(tmp_path: Path):
    dem = np.array([[1.0, -9999.0], [3.0, 5.0]], dtype=np.float32)
    result = gis.prepare_dem({"array": dem, "info": _memory_info(dem)}, nodata=-9999.0)

    assert result["array"].dtype == np.float32
    assert result["valid_count"] == 3
    assert result["mask_polarity"] == "true_valid"
    assert result["operation"]["name"] == "prepare_dem"
    assert result["array"][0, 0, 1] == pytest.approx(-9999.0)

    rgb = np.array([[[128, 0, 0], [128, 1, 0]]], dtype=np.uint8)
    decoded = gis.decode_terrarium_dem(rgb)
    np.testing.assert_allclose(decoded["array"], np.array([[[0.0, 1.0]]], dtype=np.float32))

    png = tmp_path / "tile.png"
    _write_png(png, rgb)
    decoded_path = gis.decode_terrarium_dem(png)
    np.testing.assert_allclose(decoded_path["array"], decoded["array"])

    plane = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    derivatives = gis.prepare_terrain_derivatives({"array": plane, "info": _memory_info(plane)})
    assert set(derivatives["derivatives"]) == {"slope", "hillshade"}
    assert derivatives["derivatives"]["slope"]["units"] == "degrees"
    assert derivatives["derivatives"]["slope"]["array"].shape == (1, 3, 3)
    assert derivatives["derivatives"]["hillshade"]["array"].dtype == np.uint8


def test_build_terrarium_dem_from_explicit_cached_tile(tmp_path: Path):
    tile_dir = tmp_path / "0" / "0"
    tile_dir.mkdir(parents=True)
    _write_png(tile_dir / "0.png", np.array([[[128, 0, 0]]], dtype=np.uint8))

    result = gis.build_terrarium_dem((-10.0, -10.0, 10.0, 10.0), 0, cache={"cache_dir": tmp_path})

    np.testing.assert_allclose(result["array"], np.array([[[0.0]]], dtype=np.float32))
    assert result["tile_count"] == 1
    assert result["manifest"][0]["status"] == "hit"

    with pytest.raises(RuntimeError, match="cache_miss|missing_tile"):
        gis.build_terrarium_dem((-10.0, -10.0, 10.0, 10.0), 0, cache={"cache_dir": tmp_path / "empty"})


def test_build_terrarium_dem_fetches_explicit_url_template_and_caches(tmp_path: Path):
    tile = tmp_path / "source.png"
    _write_png(tile, np.array([[[128, 0, 0], [128, 1, 0]]], dtype=np.uint8))
    body = tile.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        requests = 0

        def do_GET(self):
            type(self).requests += 1
            assert self.path == "/0/0/0.png"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = bind_loopback_or_skip(ThreadingHTTPServer, Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cache_dir = tmp_path / "cache"
    template = f"http://127.0.0.1:{server.server_port}/{{z}}/{{x}}/{{y}}.png"
    try:
        fetched = gis.build_terrarium_dem(
            (-10.0, -10.0, 10.0, 10.0),
            0,
            cache={"cache_dir": str(cache_dir)},
            url_template=template,
            timeout=2.0,
        )
        cached = gis.build_terrarium_dem(
            (-10.0, -10.0, 10.0, 10.0),
            0,
            cache={"cache_dir": str(cache_dir)},
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    np.testing.assert_array_equal(fetched["array"], cached["array"])
    assert fetched["manifest"][0]["status"] == "fetched"
    assert cached["manifest"][0]["status"] == "hit"
    assert Handler.requests == 1


def test_read_gridded_dataset_and_subset_grid(tmp_path: Path):
    path = tmp_path / "grid.tif"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    gis.write_raster(path, data, crs="EPSG:4326", transform=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0))

    grid = gis.read_gridded_dataset(path)
    subset = gis.subset_grid(path, (1.0, 1.0, 3.0, 3.0))

    assert grid["variables"] == ["band_1"]
    assert grid["dimensions"]["x"] == 4
    assert grid["dimensions"]["y"] == 4
    np.testing.assert_array_equal(subset["array"], data[1:3, 1:3].reshape(1, 2, 2))
    assert subset["bounds"] == pytest.approx((1.0, 1.0, 3.0, 3.0))
    with pytest.raises(RuntimeError, match="backend_unavailable.*netcdf"):
        gis.read_gridded_dataset(tmp_path / "sample.nc")


def test_landcover_population_osm_scene_and_utm():
    landcover = np.array([[1, 2], [3, 99]], dtype=np.uint8)
    target = _memory_info(landcover)
    land = gis.prepare_landcover_raster(
        {"array": landcover, "info": target},
        target,
        classes={1: "tree", 2: "water", 3: "urban"},
    )
    assert land["class_counts"][1] == 1
    assert land["class_table"][0]["label"] == "tree"
    assert "unknown_class" in _codes(land)

    population = gis.prepare_population_raster(
        {"array": np.array([[0.0, 10.0]], dtype=np.float32), "info": _memory_info(np.zeros((1, 2), dtype=np.float32))},
        normalization="minmax",
    )
    np.testing.assert_allclose(population["array"], np.array([[[0.0, 1.0]]], dtype=np.float32))
    with pytest.raises(ValueError, match="invalid_argument"):
        gis.prepare_population_raster({"array": np.array([[-1.0]], dtype=np.float32), "info": _memory_info(np.zeros((1, 1), dtype=np.float32))})

    osm_json = {
        "elements": [
            {"type": "node", "id": 1, "lat": 0.0, "lon": 0.0},
            {"type": "node", "id": 2, "lat": 0.0, "lon": 1.0},
            {"type": "node", "id": 3, "lat": 1.0, "lon": 1.0},
            {"type": "node", "id": 4, "lat": 1.0, "lon": 0.0},
            {"type": "way", "id": 5, "nodes": [1, 2], "tags": {"highway": "residential"}},
            {"type": "way", "id": 6, "nodes": [1, 2, 3, 4, 1], "tags": {"building": "yes", "building:levels": "2"}},
            {"type": "way", "id": 7, "nodes": [1, 2, 3, 4, 1], "tags": {"natural": "water"}},
        ]
    }
    scene = gis.prepare_osm_scene((0.0, 0.0, 1.0, 1.0), cache={"osm_json": osm_json})
    assert scene["layers"]["roads"]["feature_count"] == 1
    assert scene["layers"]["buildings"]["feature_count"] == 1
    assert scene["layers"]["water"]["feature_count"] == 1
    assert scene["building_heights"]["heights_m"][0] == pytest.approx(6.0)

    north = gis.estimate_local_utm((-123.0, 45.0, -122.0, 46.0))
    south = gis.estimate_local_utm((30.0, -20.0, 31.0, -19.0))
    anti = gis.estimate_local_utm((170.0, -5.0, -170.0, 5.0))
    assert north["epsg"] == 32610
    assert south["epsg"] == 32736
    assert anti["confidence"] == "low"
    assert anti["antimeridian"] is True
    with pytest.raises(ValueError, match="invalid_bounds"):
        gis.estimate_local_utm((0.0, 85.0, 1.0, 86.0))
