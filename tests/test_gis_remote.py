"""G-002 Later remote/cache helper tests."""

from __future__ import annotations

import hashlib
import http.server
import ast
import json
import socketserver
import threading
from pathlib import Path

import pytest
from _loopback import bind_loopback_or_skip

import forge3d.gis as gis
from forge3d._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="GIS remote tests require the compiled _forge3d extension",
)


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b""
    content_type = "application/geo+json"
    status = 200

    def do_GET(self):  # noqa: N802
        self.send_response(self.status)
        self.send_header("content-type", self.content_type)
        self.send_header("etag", '"test-etag"')
        self.send_header("content-length", str(len(self.body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *_args):
        return


class _Server(socketserver.TCPServer):
    allow_reuse_address = True


def _serve(body: bytes, *, content_type: str = "application/geo+json", status: int = 200):
    _Handler.body = body
    _Handler.content_type = content_type
    _Handler.status = status
    server = bind_loopback_or_skip(_Server, _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/data.geojson"


def _codes(result):
    return {warning["code"] for warning in result.get("warnings", [])}


def _backend_unavailable(exc: BaseException) -> bool:
    text = str(exc)
    return "backend_unavailable" in text and "gis-remote" in text


def test_python_gis_module_has_no_runtime_gis_backend_imports():
    source = Path(gis.__file__).read_text(encoding="utf-8")
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    )

    for banned in ("rasterio", "geopandas", "shapely", "rioxarray", "xarray", "terra"):
        assert banned not in imports


def test_cache_geodata_plain_key_hit(tmp_path: Path):
    cached = tmp_path / "roads.geojson"
    cached.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    result = gis.cache_geodata("roads.geojson", tmp_path)

    assert result["status"] == "hit"
    assert result["from_cache"] is True
    assert Path(result["cache_path"]) == cached
    assert result["byte_size"] == cached.stat().st_size
    assert result["checksum"].startswith("sha256:")


def test_cache_geodata_plain_key_miss(tmp_path: Path):
    with pytest.raises(RuntimeError, match="cache_miss"):
        gis.cache_geodata("missing.geojson", tmp_path)


def test_fetch_remote_geodata_rejects_non_http_scheme(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported_scheme"):
        gis.fetch_remote_geodata(f"file:///{tmp_path / 'x.geojson'}")


def test_fetch_remote_geodata_local_http_cache_and_checksum(tmp_path: Path):
    body = json.dumps({"type": "FeatureCollection", "features": []}).encode("utf-8")
    url = "http://example.test/data.geojson"
    checksum = "sha256:" + hashlib.sha256(body).hexdigest()

    first = gis.fetch_remote_geodata(
        url,
        cache={"cache_dir": tmp_path, "mock_body": body, "content_type": "application/geo+json"},
        checksum=checksum,
    )
    second = gis.fetch_remote_geodata(url, cache={"cache_dir": tmp_path})

    assert first["status"] == "fetched"
    assert first["from_cache"] is False
    assert first["content_type"] == "application/geo+json"
    assert Path(first["cache_path"]).is_file()
    assert Path(first["cache_path"]).read_bytes() == body
    assert second["status"] == "hit"
    assert second["from_cache"] is True
    assert second["cache_path"] == first["cache_path"]


def test_fetch_remote_geodata_validates_checksum_on_cache_hit(tmp_path: Path):
    body = json.dumps({"type": "FeatureCollection", "features": []}).encode("utf-8")
    url = "http://example.test/data.geojson"

    gis.fetch_remote_geodata(
        url,
        cache={"cache_dir": tmp_path, "mock_body": body, "content_type": "application/geo+json"},
    )

    with pytest.raises(RuntimeError, match="checksum_mismatch"):
        gis.fetch_remote_geodata(url, cache={"cache_dir": tmp_path}, checksum="0" * 64)


def test_cache_geodata_url_refresh_replaces_cached_file(tmp_path: Path):
    first_body = b'{"type":"FeatureCollection","features":[]}'
    second_body = b'{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":null}]}'
    server, url = _serve(first_body)
    try:
        try:
            first = gis.cache_geodata(url, tmp_path)
        except RuntimeError as exc:
            if _backend_unavailable(exc):
                pytest.skip("gis-remote feature is unavailable in this extension build")
            raise
        cache_path = Path(first["cache_path"])
        assert cache_path.read_bytes() == first_body

        _Handler.body = second_body
        cached = gis.cache_geodata(url, tmp_path)
        cached_bytes = Path(cached["cache_path"]).read_bytes()
        refreshed = gis.cache_geodata(url, tmp_path, refresh=True)

        assert cached["status"] == "hit"
        assert cached_bytes == first_body
        assert Path(refreshed["cache_path"]).read_bytes() == second_body
        assert refreshed["status"] == "fetched"
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_remote_geodata_checksum_mismatch_removes_temp(tmp_path: Path):
    body = b'{"type":"FeatureCollection","features":[]}'
    url = "http://example.test/data.geojson"

    with pytest.raises(RuntimeError, match="checksum_mismatch"):
        gis.fetch_remote_geodata(
            url,
            cache={"cache_dir": tmp_path, "mock_body": body, "content_type": "application/geo+json"},
            checksum="sha256:" + "0" * 64,
        )

    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.geojson"))


def test_fetch_remote_geodata_rejects_unknown_content_type(tmp_path: Path):
    url = "http://example.test/data.bin"

    with pytest.raises(ValueError, match="unsupported_driver"):
        gis.fetch_remote_geodata(
            url,
            cache={"cache_dir": tmp_path, "mock_body": b"hello", "content_type": "text/plain"},
        )


def test_fetch_vector_geojson_uses_remote_cache_and_reader(tmp_path: Path):
    feature = {
        "type": "Feature",
        "properties": {"name": "road"},
        "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
    }
    body = json.dumps(
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [feature],
        }
    ).encode("utf-8")
    url = "http://example.test/data.geojson"

    result = gis.fetch_vector(
        url,
        cache={"cache_dir": tmp_path, "mock_body": body, "content_type": "application/geo+json"},
    )

    assert result["type"] == "FeatureCollection"
    assert result["features"][0]["properties"]["name"] == "road"
    assert result["remote"]["status"] == "fetched"
    assert result["info"]["feature_count"] == 1
    assert _codes(result) == set()
