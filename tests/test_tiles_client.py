# tests/test_tiles_client.py
# Tests XYZ client tile math, caching, conditional requests, and mosaic composition.
# Ensures offline mode serves from cache and provider attribution is exposed.
# RELEVANT FILES:python/forge3d/tiles/client.py,python/forge3d/tiles/__init__.py,python/forge3d/tiles/overlay.py

import io
import json
from pathlib import Path
from typing import Dict, Tuple

import pytest


def make_tile_png(color: Tuple[int, int, int, int] = (255, 0, 0, 255), size: int = 256) -> bytes:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Pillow not installed: {e}")
    im = Image.new("RGBA", (size, size), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_bbox_to_tiles_math():
    from forge3d.tiles import bbox_to_tiles

    xs, ys = bbox_to_tiles(-0.1, 51.5, 0.1, 51.6, 12)
    assert len(xs) >= 1 and len(ys) >= 1


def test_mosaic_fetch_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from forge3d.tiles import TileClient, TileProvider

    provider = TileProvider(
        name="osm",
        url_template="https://example/{z}/{x}/{y}.png",
        license="ODbL",
        homepage="https://www.openstreetmap.org",
        tile_size=128,
    )

    # Setup client with temp cache
    client = TileClient(cache_root=tmp_path)

    # Fake HTTP layer to generate colored tiles per (x,y)
    calls: Dict[str, int] = {"count": 0}

    def fake_http_get(url: str, headers=None):  # noqa: ANN001
        # Must include a polite User-Agent by default
        assert headers is not None and "User-Agent" in headers and len(headers["User-Agent"]) > 8
        calls["count"] += 1
        parts = url.strip("/").split("/")
        y = int(parts[-1].split(".")[0])
        x = int(parts[-2])
        z = int(parts[-3])
        assert z == 12
        # Encode tile position into color
        r = (x * 23) % 256
        g = (y * 47) % 256
        b = ((x + y) * 19) % 256
        content = make_tile_png((r, g, b, 255), size=provider.tile_size)
        return 200, {"ETag": f"W/\"{x}-{y}\""}, content

    monkeypatch.setattr(client, "_http_get", fake_http_get)

    bbox = (-0.1, 51.5, 0.1, 51.6)  # Small area
    img = client.compose_mosaic(provider, bbox, 12, online=True)
    assert img.mode == "RGBA"
    # Expect a grid of tiles
    assert img.size[0] % provider.tile_size == 0
    assert img.size[1] % provider.tile_size == 0
    assert calls["count"] > 0

    # Second run should use conditional requests; simulate 304 Not Modified
    def fake_http_304(url: str, headers=None):  # noqa: ANN001
        assert headers and ("If-None-Match" in headers or "If-Modified-Since" in headers)
        return 304, {}, b""

    monkeypatch.setattr(client, "_http_get", fake_http_304)
    img2 = client.compose_mosaic(provider, bbox, 12, online=True)
    assert img2.size == img.size

    # Offline should read from cache only and never call HTTP
    def fail_http(url: str, headers=None):  # noqa: ANN001, ARG001
        raise AssertionError("HTTP should not be called in offline mode")

    monkeypatch.setattr(client, "_http_get", fail_http)
    img3 = client.compose_mosaic(provider, bbox, 12, online=False)
    assert img3.size == img.size

    # Provider attribution present
    meta = provider.attribution()
    assert meta["name"] and isinstance(meta["name"], str)


def test_cache_metadata_roundtrip(tmp_path: Path):
    from forge3d.tiles import TileClient, TileProvider

    provider = TileProvider(name="t", url_template="https://example/{z}/{x}/{y}.png")
    client = TileClient(cache_root=tmp_path)

    png, meta = client._tile_paths(provider, 1, 2, 3)
    content = make_tile_png()
    headers = {"ETag": "abc", "Last-Modified": "yesterday"}
    client._save_cache(png, meta, content, headers)
    loaded = client._load_cache(png, meta)
    assert loaded is not None
    assert loaded[0] == content
    saved_headers = json.loads(meta.read_text())
    assert saved_headers.get("ETag") == "abc"
