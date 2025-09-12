# python/forge3d/tiles/client.py
# XYZ/WMTS tile client with simple on-disk cache and mosaic composition.
# Provides offline mode, basic ETag/Last-Modified handling, and attribution metadata surfacing.
# RELEVANT FILES:python/forge3d/tiles/__init__.py,python/forge3d/tiles/overlay.py,tests/test_tiles_client.py

from __future__ import annotations

import io
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _user_cache_dir() -> Path:
    """Return a cross-platform user cache root path.

    Windows uses %LOCALAPPDATA%.

    macOS uses ~/Library/Caches.

    Linux/Unix uses ~/.cache.
    """
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root)
        return Path.home() / "AppData" / "Local"
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


def sys_platform() -> str:
    # Small helper to avoid importing platform for a single check.
    return os.uname().sysname.lower() if hasattr(os, "uname") else os.name


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert WGS84 lat/lon to slippy-map tile indices (x, y) at zoom.

    This uses the standard Web Mercator (EPSG:3857) tiling.

    Returns floor indices for the tile containing the coordinate.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    xtile = max(0, min(int(n) - 1, xtile))
    ytile = max(0, min(int(n) - 1, ytile))
    return xtile, ytile


def bbox_to_tiles(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    zoom: int,
) -> Tuple[range, range]:
    """Compute inclusive tile x and y ranges covering a lon/lat bbox at zoom.

    Returns (x_range, y_range), each suitable for iteration.
    """
    # Normalize bbox ordering
    lmin, lmax = sorted([lon_min, lon_max])
    bmin, bmax = sorted([lat_min, lat_max])
    x0, y1 = deg2num(bmin, lmin, zoom)
    x1, y0 = deg2num(bmax, lmax, zoom)
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    return range(xmin, xmax + 1), range(ymin, ymax + 1)


@dataclass
class TileProvider:
    name: str
    url_template: str  # e.g. https://{s}.tile.server/{z}/{x}/{y}.png
    subdomains: Optional[List[str]] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    tile_size: int = 256

    @property
    def id(self) -> str:
        # Sanitize a compact identifier for cache path
        return "".join(ch for ch in self.name.lower() if ch.isalnum() or ch in ("-", "_")) or "provider"

    def attribution(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "url": self.homepage or "",
            "license": self.license or "",
        }

    def format_url(self, z: int, x: int, y: int) -> str:
        sub = ""
        if self.subdomains:
            sub = random.choice(self.subdomains)
        return (
            self.url_template
            .replace("{z}", str(z))
            .replace("{x}", str(x))
            .replace("{y}", str(y))
            .replace("{s}", sub)
        )


class TileClient:
    def __init__(
        self,
        cache_root: Optional[Path] = None,
        cache_namespace: str = "forge3d/tiles",
        request_timeout_s: float = 10.0,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cache_root = Path(cache_root) if cache_root else (_user_cache_dir() / cache_namespace)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.request_timeout_s = request_timeout_s
        # Base headers with User-Agent/Referer per provider policies (e.g., OSM).
        env_ua = os.environ.get("FORGE3D_TILE_USER_AGENT", "").strip()
        ua = (default_headers or {}).get("User-Agent") if default_headers else None
        if not ua:
            ua = env_ua or "forge3d-tiles/0.12 (+https://forge3d.readthedocs.io; contact: set FORGE3D_TILE_USER_AGENT)"
        ref = (default_headers or {}).get("Referer") if default_headers else None
        if not ref:
            ref = os.environ.get("FORGE3D_TILE_REFERER", "https://forge3d.readthedocs.io/")
        self.base_headers: Dict[str, str] = {"User-Agent": ua, "Referer": ref}
        if default_headers:
            self.base_headers.update({k: v for k, v in default_headers.items() if v})

    # --- HTTP layer (overridable for tests) ---
    def _http_get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, str], bytes]:
        import urllib.request  # Delayed import to keep dependency surface small

        eff_headers = dict(self.base_headers)
        if headers:
            eff_headers.update(headers)
        req = urllib.request.Request(url, headers=eff_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                status = getattr(resp, "status", 200)
                data = resp.read()
                # Convert headers to plain dict[str, str]
                hdrs = {k: v for k, v in resp.headers.items()}
                return status, hdrs, data
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"HTTP GET failed: {e}")

    # --- Cache paths & metadata ---
    def _tile_paths(self, provider: TileProvider, z: int, x: int, y: int) -> Tuple[Path, Path]:
        base = self.cache_root / provider.id / str(z) / str(x)
        base.mkdir(parents=True, exist_ok=True)
        png = base / f"{y}.png"
        meta = base / f"{y}.json"
        return png, meta

    def _load_cache(self, png: Path, meta: Path) -> Optional[Tuple[bytes, Dict[str, str]]]:
        if not png.exists():
            return None
        try:
            content = png.read_bytes()
            headers = json.loads(meta.read_text()) if meta.exists() else {}
            return content, headers
        except Exception:
            return None

    def _save_cache(self, png: Path, meta: Path, content: bytes, headers: Dict[str, str]) -> None:
        png.write_bytes(content)
        safe_headers = {k: headers.get(k, "") for k in ("ETag", "Last-Modified", "Date")}
        meta.write_text(json.dumps(safe_headers))

    # --- Public API ---
    def get_tile(
        self,
        provider: TileProvider,
        z: int,
        x: int,
        y: int,
        *,
        online: bool = True,
        max_retries: int = 2,
        backoff_s: float = 0.2,
    ) -> bytes:
        """Fetch a single tile, honoring on-disk cache and conditional requests.

        If online is False, only cache is used and a FileNotFoundError is raised if missing.
        """
        png, meta = self._tile_paths(provider, z, x, y)
        cached = self._load_cache(png, meta)

        if not online:
            if cached is None:
                raise FileNotFoundError(f"Tile not in cache for offline mode: z={z} x={x} y={y}")
            return cached[0]

        headers: Dict[str, str] = {}
        if cached and cached[1].get("ETag"):
            headers["If-None-Match"] = cached[1]["ETag"]
        if cached and cached[1].get("Last-Modified"):
            headers["If-Modified-Since"] = cached[1]["Last-Modified"]

        url = provider.format_url(z, x, y)
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt <= max_retries:
            try:
                req_headers = dict(self.base_headers)
                req_headers.update(headers)
                status, resp_headers, content = self._http_get(url, headers=req_headers)
                if status == 304 and cached is not None:
                    return cached[0]
                if status == 200 and content:
                    self._save_cache(png, meta, content, resp_headers)
                    return content
                # Unexpected status: fall back to cache if present
                if cached is not None:
                    return cached[0]
                raise RuntimeError(f"Unexpected HTTP status: {status}")
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt == max_retries:
                    break
                time.sleep(backoff_s)
                attempt += 1

        # Final fallback
        if cached is not None:
            return cached[0]
        raise RuntimeError(f"Failed to get tile and no cache available: {last_error}")

    def compose_mosaic(
        self,
        provider: TileProvider,
        bbox: Tuple[float, float, float, float],
        zoom: int,
        *,
        online: bool = True,
    ) -> "Image.Image":
        """Fetch and compose a tile mosaic for bbox=(lon_min, lat_min, lon_max, lat_max).

        Returns a Pillow RGBA image.
        """
        try:
            from PIL import Image  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("Pillow is required for mosaic composition") from e

        lon_min, lat_min, lon_max, lat_max = bbox
        xs, ys = bbox_to_tiles(lon_min, lat_min, lon_max, lat_max, zoom)
        tile_size = provider.tile_size
        width = len(xs) * tile_size
        height = len(ys) * tile_size
        mosaic = Image.new("RGBA", (width, height))

        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                content = self.get_tile(provider, zoom, x, y, online=online)
                im = Image.open(io.BytesIO(content)).convert("RGBA")
                mosaic.paste(im, (i * tile_size, j * tile_size))

        return mosaic
