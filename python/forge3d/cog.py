"""P3: Cloud Optimized GeoTIFF (COG) streaming API.

This module provides streaming access to terrain data from Cloud Optimized GeoTIFFs
without requiring pre-tiling. Data is fetched on-demand via HTTP range requests
and cached with an LRU eviction policy.

Example:
    >>> from forge3d.cog import CogDataset
    >>> ds = CogDataset("https://example.com/terrain.tif", cache_size_mb=256)
    >>> tile = ds.read_tile(x=5, y=3, lod=2)
    >>> print(f"Tile shape: {tile.shape}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import native COG support
_COG_AVAILABLE = False
_CogDatasetNative = None

try:
    from . import _forge3d
    if hasattr(_forge3d, 'CogDataset'):
        _CogDatasetNative = _forge3d.CogDataset
        _COG_AVAILABLE = True
except ImportError:
    pass


def is_cog_available() -> bool:
    """Check if COG streaming is available.
    
    Returns:
        True if the forge3d extension was built with cog_streaming feature.
    """
    return _COG_AVAILABLE


@dataclass
class CogStats:
    """COG streaming statistics.
    
    Attributes:
        cache_hits: Number of tile cache hits
        cache_misses: Number of tile cache misses
        cache_evictions: Number of tiles evicted from cache
        memory_used_bytes: Current cache memory usage in bytes
        memory_budget_bytes: Cache memory budget in bytes
        hit_rate_percent: Cache hit rate as percentage
    """
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    memory_used_bytes: int
    memory_budget_bytes: int
    hit_rate_percent: float
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CogStats":
        """Create from stats dictionary."""
        return cls(
            cache_hits=int(d.get("cache_hits", 0)),
            cache_misses=int(d.get("cache_misses", 0)),
            cache_evictions=int(d.get("cache_evictions", 0)),
            memory_used_bytes=int(d.get("memory_used_bytes", 0)),
            memory_budget_bytes=int(d.get("memory_budget_bytes", 0)),
            hit_rate_percent=d.get("hit_rate_percent", 0.0),
        )


@dataclass
class IfdInfo:
    """Information about a TIFF IFD (Image File Directory) / overview level.
    
    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        tile_width: Tile width in pixels
        tile_height: Tile height in pixels
        tiles_across: Number of tiles horizontally
        tiles_down: Number of tiles vertically
        bits_per_sample: Bits per sample (e.g., 32 for float32)
        compression: TIFF compression code
        tile_count: Total number of tiles
    """
    width: int
    height: int
    tile_width: int
    tile_height: int
    tiles_across: int
    tiles_down: int
    bits_per_sample: int
    compression: int
    tile_count: int
    
    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "IfdInfo":
        """Create from info dictionary."""
        return cls(
            width=d.get("width", 0),
            height=d.get("height", 0),
            tile_width=d.get("tile_width", 256),
            tile_height=d.get("tile_height", 256),
            tiles_across=d.get("tiles_across", 0),
            tiles_down=d.get("tiles_down", 0),
            bits_per_sample=d.get("bits_per_sample", 32),
            compression=d.get("compression", 1),
            tile_count=d.get("tile_count", 0),
        )


class CogDataset:
    """Stream tiles from a Cloud Optimized GeoTIFF.
    
    This class provides access to terrain elevation data stored in COG format.
    Data is fetched on-demand via HTTP range requests (for remote files) or
    direct file I/O (for local files) and cached with LRU eviction.
    
    Args:
        url: HTTP(S) URL or file:// path to COG file
        cache_size_mb: Tile cache memory budget in MB (default: 256)
    
    Example:
        >>> ds = CogDataset("https://example.com/dem.tif", cache_size_mb=128)
        >>> print(f"Bounds: {ds.bounds}")
        >>> print(f"Overview levels: {ds.overview_count}")
        >>> tile = ds.read_tile(0, 0, lod=0)
    
    Raises:
        RuntimeError: If COG streaming feature is not available
    """
    
    def __init__(self, url: str, *, cache_size_mb: int = 256):
        if not _COG_AVAILABLE:
            raise RuntimeError(
                "COG streaming is not available. "
                "Rebuild forge3d with: maturin develop --release --features cog_streaming"
            )
        
        self._native = _CogDatasetNative(url, cache_size_mb)
        self._url = url
        self._cache_size_mb = cache_size_mb
    
    @property
    def url(self) -> str:
        """URL of this COG dataset."""
        return self._url
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds (minx, miny, maxx, maxy).
        
        Note: These are pixel bounds unless the COG has geotransform metadata.
        """
        return self._native.bounds
    
    @property
    def overview_count(self) -> int:
        """Number of overview (pyramid) levels in this COG."""
        return self._native.overview_count
    
    def read_tile(
        self,
        x: int,
        y: int,
        lod: int = 0,
    ) -> "NDArray[np.float32]":
        """Read a single tile at specified coordinates and LOD.
        
        Args:
            x: Tile X coordinate (0-indexed from left)
            y: Tile Y coordinate (0-indexed from top)
            lod: Level of detail (0 = full resolution, higher = coarser)
        
        Returns:
            2D numpy array of float32 heights with shape (tile_height, tile_width)
        
        Raises:
            RuntimeError: If tile fetch fails (e.g., network error, invalid coords)
        """
        return self._native.read_tile(x, y, lod)
    
    def stats(self) -> CogStats:
        """Get cache statistics.
        
        Returns:
            CogStats with hit/miss counts, memory usage, etc.
        """
        raw = self._native.stats()
        return CogStats.from_dict(raw)
    
    def ifd_info(self, level: int = 0) -> IfdInfo:
        """Get information about a specific IFD/overview level.
        
        Args:
            level: Overview level (0 = full resolution)
        
        Returns:
            IfdInfo with dimensions, tile size, compression, etc.
        """
        raw = self._native.ifd_info(level)
        return IfdInfo.from_dict(raw)
    
    def __repr__(self) -> str:
        return f"CogDataset(url={self._url!r}, cache_size_mb={self._cache_size_mb})"


class CogDatasetFallback:
    """Fallback COG dataset using rasterio (when native COG streaming unavailable).
    
    This provides basic COG access via rasterio's built-in support for
    HTTP range requests and overviews, but without the optimized caching
    of the native implementation.
    
    Note: This fallback requires rasterio to be installed.
    """
    
    def __init__(self, url: str, *, cache_size_mb: int = 256):
        try:
            import rasterio
        except ImportError:
            raise RuntimeError(
                "Neither native COG streaming nor rasterio is available. "
                "Install rasterio or rebuild forge3d with cog_streaming feature."
            )
        
        self._url = url
        self._cache_size_mb = cache_size_mb
        self._src = rasterio.open(url)
        self._cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._cache_order: list = []
        self._hits = 0
        self._misses = 0
    
    @property
    def url(self) -> str:
        return self._url
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        b = self._src.bounds
        return (b.left, b.bottom, b.right, b.top)
    
    @property
    def overview_count(self) -> int:
        return len(self._src.overviews(1)) + 1
    
    def read_tile(
        self,
        x: int,
        y: int,
        lod: int = 0,
    ) -> "NDArray[np.float32]":
        import rasterio
        from rasterio.windows import Window
        
        cache_key = (x, y, lod)
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        
        self._misses += 1
        
        tile_size = 256
        scale = 2 ** lod
        
        window = Window(
            x * tile_size * scale,
            y * tile_size * scale,
            tile_size * scale,
            tile_size * scale,
        )
        
        data = self._src.read(
            1,
            window=window,
            out_shape=(tile_size, tile_size),
        ).astype(np.float32)
        
        max_cached = (self._cache_size_mb * 1024 * 1024) // (tile_size * tile_size * 4)
        if len(self._cache) >= max_cached and self._cache_order:
            old_key = self._cache_order.pop(0)
            self._cache.pop(old_key, None)
        
        self._cache[cache_key] = data
        self._cache_order.append(cache_key)
        
        return data
    
    def stats(self) -> CogStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        tile_size = 256
        memory_used = len(self._cache) * tile_size * tile_size * 4
        
        return CogStats(
            cache_hits=self._hits,
            cache_misses=self._misses,
            cache_evictions=0,
            memory_used_bytes=memory_used,
            memory_budget_bytes=self._cache_size_mb * 1024 * 1024,
            hit_rate_percent=hit_rate,
        )
    
    def ifd_info(self, level: int = 0) -> IfdInfo:
        scale = 2 ** level
        return IfdInfo(
            width=self._src.width // scale,
            height=self._src.height // scale,
            tile_width=256,
            tile_height=256,
            tiles_across=(self._src.width // scale + 255) // 256,
            tiles_down=(self._src.height // scale + 255) // 256,
            bits_per_sample=32,
            compression=1,
            tile_count=0,
        )
    
    def close(self):
        """Close the underlying rasterio dataset."""
        self._src.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self) -> str:
        return f"CogDatasetFallback(url={self._url!r})"


def open_cog(url: str, *, cache_size_mb: int = 256, prefer_native: bool = True) -> CogDataset:
    """Open a COG dataset, using native streaming if available.
    
    This is the recommended way to open COG files as it automatically
    selects the best available implementation.
    
    Args:
        url: HTTP(S) URL or file:// path to COG file
        cache_size_mb: Tile cache memory budget in MB
        prefer_native: If True, prefer native implementation over rasterio fallback
    
    Returns:
        CogDataset or CogDatasetFallback depending on availability
    
    Example:
        >>> ds = open_cog("https://example.com/dem.tif")
        >>> tile = ds.read_tile(0, 0)
    """
    if prefer_native and _COG_AVAILABLE:
        return CogDataset(url, cache_size_mb=cache_size_mb)
    
    try:
        return CogDatasetFallback(url, cache_size_mb=cache_size_mb)
    except RuntimeError:
        if _COG_AVAILABLE:
            return CogDataset(url, cache_size_mb=cache_size_mb)
        raise


__all__ = [
    "CogDataset",
    "CogDatasetFallback",
    "CogStats",
    "IfdInfo",
    "is_cog_available",
    "open_cog",
]
