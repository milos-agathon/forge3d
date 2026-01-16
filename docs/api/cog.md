# Cloud Optimized GeoTIFF (COG) Streaming

Phase 3 introduces streaming terrain data directly from Cloud Optimized GeoTIFFs
without pre-tiling. This enables working with large datasets hosted on cloud
storage without downloading entire files.

## Quick Start

```python
from forge3d.cog import open_cog

# Open a remote COG
ds = open_cog("https://example.com/terrain.tif", cache_size_mb=256)

# Check dataset info
print(f"Bounds: {ds.bounds}")
print(f"Overview levels: {ds.overview_count}")

# Read a tile at LOD 0 (full resolution)
tile = ds.read_tile(x=5, y=3, lod=0)
print(f"Tile shape: {tile.shape}")  # e.g., (256, 256)

# Check cache performance
stats = ds.stats()
print(f"Cache hit rate: {stats.hit_rate_percent:.1f}%")
```

## Features

- **HTTP Range Requests**: Fetches only required bytes from remote files
- **Overview Selection**: Automatic LOD selection from COG internal overviews
- **LRU Cache**: Memory-bounded tile caching with configurable budget
- **Statistics**: Hit rate, latency, and bandwidth monitoring
- **Fallback**: Rasterio-based fallback when native streaming unavailable

## Installation

COG streaming requires building forge3d with the `cog_streaming` feature:

```bash
maturin develop --release --features cog_streaming
```

If the feature is not available, a rasterio-based fallback is used (requires
`pip install rasterio`).

## API Reference

### CogDataset

The main class for streaming COG data.

```python
class CogDataset:
    def __init__(self, url: str, *, cache_size_mb: int = 256):
        """Open a COG for streaming.
        
        Args:
            url: HTTP(S) URL or file:// path to COG file
            cache_size_mb: Tile cache memory budget in MB
        """
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds (minx, miny, maxx, maxy)."""
    
    @property
    def overview_count(self) -> int:
        """Number of overview (pyramid) levels."""
    
    def read_tile(self, x: int, y: int, lod: int = 0) -> np.ndarray:
        """Read a single tile.
        
        Args:
            x: Tile X coordinate (0-indexed)
            y: Tile Y coordinate (0-indexed)
            lod: Level of detail (0 = full resolution)
        
        Returns:
            2D float32 array with shape (tile_height, tile_width)
        """
    
    def stats(self) -> CogStats:
        """Get cache statistics."""
    
    def ifd_info(self, level: int = 0) -> IfdInfo:
        """Get info about a specific overview level."""
```

### CogStats

Cache performance statistics.

```python
@dataclass
class CogStats:
    cache_hits: int          # Number of cache hits
    cache_misses: int        # Number of cache misses
    cache_evictions: int     # Number of tiles evicted
    memory_used_bytes: int   # Current memory usage
    memory_budget_bytes: int # Memory budget
    hit_rate_percent: float  # Hit rate as percentage
```

### IfdInfo

Information about a TIFF IFD (Image File Directory).

```python
@dataclass
class IfdInfo:
    width: int           # Image width in pixels
    height: int          # Image height in pixels
    tile_width: int      # Tile width in pixels
    tile_height: int     # Tile height in pixels
    tiles_across: int    # Number of tiles horizontally
    tiles_down: int      # Number of tiles vertically
    bits_per_sample: int # Bits per sample (e.g., 32)
    compression: int     # TIFF compression code
    tile_count: int      # Total number of tiles
```

### Helper Functions

```python
def is_cog_available() -> bool:
    """Check if native COG streaming is available."""

def open_cog(url: str, *, cache_size_mb: int = 256) -> CogDataset:
    """Open COG with automatic backend selection."""
```

## Usage Patterns

### Streaming Remote Data

```python
from forge3d.cog import open_cog

# USGS 3DEP elevation data (example URL)
url = "https://prd-tnm.s3.amazonaws.com/.../USGS_1_n47w122.tif"
ds = open_cog(url, cache_size_mb=512)

# Fetch tiles as needed
for y in range(ds.ifd_info().tiles_down):
    for x in range(ds.ifd_info().tiles_across):
        tile = ds.read_tile(x, y, lod=0)
        process_tile(tile)
```

### Working with Overviews

```python
ds = open_cog(url)

# Use coarser LOD for faster preview
preview_tile = ds.read_tile(0, 0, lod=2)  # 4x coarser

# Full resolution for final render
hires_tile = ds.read_tile(0, 0, lod=0)
```

### Monitoring Performance

```python
ds = open_cog(url, cache_size_mb=256)

# Fetch some tiles
for i in range(100):
    ds.read_tile(i % 10, i // 10, lod=0)

# Check performance
stats = ds.stats()
print(f"Hit rate: {stats.hit_rate_percent:.1f}%")
print(f"Memory: {stats.memory_used_bytes / 1024 / 1024:.1f} MB")

if stats.hit_rate_percent < 80:
    print("Consider increasing cache size")
```

### Local Files

```python
# Use file:// URL for local COGs
ds = open_cog("file:///path/to/local.tif")

# Or just the path (converted internally)
ds = open_cog("/path/to/local.tif")  # Works on Unix
```

## Example

See `examples/cog_streaming_demo.py` for a complete example:

```bash
# Stream from local DEM
python examples/cog_streaming_demo.py --local-dem fuji --stats

# Run benchmark
python examples/cog_streaming_demo.py --local-dem fuji --benchmark

# Show COG info
python examples/cog_streaming_demo.py --local-dem fuji --info
```

## Supported Formats

The COG reader supports:

- **Sample Formats**: Float32, Float64, Int16, UInt16, Int32, UInt8
- **Compression**: None, Deflate/zlib, LZW
- **Overviews**: Internal TIFF overviews (IFD chain)

Not yet supported:
- JPEG compression
- External overviews (.ovr files)

## Performance Tips

1. **Cache Size**: Set cache to ~25% of dataset size for good hit rates
2. **LOD Selection**: Use coarser LODs for distant terrain
3. **Prefetching**: Read tiles in spatial order for better HTTP performance
4. **Local Cache**: For repeated access, consider downloading the COG locally

## Troubleshooting

### "COG streaming is not available"

Rebuild with the feature flag:
```bash
maturin develop --release --features cog_streaming
```

### Slow performance

- Check cache hit rate with `ds.stats()`
- Increase `cache_size_mb` if hit rate is low
- Use coarser LODs for preview
- Consider local caching for repeated access

### "Tile not found" errors

- Verify tile coordinates are within bounds
- Check `ds.ifd_info()` for valid tile ranges
- Ensure LOD level exists (check `ds.overview_count`)

## See Also

- [Virtual Texture Streaming](../memory/virtual_texturing.md) - GPU-side texture streaming
- [Terrain Rendering](../terrain/terrain_rendering.rst) - Terrain visualization
- [Rasterio Tiles](../ingest/rasterio_tiles.md) - Alternative tile loading
