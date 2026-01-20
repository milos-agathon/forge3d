#!/usr/bin/env python3
"""P3: Cloud Optimized GeoTIFF streaming demo.

Demonstrates streaming terrain from a remote COG without pre-tiling:
- HTTP range requests for on-demand tile fetching
- Overview selection for LOD
- LRU cache with memory budget
- Integration with terrain visualization

Usage:
    # Stream from a local COG file
    python examples/cog_streaming_demo.py --url file:///path/to/dem.tif
    
    # Stream from a remote COG (requires network access)
    python examples/cog_streaming_demo.py --url https://example.com/dem.tif
    
    # Use a sample local DEM as COG
    python examples/cog_streaming_demo.py --local-dem fuji
    
    # With cache stats monitoring
    python examples/cog_streaming_demo.py --url ... --stats
    
    # Run benchmark
    python examples/cog_streaming_demo.py --url ... --benchmark

Exit Criteria (P3.x/M6):
    - Renders terrain without pre-tiling
    - Cache hit/miss stats exposed
    - Deterministic output for same viewport
"""

from __future__ import annotations

import argparse
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

AVAILABLE_DEMS = {
    "fuji": PROJECT_ROOT / "assets/tif/Mount_Fuji_30m.tif",
    "rainier": PROJECT_ROOT / "assets/tif/dem_rainier.tif",
    "gore": PROJECT_ROOT / "assets/tif/Gore_Range_Albers_1m.tif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3: Stream terrain from Cloud Optimized GeoTIFF"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL to COG file (http://, https://, or file://)",
    )
    parser.add_argument(
        "--local-dem",
        type=str,
        choices=list(AVAILABLE_DEMS.keys()),
        help="Use a local DEM from assets/tif/ (converted to file:// URL)",
    )
    parser.add_argument(
        "--cache-mb",
        type=int,
        default=256,
        help="Tile cache memory budget in MB (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/cog_demo.png",
        help="Output image path",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(1024, 768),
        metavar=("W", "H"),
        help="Output resolution",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print cache statistics after operations",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark (fetch 100 tiles, report p50/p95)",
    )
    parser.add_argument(
        "--tiles",
        type=int,
        default=16,
        help="Number of tiles to fetch in each dimension for visualization",
    )
    parser.add_argument(
        "--lod",
        type=int,
        default=0,
        help="Level of detail to fetch (0=full res, higher=coarser)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print COG info and exit",
    )
    return parser.parse_args()


def run_benchmark(dataset, num_tiles: int = 100) -> dict:
    """Benchmark tile fetch latency."""
    info = dataset.ifd_info(0)
    max_x = max(1, info.tiles_across - 1)
    max_y = max(1, info.tiles_down - 1)
    max_lod = max(0, dataset.overview_count - 1)
    
    latencies = []
    for _ in range(num_tiles):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        lod = random.randint(0, min(max_lod, 3))
        
        start = time.perf_counter()
        try:
            _ = dataset.read_tile(x, y, lod)
        except Exception as e:
            print(f"  Warning: Failed to fetch tile ({x}, {y}, {lod}): {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
    
    if not latencies:
        return {"error": "No tiles fetched successfully"}
    
    latencies.sort()
    return {
        "count": len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) > 100 else latencies[-1],
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
    }


def fetch_tile_grid(dataset, num_tiles: int, lod: int) -> np.ndarray:
    """Fetch a grid of tiles and combine into single array."""
    info = dataset.ifd_info(lod)
    
    tiles_x = min(num_tiles, info.tiles_across)
    tiles_y = min(num_tiles, info.tiles_down)
    
    tile_w = info.tile_width
    tile_h = info.tile_height
    
    combined = np.zeros((tiles_y * tile_h, tiles_x * tile_w), dtype=np.float32)
    
    print(f"[FETCH] Fetching {tiles_x}x{tiles_y} tile grid at LOD {lod}...")
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            try:
                tile = dataset.read_tile(tx, ty, lod)
                y_start = ty * tile_h
                x_start = tx * tile_w
                h = min(tile_h, tile.shape[0])
                w = min(tile_w, tile.shape[1])
                combined[y_start:y_start + h, x_start:x_start + w] = tile[:h, :w]
            except Exception as e:
                print(f"  Warning: Failed tile ({tx}, {ty}): {e}")
    
    return combined


def visualize_heightmap(heightmap: np.ndarray, output_path: Path, output_size: tuple[int, int] | None = None):
    """Save heightmap as grayscale PNG, optionally resized to output_size.
    
    Args:
        heightmap: 2D numpy array of elevation values
        output_path: Path to save PNG
        output_size: Optional (width, height) tuple to resize output
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid = heightmap[np.isfinite(heightmap)]
    if len(valid) == 0:
        print("Warning: No valid height values")
        return
    
    h_min = np.percentile(valid, 1)
    h_max = np.percentile(valid, 99)
    
    if h_max <= h_min:
        h_max = h_min + 1
    
    normalized = np.clip((heightmap - h_min) / (h_max - h_min), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    img_data = (normalized * 255).astype(np.uint8)
    
    try:
        from PIL import Image
        img = Image.fromarray(img_data, mode='L')
        
        # Resize to output_size if specified
        if output_size is not None:
            w, h = output_size
            img = img.resize((w, h), Image.LANCZOS)
            print(f"[OUTPUT] Resampled to {w}x{h}")
        
        img.save(output_path)
        print(f"[OUTPUT] Saved heightmap to {output_path}")
    except ImportError:
        import struct
        import zlib
        
        def write_png(data, path):
            height, width = data.shape
            
            def png_chunk(chunk_type, chunk_data):
                chunk = chunk_type + chunk_data
                crc = zlib.crc32(chunk) & 0xffffffff
                return struct.pack('>I', len(chunk_data)) + chunk + struct.pack('>I', crc)
            
            signature = b'\x89PNG\r\n\x1a\n'
            ihdr = struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)
            
            raw_data = b''
            for row in data:
                raw_data += b'\x00' + bytes(row)
            
            compressed = zlib.compress(raw_data, 9)
            
            with open(path, 'wb') as f:
                f.write(signature)
                f.write(png_chunk(b'IHDR', ihdr))
                f.write(png_chunk(b'IDAT', compressed))
                f.write(png_chunk(b'IEND', b''))
        
        write_png(img_data, output_path)
        print(f"[OUTPUT] Saved heightmap to {output_path}")


def print_cog_info(dataset):
    """Print detailed COG information."""
    print(f"\n{'='*60}")
    print("COG Dataset Information")
    print(f"{'='*60}")
    print(f"URL: {dataset.url}")
    print(f"Bounds: {dataset.bounds}")
    print(f"Overview levels: {dataset.overview_count}")
    
    for level in range(dataset.overview_count):
        info = dataset.ifd_info(level)
        print(f"\n  Level {level}:")
        print(f"    Dimensions: {info.width} x {info.height}")
        print(f"    Tile size: {info.tile_width} x {info.tile_height}")
        print(f"    Tile grid: {info.tiles_across} x {info.tiles_down}")
        print(f"    Total tiles: {info.tile_count}")
        print(f"    Bits/sample: {info.bits_per_sample}")
        print(f"    Compression: {info.compression}")
    print(f"{'='*60}\n")


def print_stats(dataset):
    """Print cache statistics."""
    stats = dataset.stats()
    print(f"\n[STATS] Cache performance:")
    print(f"  Hits:       {stats.cache_hits}")
    print(f"  Misses:     {stats.cache_misses}")
    print(f"  Hit rate:   {stats.hit_rate_percent:.1f}%")
    print(f"  Evictions:  {stats.cache_evictions}")
    print(f"  Memory:     {stats.memory_used_bytes / 1024 / 1024:.1f} / {stats.memory_budget_bytes / 1024 / 1024:.1f} MB")


def main() -> int:
    args = parse_args()
    
    if not args.url and not args.local_dem:
        print("Error: Must specify --url or --local-dem")
        print("Example: python cog_streaming_demo.py --local-dem fuji")
        return 1
    
    if args.local_dem:
        dem_path = AVAILABLE_DEMS[args.local_dem]
        if not dem_path.exists():
            print(f"Error: DEM file not found: {dem_path}")
            return 1
        url = f"file://{dem_path.absolute()}"
    else:
        url = args.url
    
    print(f"[COG] Opening: {url}")
    print(f"[COG] Cache budget: {args.cache_mb} MB")
    
    try:
        from forge3d.cog import open_cog, is_cog_available
        
        if not is_cog_available():
            print("[INFO] Native COG streaming not available, using rasterio fallback")
        
        dataset = open_cog(url, cache_size_mb=args.cache_mb)
        
    except Exception as e:
        print(f"Error opening COG: {e}")
        print("\nTrying rasterio fallback...")
        
        try:
            from forge3d.cog import CogDatasetFallback
            dataset = CogDatasetFallback(url, cache_size_mb=args.cache_mb)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return 1
    
    print(f"[COG] Bounds: {dataset.bounds}")
    print(f"[COG] Overview levels: {dataset.overview_count}")
    
    if args.info:
        print_cog_info(dataset)
        return 0
    
    if args.benchmark:
        print("\n[BENCHMARK] Running tile fetch latency test...")
        results = run_benchmark(dataset)
        if "error" in results:
            print(f"  Error: {results['error']}")
        else:
            print(f"  Tiles fetched: {results['count']}")
            print(f"  p50:  {results['p50_ms']:.1f} ms")
            print(f"  p95:  {results['p95_ms']:.1f} ms")
            print(f"  p99:  {results['p99_ms']:.1f} ms")
            print(f"  mean: {results['mean_ms']:.1f} ms")
            print(f"  min:  {results['min_ms']:.1f} ms")
            print(f"  max:  {results['max_ms']:.1f} ms")
        
        if args.stats:
            print_stats(dataset)
        return 0
    
    heightmap = fetch_tile_grid(dataset, args.tiles, args.lod)
    print(f"[DATA] Combined heightmap shape: {heightmap.shape}")
    print(f"[DATA] Height range: {np.nanmin(heightmap):.1f} - {np.nanmax(heightmap):.1f}")
    
    visualize_heightmap(heightmap, args.output, output_size=tuple(args.size))
    
    if args.stats:
        print_stats(dataset)
    
    print("\n[DONE] COG streaming demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
