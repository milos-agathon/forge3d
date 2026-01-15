#!/usr/bin/env python3
"""P2.1/M5: Clipmap terrain scalability demo with real DEM data.

Demonstrates nested-ring clipmap terrain rendering with:
- Multiple LOD rings centered on camera
- Geo-morphing at LOD boundaries
- Triangle budget verification (≥40% reduction at distance)
- Integration with real DEM data from assets/tif/

Usage:
    python examples/clipmap_demo.py --dem assets/tif/Mount_Fuji_30m.tif
    python examples/clipmap_demo.py --dem assets/tif/dem_rainier.tif --rings 6
    python examples/clipmap_demo.py --synthetic --extent 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure forge3d is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

# Project root for resolving asset paths
PROJECT_ROOT = Path(__file__).parent.parent

# Available DEM files in assets/tif/
AVAILABLE_DEMS = {
    "fuji": PROJECT_ROOT / "assets/tif/Mount_Fuji_30m.tif",
    "rainier": PROJECT_ROOT / "assets/tif/dem_rainier.tif",
    "switzerland": PROJECT_ROOT / "assets/tif/switzerland_dem.tif",
    "luxembourg": PROJECT_ROOT / "assets/tif/luxembourg_dem.tif",
    "gore": PROJECT_ROOT / "assets/tif/Gore_Range_Albers_1m.tif",
}

DEFAULT_DEM = AVAILABLE_DEMS["fuji"]


def load_dem(dem_path: Path) -> tuple[np.ndarray, dict]:
    """Load DEM data from GeoTIFF file.
    
    Returns:
        Tuple of (height_data, metadata) where metadata contains:
        - width, height: raster dimensions
        - min_h, max_h: elevation range
        - pixel_size: ground resolution in meters (if available)
        - crs: coordinate reference system (if available)
    """
    try:
        import rasterio
        with rasterio.open(dem_path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            pixel_size_native = abs(transform[0])
            
            # Convert pixel size to meters if CRS is geographic (degrees)
            crs_str = str(src.crs) if src.crs else "unknown"
            if src.crs and src.crs.is_geographic:
                # Approximate conversion: 1 degree ≈ 111km at equator
                # Use center latitude for better accuracy
                center_lat = (src.bounds.top + src.bounds.bottom) / 2
                lat_correction = np.cos(np.radians(center_lat))
                pixel_size = pixel_size_native * 111000 * lat_correction  # degrees to meters
            else:
                pixel_size = pixel_size_native
            
            # Handle nodata
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            
            valid_data = data[~np.isnan(data)]
            metadata = {
                "width": src.width,
                "height": src.height,
                "min_h": float(valid_data.min()) if len(valid_data) > 0 else 0.0,
                "max_h": float(valid_data.max()) if len(valid_data) > 0 else 1.0,
                "pixel_size": pixel_size,
                "crs": crs_str,
                "bounds": src.bounds,
            }
            return data, metadata
    except ImportError:
        print("Warning: rasterio not available, using numpy fallback")
        return _load_dem_fallback(dem_path)


def _load_dem_fallback(dem_path: Path) -> tuple[np.ndarray, dict]:
    """Fallback DEM loader using basic numpy (limited functionality)."""
    # Try to load as raw binary or use synthetic data
    print(f"  Note: Install rasterio for full GeoTIFF support: pip install rasterio")
    # Generate synthetic terrain as fallback
    size = 512
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    # Procedural terrain: mountain with ridges
    data = (
        np.exp(-2 * (xx**2 + yy**2)) * 1000  # Central peak
        + np.sin(xx * 10) * np.cos(yy * 8) * 50  # Ridges
        + np.random.randn(size, size) * 10  # Noise
    ).astype(np.float32)
    
    metadata = {
        "width": size,
        "height": size,
        "min_h": float(data.min()),
        "max_h": float(data.max()),
        "pixel_size": 30.0,  # Assume 30m resolution
        "crs": "synthetic",
        "bounds": None,
    }
    return data, metadata


def generate_synthetic_dem(size: int = 512, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Generate synthetic terrain data for testing."""
    np.random.seed(seed)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Multi-octave noise for realistic terrain
    data = np.zeros((size, size), dtype=np.float32)
    for octave in range(5):
        freq = 2 ** octave
        amp = 1.0 / (octave + 1)
        data += amp * np.sin(xx * freq * 3 + np.random.rand() * 10)
        data += amp * np.cos(yy * freq * 3 + np.random.rand() * 10)
    
    # Add central mountain
    data += np.exp(-3 * (xx**2 + yy**2)) * 2
    
    # Scale to realistic elevation range (0-3000m)
    data = (data - data.min()) / (data.max() - data.min()) * 3000
    
    metadata = {
        "width": size,
        "height": size,
        "min_h": float(data.min()),
        "max_h": float(data.max()),
        "pixel_size": 30.0,
        "crs": "synthetic",
        "bounds": None,
    }
    return data, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P2.1/M5: Clipmap terrain scalability demo with real DEM data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available DEMs in assets/tif/:
  --dem fuji        Mount Fuji, Japan (30m resolution)
  --dem rainier     Mount Rainier, USA
  --dem switzerland Swiss Alps
  --dem luxembourg  Luxembourg
  --dem gore        Gore Range, Colorado (1m resolution)

Or specify a custom path:
  --dem /path/to/your/dem.tif
""",
    )
    parser.add_argument(
        "--dem", type=str, default=None,
        help="DEM file path or shortname (fuji, rainier, switzerland, luxembourg, gore)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic terrain data instead of real DEM"
    )
    parser.add_argument(
        "--rings", type=int, default=4,
        help="Number of LOD rings (default: 4)"
    )
    parser.add_argument(
        "--resolution", type=int, default=64,
        help="Grid resolution per ring (default: 64)"
    )
    parser.add_argument(
        "--extent", type=float, default=None,
        help="Terrain extent in world units (auto-calculated from DEM if not specified)"
    )
    parser.add_argument(
        "--center", type=float, nargs=2, default=None,
        metavar=("X", "Z"),
        help="Clipmap center position (default: terrain center)"
    )
    parser.add_argument(
        "--skirt-depth", type=float, default=None,
        help="Skirt depth for hiding LOD seams (auto-calculated if not specified)"
    )
    parser.add_argument(
        "--morph-range", type=float, default=0.3,
        help="Geo-morph blend range [0.0-1.0] (default: 0.3)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output image path (optional)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed mesh statistics"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from forge3d import ClipmapConfig, clipmap_generate_py, calculate_triangle_reduction_py
    except ImportError as e:
        print(f"Error: Could not import forge3d clipmap module: {e}")
        print("Make sure to run: maturin develop --release")
        return 1

    print("=" * 60)
    print("P2.1/M5: Clipmap Terrain Demo")
    print("=" * 60)

    # Load terrain data
    dem_data = None
    dem_metadata = None
    
    if args.synthetic:
        print("\nUsing synthetic terrain data...")
        dem_data, dem_metadata = generate_synthetic_dem()
    elif args.dem:
        # Resolve DEM path (shortname or full path)
        if args.dem in AVAILABLE_DEMS:
            dem_path = AVAILABLE_DEMS[args.dem]
        else:
            dem_path = Path(args.dem)
            if not dem_path.is_absolute():
                dem_path = PROJECT_ROOT / dem_path
        
        if not dem_path.exists():
            print(f"Error: DEM file not found: {dem_path}")
            print(f"Available shortnames: {', '.join(AVAILABLE_DEMS.keys())}")
            return 1
        
        print(f"\nLoading DEM: {dem_path.name}...")
        dem_data, dem_metadata = load_dem(dem_path)
    else:
        # Default: use Mount Fuji if available, otherwise synthetic
        if DEFAULT_DEM.exists():
            print(f"\nLoading default DEM: {DEFAULT_DEM.name}...")
            dem_data, dem_metadata = load_dem(DEFAULT_DEM)
        else:
            print("\nNo DEM specified and default not found, using synthetic terrain...")
            dem_data, dem_metadata = generate_synthetic_dem()

    # Print DEM statistics
    print(f"\n{'DEM Statistics':=^40}")
    print(f"  Dimensions:  {dem_metadata['width']} x {dem_metadata['height']}")
    print(f"  Elevation:   {dem_metadata['min_h']:.1f}m - {dem_metadata['max_h']:.1f}m")
    print(f"  Relief:      {dem_metadata['max_h'] - dem_metadata['min_h']:.1f}m")
    print(f"  Pixel size:  {dem_metadata['pixel_size']:.1f}m")
    print(f"  CRS:         {dem_metadata['crs']}")

    # Calculate terrain extent from DEM if not specified
    if args.extent is None:
        terrain_extent = dem_metadata['width'] * dem_metadata['pixel_size']
        print(f"  Auto-extent: {terrain_extent:.1f}m")
    else:
        terrain_extent = args.extent

    # Calculate center from DEM if not specified
    if args.center is None:
        center = (terrain_extent / 2.0, terrain_extent / 2.0)
    else:
        center = tuple(args.center)

    # Calculate skirt depth from terrain relief if not specified
    if args.skirt_depth is None:
        relief = dem_metadata['max_h'] - dem_metadata['min_h']
        skirt_depth = max(10.0, relief * 0.05)  # 5% of relief, min 10m
    else:
        skirt_depth = args.skirt_depth

    # Create clipmap configuration
    config = ClipmapConfig(
        ring_count=args.rings,
        ring_resolution=args.resolution,
        skirt_depth=skirt_depth,
        morph_range=args.morph_range,
    )
    print(f"\n{'Clipmap Configuration':=^40}")
    print(f"  {config}")
    print(f"  Center:      ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  Extent:      {terrain_extent:.1f}m")

    # Generate clipmap mesh
    print(f"\nGenerating clipmap mesh...")
    mesh = clipmap_generate_py(config, center, terrain_extent)

    # Print mesh statistics
    print(f"\n{'Mesh Statistics':=^40}")
    print(f"  Vertices:    {mesh.vertex_count:,}")
    print(f"  Triangles:   {mesh.triangle_count:,}")
    print(f"  Rings:       {mesh.rings_count}")
    
    # Calculate full-resolution triangle count for comparison
    full_res_triangles = dem_metadata['width'] * dem_metadata['height'] * 2
    actual_reduction = (1 - mesh.triangle_count / full_res_triangles) * 100 if full_res_triangles > 0 else 0
    
    print(f"\n{'LOD Reduction':=^40}")
    print(f"  Full-res triangles:   {full_res_triangles:,}")
    print(f"  Clipmap triangles:    {mesh.triangle_count:,}")
    print(f"  Triangle reduction:   {mesh.triangle_reduction_percent:.1f}% (internal)")
    print(f"  vs Full DEM:          {actual_reduction:.1f}%")
    
    # P2.1 exit criteria check
    if mesh.triangle_reduction_percent >= 40.0:
        print(f"  ✓ Meets P2.1 requirement (≥40%)")
    else:
        print(f"  ✗ Below P2.1 requirement (≥40%)")

    if args.verbose:
        print(f"\n{'Detailed Mesh Statistics':=^40}")
        
        # Get numpy arrays
        positions = mesh.positions()
        uvs = mesh.uvs()
        morph_data = mesh.morph_data()
        indices = mesh.indices()
        
        print(f"  Position bounds:")
        print(f"    X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]")
        print(f"    Z: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
        
        print(f"  UV bounds:")
        print(f"    U: [{uvs[:, 0].min():.3f}, {uvs[:, 0].max():.3f}]")
        print(f"    V: [{uvs[:, 1].min():.3f}, {uvs[:, 1].max():.3f}]")
        
        morph_weights = morph_data[:, 0]
        ring_indices = morph_data[:, 1]
        
        # Count skirt vertices (negative morph weight)
        skirt_count = np.sum(morph_weights < 0)
        print(f"  Skirt vertices: {skirt_count}")
        
        # Count vertices per ring
        for r in range(args.rings + 1):  # +1 for center block (ring 0)
            if r == 0:
                count = np.sum((ring_indices == 0) & (morph_weights >= 0))
                print(f"  Center block vertices: {count}")
            else:
                count = np.sum(ring_indices == r)
                print(f"  Ring {r} vertices: {count}")
        
        # Morph weight statistics
        valid_morph = morph_weights[morph_weights >= 0]
        if len(valid_morph) > 0:
            print(f"  Morph weights (non-skirt):")
            print(f"    Min: {valid_morph.min():.3f}")
            print(f"    Max: {valid_morph.max():.3f}")
            print(f"    Mean: {valid_morph.mean():.3f}")
        
        # DEM height sampling preview
        print(f"\n{'Height Sampling Preview':=^40}")
        sample_uvs = uvs[::max(1, len(uvs) // 10)]  # Sample 10 vertices
        for i, (u, v) in enumerate(sample_uvs[:5]):
            px = int(u * (dem_metadata['width'] - 1))
            py = int(v * (dem_metadata['height'] - 1))
            if 0 <= px < dem_metadata['width'] and 0 <= py < dem_metadata['height']:
                h = dem_data[py, px]
                if not np.isnan(h):
                    print(f"  UV({u:.2f}, {v:.2f}) → Pixel({px}, {py}) → Height: {h:.1f}m")

    print(f"\n{'=' * 60}")
    print("Clipmap demo complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
