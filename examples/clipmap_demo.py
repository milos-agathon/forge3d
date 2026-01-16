#!/usr/bin/env python3
"""P2.1/M5: Clipmap terrain scalability demo with real DEM data.

Demonstrates nested-ring clipmap terrain rendering with:
- Multiple LOD rings centered on camera
- Geo-morphing at LOD boundaries (P2.2)
- GPU LOD selection visualization (P2.3)
- Triangle budget verification (≥40% reduction at distance)
- Integration with real DEM data from assets/tif/

Usage:
    python examples/clipmap_demo.py --dem fuji
    python examples/clipmap_demo.py --dem rainier --rings 6
    python examples/clipmap_demo.py --synthetic --extent 10000
    
Visualization (P2.2/P2.3):
    python examples/clipmap_demo.py --dem fuji --visualize
    python examples/clipmap_demo.py --dem fuji --lod-stats
    python examples/clipmap_demo.py --dem fuji --visualize --lod-stats --output clipmap_vis.png
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


# =============================================================================
# P2.2/P2.3 Visualization Functions
# =============================================================================

# Ring colors for LOD visualization (from fine to coarse)
RING_COLORS = [
    (0.2, 0.6, 1.0),   # Blue - center/finest
    (0.2, 0.8, 0.4),   # Green - ring 1
    (1.0, 0.8, 0.2),   # Yellow - ring 2
    (1.0, 0.5, 0.2),   # Orange - ring 3
    (1.0, 0.2, 0.2),   # Red - ring 4
    (0.8, 0.2, 0.8),   # Purple - ring 5
    (0.5, 0.5, 0.5),   # Gray - ring 6+
]


def create_visualization_image(
    mesh,
    dem_data: np.ndarray,
    dem_metadata: dict,
    terrain_extent: float,
    center: tuple[float, float],
    config,
    output_size: int = 1024,
) -> np.ndarray:
    """Create a visualization image showing clipmap structure and geo-morphing.
    
    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255].
    """
    positions = mesh.positions()
    morph_data = mesh.morph_data()
    
    # Create output image
    img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Calculate bounds
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    z_min, z_max = positions[:, 1].min(), positions[:, 1].max()
    extent = max(x_max - x_min, z_max - z_min) * 1.1
    cx, cz = center
    
    def world_to_pixel(wx, wz):
        px = int((wx - cx + extent/2) / extent * output_size)
        pz = int((wz - cz + extent/2) / extent * output_size)
        return px, output_size - 1 - pz  # Flip Y
    
    # Draw terrain background (grayscale heightmap)
    for py in range(output_size):
        for px in range(output_size):
            # Convert pixel to world coordinates
            wx = cx - extent/2 + px / output_size * extent
            wz = cz + extent/2 - py / output_size * extent
            
            # Convert to UV
            u = (wx + terrain_extent * 0.5) / terrain_extent
            v = (wz + terrain_extent * 0.5) / terrain_extent
            
            if 0 <= u <= 1 and 0 <= v <= 1:
                tx = int(u * (dem_metadata['width'] - 1))
                ty = int(v * (dem_metadata['height'] - 1))
                if 0 <= tx < dem_metadata['width'] and 0 <= ty < dem_metadata['height']:
                    h = dem_data[ty, tx]
                    if not np.isnan(h):
                        # Normalize height to grayscale
                        h_norm = (h - dem_metadata['min_h']) / max(dem_metadata['max_h'] - dem_metadata['min_h'], 1)
                        gray = int(h_norm * 100 + 30)  # Dark background
                        img[py, px] = [gray, gray, gray]
    
    # Draw vertices colored by ring index
    morph_weights = morph_data[:, 0]
    ring_indices = morph_data[:, 1]
    
    for i, (pos, mw, ri) in enumerate(zip(positions, morph_weights, ring_indices)):
        px, py = world_to_pixel(pos[0], pos[1])
        if 0 <= px < output_size and 0 <= py < output_size:
            # Get ring color
            ring_idx = int(ri) if mw >= 0 else 0
            color = RING_COLORS[min(ring_idx, len(RING_COLORS) - 1)]
            
            # Modulate by morph weight
            if mw >= 0:
                brightness = 0.5 + 0.5 * (1 - mw)  # Brighter = less morphed
            else:
                brightness = 0.3  # Skirt vertices are dim
            
            r = int(color[0] * brightness * 255)
            g = int(color[1] * brightness * 255)
            b = int(color[2] * brightness * 255)
            
            # Draw vertex as small dot
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if 0 <= px+dx < output_size and 0 <= py+dy < output_size:
                        img[py+dy, px+dx] = [r, g, b]
    
    # Draw ring boundaries
    draw_ring_boundaries(img, center, config, terrain_extent, extent, output_size)
    
    return img


def create_morph_weight_heatmap(
    mesh,
    terrain_extent: float,
    center: tuple[float, float],
    output_size: int = 1024,
) -> np.ndarray:
    """Create a heatmap visualization of morph weights.
    
    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255].
    """
    positions = mesh.positions()
    morph_data = mesh.morph_data()
    
    # Create output image (black background)
    img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Calculate bounds
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    z_min, z_max = positions[:, 1].min(), positions[:, 1].max()
    extent = max(x_max - x_min, z_max - z_min) * 1.1
    cx, cz = center
    
    def world_to_pixel(wx, wz):
        px = int((wx - cx + extent/2) / extent * output_size)
        pz = int((wz - cz + extent/2) / extent * output_size)
        return px, output_size - 1 - pz
    
    morph_weights = morph_data[:, 0]
    
    for pos, mw in zip(positions, morph_weights):
        px, py = world_to_pixel(pos[0], pos[1])
        if 0 <= px < output_size and 0 <= py < output_size:
            if mw < 0:
                # Skirt vertex - cyan
                color = (0, 200, 200)
            else:
                # Heatmap: blue (0) -> green (0.5) -> red (1)
                if mw < 0.5:
                    t = mw * 2
                    color = (0, int(255 * t), int(255 * (1 - t)))
                else:
                    t = (mw - 0.5) * 2
                    color = (int(255 * t), int(255 * (1 - t)), 0)
            
            # Draw vertex
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if 0 <= px+dx < output_size and 0 <= py+dy < output_size:
                        img[py+dy, px+dx] = color
    
    return img


def create_lod_zones_visualization(
    mesh,
    terrain_extent: float,
    center: tuple[float, float],
    config,
    camera_distance: float = 1000.0,
    output_size: int = 1024,
) -> np.ndarray:
    """Create visualization of LOD selection zones based on distance.
    
    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255].
    """
    positions = mesh.positions()
    morph_data = mesh.morph_data()
    
    # Create output image
    img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Calculate bounds
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    z_min, z_max = positions[:, 1].min(), positions[:, 1].max()
    extent = max(x_max - x_min, z_max - z_min) * 1.1
    cx, cz = center
    
    # Draw distance-based LOD zones (concentric circles)
    for py in range(output_size):
        for px in range(output_size):
            # Convert pixel to world coordinates
            wx = cx - extent/2 + px / output_size * extent
            wz = cz + extent/2 - py / output_size * extent
            
            # Calculate distance from center
            dist = np.sqrt((wx - cx)**2 + (wz - cz)**2)
            
            # Determine LOD zone based on distance
            # LOD zones double in size for each level
            base_zone_size = extent / (2 ** (config.ring_count + 1))
            lod = 0
            zone_boundary = base_zone_size
            while dist > zone_boundary and lod < config.ring_count:
                lod += 1
                zone_boundary *= 2
            
            # Color by LOD zone (darker = coarser LOD)
            color = RING_COLORS[min(lod, len(RING_COLORS) - 1)]
            intensity = 0.3 + 0.1 * (config.ring_count - lod)
            img[py, px] = [
                int(color[0] * intensity * 255),
                int(color[1] * intensity * 255),
                int(color[2] * intensity * 255),
            ]
    
    # Overlay actual vertex positions
    ring_indices = morph_data[:, 1]
    for pos, ri in zip(positions, ring_indices):
        px = int((pos[0] - cx + extent/2) / extent * output_size)
        py = output_size - 1 - int((pos[1] - cz + extent/2) / extent * output_size)
        if 0 <= px < output_size and 0 <= py < output_size:
            img[py, px] = [255, 255, 255]  # White vertices
    
    return img


def draw_ring_boundaries(
    img: np.ndarray,
    center: tuple[float, float],
    config,
    terrain_extent: float,
    extent: float,
    output_size: int,
):
    """Draw ring boundaries on the visualization image."""
    cx, cz = center
    
    # Calculate ring extents
    base_cell_size = terrain_extent / (config.center_resolution * 8.0)
    
    # Center block boundary
    center_half = base_cell_size * config.center_resolution * 0.5
    draw_square(img, cx, cz, center_half, extent, output_size, (255, 255, 255))
    
    # Ring boundaries
    current_extent = center_half
    for ring_idx in range(config.ring_count):
        ring_extent = base_cell_size * config.ring_resolution * (1 << ring_idx)
        current_extent += ring_extent
        color = RING_COLORS[min(ring_idx + 1, len(RING_COLORS) - 1)]
        color_int = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        draw_square(img, cx, cz, current_extent, extent, output_size, color_int)


def draw_square(
    img: np.ndarray,
    cx: float, cz: float,
    half_size: float,
    extent: float,
    output_size: int,
    color: tuple[int, int, int],
):
    """Draw a square boundary on the image."""
    def world_to_pixel(wx, wz):
        px = int((wx - cx + extent/2) / extent * output_size)
        pz = int((wz - cz + extent/2) / extent * output_size)
        return px, output_size - 1 - pz
    
    # Draw four edges
    corners = [
        (cx - half_size, cz - half_size),
        (cx + half_size, cz - half_size),
        (cx + half_size, cz + half_size),
        (cx - half_size, cz + half_size),
    ]
    
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        draw_line(img, p1, p2, cx, cz, extent, output_size, color)


def draw_line(
    img: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    cx: float, cz: float,
    extent: float,
    output_size: int,
    color: tuple[int, int, int],
):
    """Draw a line on the image using Bresenham's algorithm."""
    def world_to_pixel(wx, wz):
        px = int((wx - cx + extent/2) / extent * output_size)
        pz = int((wz - cz + extent/2) / extent * output_size)
        return px, output_size - 1 - pz
    
    x1, y1 = world_to_pixel(p1[0], p1[1])
    x2, y2 = world_to_pixel(p2[0], p2[1])
    
    # Simple line drawing
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for i in range(steps + 1):
        t = i / steps
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        if 0 <= px < output_size and 0 <= py < output_size:
            img[py, px] = color


def print_lod_statistics(
    mesh,
    dem_metadata: dict,
    terrain_extent: float,
    center: tuple[float, float],
    config,
    camera_height: float = 1000.0,
    viewport_height: int = 1080,
    fov_y: float = 45.0,
):
    """Print detailed LOD selection statistics (P2.3)."""
    positions = mesh.positions()
    morph_data = mesh.morph_data()
    morph_weights = morph_data[:, 0]
    ring_indices = morph_data[:, 1]
    
    print(f"\n{'P2.3 LOD Selection Statistics':=^50}")
    
    # Calculate triangles per ring
    print(f"\n  Triangle Distribution by LOD Ring:")
    indices = mesh.indices()
    total_triangles = len(indices) // 3
    
    # Estimate triangles per ring based on vertex distribution
    ring_vertex_counts = {}
    for ri in ring_indices:
        ri_int = int(ri) if ri >= 0 else -1
        ring_vertex_counts[ri_int] = ring_vertex_counts.get(ri_int, 0) + 1
    
    total_verts = len(positions)
    for ring_idx in sorted(ring_vertex_counts.keys()):
        count = ring_vertex_counts[ring_idx]
        pct = count / total_verts * 100
        tri_estimate = int(total_triangles * count / total_verts)
        lod_name = "Center" if ring_idx == 0 else f"Ring {ring_idx}" if ring_idx > 0 else "Skirt"
        print(f"    {lod_name:10s}: {count:5d} verts ({pct:5.1f}%), ~{tri_estimate:,} triangles")
    
    # Screen-space error analysis
    print(f"\n  Screen-Space Error Analysis:")
    print(f"    Viewport height: {viewport_height}px")
    print(f"    FOV (vertical):  {fov_y}°")
    print(f"    Camera height:   {camera_height}m")
    
    cx, cz = center
    fov_rad = np.radians(fov_y)
    
    # Calculate screen-space error at different distances
    print(f"\n  Projected Pixel Size by Distance:")
    distances = [100, 500, 1000, 2000, 5000, 10000]
    for dist in distances:
        if dist > terrain_extent:
            continue
        # Pixels per meter at this distance
        pixels_per_meter = (viewport_height * 0.5) / (dist * np.tan(fov_rad * 0.5))
        cell_size = terrain_extent / dem_metadata['width']
        projected_cell = cell_size * pixels_per_meter
        
        # Determine which LOD would be appropriate
        # Target: ~2 pixels per cell
        target_pixels = 2.0
        ideal_lod = 0
        while projected_cell / (2 ** ideal_lod) > target_pixels and ideal_lod < config.ring_count:
            ideal_lod += 1
        
        print(f"    {dist:5d}m: {pixels_per_meter:.2f} px/m, cell={projected_cell:.1f}px → LOD {ideal_lod}")
    
    # Frustum culling statistics
    print(f"\n  Frustum Culling Potential:")
    # Calculate how much of the clipmap is typically visible
    visible_radius = camera_height * np.tan(fov_rad)  # Approximate visible ground radius
    clipmap_radius = terrain_extent * 0.5
    visible_area = min(1.0, (visible_radius / clipmap_radius) ** 2)
    print(f"    Visible radius (approx): {visible_radius:.0f}m")
    print(f"    Clipmap radius:          {clipmap_radius:.0f}m")
    print(f"    Visible fraction:        {visible_area * 100:.1f}%")
    print(f"    Potential cull savings:  {(1 - visible_area) * 100:.1f}%")
    
    # Geo-morphing statistics (P2.2)
    print(f"\n{'P2.2 Geo-Morphing Statistics':=^50}")
    
    valid_morph = morph_weights[morph_weights >= 0]
    if len(valid_morph) > 0:
        print(f"\n  Morph Weight Distribution:")
        print(f"    Range:    [{valid_morph.min():.3f}, {valid_morph.max():.3f}]")
        print(f"    Mean:     {valid_morph.mean():.3f}")
        print(f"    Std Dev:  {valid_morph.std():.3f}")
        
        # Histogram
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(valid_morph, bins=bins)
        print(f"\n  Morph Weight Histogram:")
        max_count = max(hist) if max(hist) > 0 else 1
        for i, count in enumerate(hist):
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}]: {count:5d} {bar}")
    
    # Boundary vertex analysis
    print(f"\n  Boundary Vertex Analysis:")
    boundary_threshold = 0.8
    boundary_verts = np.sum(valid_morph > boundary_threshold)
    inner_verts = np.sum(valid_morph < 0.2)
    transition_verts = len(valid_morph) - boundary_verts - inner_verts
    print(f"    Inner (morph < 0.2):      {inner_verts:5d} ({inner_verts/len(valid_morph)*100:.1f}%)")
    print(f"    Transition (0.2-0.8):     {transition_verts:5d} ({transition_verts/len(valid_morph)*100:.1f}%)")
    print(f"    Boundary (morph > 0.8):   {boundary_verts:5d} ({boundary_verts/len(valid_morph)*100:.1f}%)")


def save_visualization(img: np.ndarray, output_path: Path, label: str = ""):
    """Save visualization image to file."""
    try:
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img.save(output_path)
        print(f"  Saved {label}: {output_path}")
        return True
    except ImportError:
        # Fallback: save as raw numpy
        np.save(output_path.with_suffix('.npy'), img)
        print(f"  Saved {label} (numpy): {output_path.with_suffix('.npy')}")
        print(f"  Note: Install Pillow for PNG output: pip install Pillow")
        return True


def create_combined_visualization(
    mesh,
    dem_data: np.ndarray,
    dem_metadata: dict,
    terrain_extent: float,
    center: tuple[float, float],
    config,
    output_path: Path | None = None,
) -> np.ndarray:
    """Create a combined visualization with all three views.
    
    Returns:
        Combined RGB image (H, W*3, 3).
    """
    size = 512
    
    # Create three visualizations
    print(f"\n{'Creating Visualizations':=^40}")
    
    print("  Generating ring structure view...")
    ring_img = create_visualization_image(
        mesh, dem_data, dem_metadata, terrain_extent, center, config, size
    )
    
    print("  Generating morph weight heatmap...")
    morph_img = create_morph_weight_heatmap(mesh, terrain_extent, center, size)
    
    print("  Generating LOD zones view...")
    lod_img = create_lod_zones_visualization(mesh, terrain_extent, center, config, 1000.0, size)
    
    # Combine horizontally
    combined = np.concatenate([ring_img, morph_img, lod_img], axis=1)
    
    # Add labels
    add_text_label(combined, "Ring Structure", 0, size)
    add_text_label(combined, "Morph Weights", size, size)
    add_text_label(combined, "LOD Zones", size * 2, size)
    
    if output_path:
        save_visualization(combined, output_path, "combined visualization")
    
    return combined


def add_text_label(img: np.ndarray, text: str, x_offset: int, width: int):
    """Add a simple text label to the image (basic pixel font)."""
    # Simple approach: draw a dark bar with text approximation
    h, w = img.shape[:2]
    bar_height = 20
    
    # Dark background bar
    img[:bar_height, x_offset:x_offset+width] = [30, 30, 30]
    
    # Simple "font" - just indicate position with white bar
    text_len = len(text) * 8
    start_x = x_offset + (width - text_len) // 2
    img[5:15, start_x:start_x+text_len] = [200, 200, 200]


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
        help="Output image path for visualization (default: clipmap_visualization.png)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed mesh statistics"
    )
    # P2.2/P2.3 Visualization options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization images showing ring structure, morph weights, and LOD zones"
    )
    parser.add_argument(
        "--lod-stats", action="store_true",
        help="Print detailed LOD selection and geo-morphing statistics (P2.2/P2.3)"
    )
    parser.add_argument(
        "--camera-height", type=float, default=1000.0,
        help="Simulated camera height for LOD calculations (default: 1000m)"
    )
    parser.add_argument(
        "--viewport-height", type=int, default=1080,
        help="Viewport height for screen-space error calculations (default: 1080)"
    )
    parser.add_argument(
        "--fov", type=float, default=45.0,
        help="Camera field of view in degrees (default: 45)"
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

    # P2.2/P2.3: LOD Statistics
    if args.lod_stats:
        print_lod_statistics(
            mesh,
            dem_metadata,
            terrain_extent,
            center,
            config,
            camera_height=args.camera_height,
            viewport_height=args.viewport_height,
            fov_y=args.fov,
        )

    # P2.2/P2.3: Visualization
    if args.visualize:
        output_path = args.output or Path("clipmap_visualization.png")
        create_combined_visualization(
            mesh,
            dem_data,
            dem_metadata,
            terrain_extent,
            center,
            config,
            output_path=output_path,
        )

    print(f"\n{'=' * 60}")
    print("Clipmap demo complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
