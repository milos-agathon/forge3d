#!/usr/bin/env python3
"""Swiss terrain viewer with land cover overlay.

This script launches an interactive 3D viewer showing Switzerland DEM elevation
with land cover classification draped as a lit overlay. The land cover is
resampled to match DEM resolution and projected to EPSG:3035.

**Features:**
- Elevation from switzerland_dem.tif
- Land cover overlay from switzerland_land_cover.tif (resampled to match DEM)
- EPSG:3035 (LAEA Europe) projection
- 4 high-quality snapshot presets
- Full PBR rendering with lit overlays

**High-Quality Presets:**
1. `--preset hq1` - Standard high quality (4K, MSAA 8x, height AO, sun visibility)
2. `--preset hq2` - Alpine preset (snow/rock layers, cool temperature)
3. `--preset hq3` - Cinematic (DoF, lens effects, warm tones)
4. `--preset hq4` - Maximum quality (all effects combined)

Usage:
    # Basic interactive viewer
    python examples/swiss_terrain_landcover_viewer.py
    
    # With high-quality preset 1
    python examples/swiss_terrain_landcover_viewer.py --preset hq1
    
    # Take snapshot with preset 4 (maximum quality)
    python examples/swiss_terrain_landcover_viewer.py --preset hq4 --snapshot swiss_render.png
    
    # Custom overlay opacity
    python examples/swiss_terrain_landcover_viewer.py --overlay-opacity 0.7

Interactive Commands:
    overlay on/off        Toggle land cover overlay
    overlay opacity=0.5   Set overlay opacity
    snap <path>           Take screenshot
    quit                  Exit viewer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Optional: rasterio for GeoTIFF handling and reprojection
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Install with: pip install rasterio")
    print("         Land cover reprojection/resampling will be skipped.")

# Optional: PIL for image saving
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Land cover legend configuration (matching R script colors)
LANDCOVER_LEGEND = [
    ("#419bdf", "Water"),
    ("#397d49", "Trees"),
    ("#7a87c6", "Flooded vegetation"),
    ("#e49635", "Crops"),
    ("#c4281b", "Built area"),
    ("#a59b8f", "Bare ground"),
    ("#a8ebff", "Snow"),
    ("#e3e2c3", "Rangeland"),
]


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_legend_image(width: int = 400, height: int = 600) -> Image.Image:
    """Create a legend image with land cover classes and colors.
    
    Args:
        width: Legend image width
        height: Legend image height
        
    Returns:
        PIL Image with transparent background containing the legend
    """
    if not HAS_PIL:
        return None
    
    # Create transparent image
    legend = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(legend)
    
    # Try to load a nice font, fall back to default
    font_size = height // 14  # Larger title font
    small_font_size = height // 18  # Larger label font for better readability
    try:
        # Try common system fonts
        for font_name in ['arial.ttf', 'Arial.ttf', 'DejaVuSans.ttf', 'Helvetica.ttf']:
            try:
                font = ImageFont.truetype(font_name, font_size)
                small_font = ImageFont.truetype(font_name, small_font_size)
                break
            except (OSError, IOError):
                continue
        else:
            font = ImageFont.load_default()
            small_font = font
    except Exception:
        font = ImageFont.load_default()
        small_font = font
    
    # Layout parameters - ensure content fits within bounds
    left_padding = width // 12
    top_padding = height // 15
    circle_radius = height // 28
    
    # Calculate vertical spacing based on number of items
    # Title + items need to fit with some bottom margin
    num_items = len(LANDCOVER_LEGEND)
    title_height = font_size + top_padding
    available_height = height - title_height - top_padding  # Leave margin at bottom
    line_height = available_height // num_items
    
    # Draw title - left-aligned with legend items
    title = "Land Cover"
    title_x = left_padding + circle_radius  # Align with circle centers
    draw.text((title_x, top_padding), title, fill=(50, 50, 50, 255), font=font)
    
    # Start items below title
    start_y = title_height + top_padding // 2
    
    # Draw legend items
    for i, (color_hex, label) in enumerate(LANDCOVER_LEGEND):
        y = start_y + i * line_height
        rgb = hex_to_rgb(color_hex)
        
        # Draw filled circle
        circle_x = left_padding + circle_radius
        circle_y = y + line_height // 2
        draw.ellipse(
            [circle_x - circle_radius, circle_y - circle_radius,
             circle_x + circle_radius, circle_y + circle_radius],
            fill=(*rgb, 255),
            outline=(80, 80, 80, 255),
            width=max(2, circle_radius // 8)  # Scale outline width
        )
        
        # Draw label text - aligned to the right of circles with consistent spacing
        text_x = circle_x + circle_radius + left_padding // 2
        text_y = circle_y - small_font_size // 2
        draw.text((text_x, text_y), label, fill=(40, 40, 40, 255), font=small_font)
    
    return legend


def add_legend_to_image(image_path: Path, output_path: Path = None, 
                        legend_scale: float = 0.3, position: str = "west",
                        max_legend_height: int = 400, transparent_background: bool = True) -> Path:
    """Add land cover legend to an image.
    
    The legend is created at full map-proportional resolution (for crisp fonts),
    then scaled down to fit within max_legend_height if necessary.
    
    Args:
        image_path: Path to the source image
        output_path: Path for output (default: overwrites source)
        legend_scale: Scale factor for legend relative to image height
        position: Legend position ("west", "east", "northwest", "southwest")
        max_legend_height: Maximum legend height in pixels to prevent oversized legends
        transparent_background: If True, legend has transparent background; if False, semi-transparent white
        
    Returns:
        Path to the output image
    """
    if not HAS_PIL:
        print("Warning: PIL not available, cannot add legend")
        return image_path
    
    if output_path is None:
        output_path = image_path
    
    # Load the map image
    map_img = Image.open(image_path).convert('RGBA')
    map_width, map_height = map_img.size
    
    # Create legend at full map-proportional resolution (for crisp fonts/graphics)
    full_legend_height = int(map_height * legend_scale)
    full_legend_width = int(full_legend_height * 1.2)
    legend_img = create_legend_image(full_legend_width, full_legend_height)
    
    if legend_img is None:
        return image_path
    
    # Scale down to max_legend_height if needed (preserves font quality via high-res rendering)
    if full_legend_height > max_legend_height:
        scale_factor = max_legend_height / full_legend_height
        final_legend_width = int(full_legend_width * scale_factor)
        final_legend_height = max_legend_height
        legend_img = legend_img.resize((final_legend_width, final_legend_height), Image.LANCZOS)
    else:
        final_legend_width = full_legend_width
        final_legend_height = full_legend_height
    
    # Calculate position
    margin = int(map_width * 0.02)
    
    if position == "west":
        x = margin
        y = (map_height - final_legend_height) // 2
    elif position == "east":
        x = map_width - final_legend_width - margin
        y = (map_height - final_legend_height) // 2
    elif position == "northwest":
        x = margin
        y = margin
    elif position == "southwest":
        x = margin
        y = map_height - final_legend_height - margin
    elif position == "northeast":
        x = map_width - final_legend_width - margin
        y = margin
    elif position == "southeast":
        x = map_width - final_legend_width - margin
        y = map_height - final_legend_height - margin
    else:
        x = margin
        y = (map_height - final_legend_height) // 2
    
    # Optionally add semi-transparent white background behind legend for readability
    if not transparent_background:
        bg_padding = int(final_legend_width * 0.05)
        bg = Image.new('RGBA', (final_legend_width + bg_padding * 2, final_legend_height + bg_padding * 2), 
                       (255, 255, 255, 200))
        # Composite background then legend
        map_img.paste(bg, (x - bg_padding, y - bg_padding), bg)
    
    # Composite legend onto map
    map_img.paste(legend_img, (x, y), legend_img)
    
    # Save result
    map_img.save(output_path)
    return output_path


def find_viewer_binary() -> str:
    """Find the interactive_viewer binary."""
    import platform
    ext = ".exe" if platform.system() == "Windows" else ""
    candidates = [
        Path(__file__).parent.parent / "target" / "release" / f"interactive_viewer{ext}",
        Path(__file__).parent.parent / "target" / "debug" / f"interactive_viewer{ext}",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "interactive_viewer binary not found. "
        "Build with: cargo build --release --bin interactive_viewer"
    )


def send_ipc(sock: socket.socket, cmd: dict) -> dict:
    """Send an IPC command and receive response."""
    msg = json.dumps(cmd) + "\n"
    sock.sendall(msg.encode())
    
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    
    response_str = data.decode().strip()
    if response_str:
        return json.loads(response_str)
    return {"ok": False, "error": "Empty response"}


def get_default_paths() -> tuple[Path, Path]:
    """Get default paths for Switzerland DEM and land cover."""
    base = Path(__file__).parent.parent / "assets" / "tif"
    dem_path = base / "switzerland_dem.tif"
    landcover_path = base / "switzerland_land_cover.tif"
    return dem_path, landcover_path


def resample_landcover_to_dem(
    landcover_path: Path,
    dem_path: Path,
    output_path: Path,
    target_crs: str = "EPSG:3035",
) -> Path:
    """Resample land cover to match DEM resolution and CRS.
    
    Args:
        landcover_path: Path to land cover GeoTIFF
        dem_path: Path to DEM GeoTIFF (reference for resolution/extent)
        output_path: Path to save resampled land cover
        target_crs: Target coordinate reference system
        
    Returns:
        Path to resampled land cover RGBA PNG
    """
    if not HAS_RASTERIO:
        print("Skipping resample - rasterio not available")
        return landcover_path
    
    print(f"Resampling land cover to match DEM...")
    
    # Read DEM to get target transform and dimensions
    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_width = dem_src.width
        dem_height = dem_src.height
        dem_bounds = dem_src.bounds
        print(f"  DEM: {dem_width}x{dem_height}, CRS: {dem_crs}")
    
    # Read and reproject land cover
    with rasterio.open(landcover_path) as lc_src:
        lc_crs = lc_src.crs
        print(f"  Land cover: {lc_src.width}x{lc_src.height}, CRS: {lc_crs}")
        
        # Calculate transform to match DEM exactly
        dst_transform, dst_width, dst_height = calculate_default_transform(
            lc_src.crs, target_crs,
            lc_src.width, lc_src.height,
            *lc_src.bounds,
            dst_width=dem_width,
            dst_height=dem_height,
        )
        
        # Use DEM transform directly for perfect alignment
        dst_transform = dem_transform
        dst_width = dem_width
        dst_height = dem_height
        
        # Reproject each band
        lc_data = lc_src.read()
        num_bands = lc_data.shape[0]
        
        resampled = np.zeros((num_bands, dst_height, dst_width), dtype=lc_data.dtype)
        
        for i in range(num_bands):
            reproject(
                source=lc_data[i],
                destination=resampled[i],
                src_transform=lc_src.transform,
                src_crs=lc_src.crs,
                dst_transform=dst_transform,
                dst_crs=dem_crs,  # Use DEM's CRS to match the transform
                resampling=Resampling.nearest,  # Nearest for categorical data
            )
        
        print(f"  Resampled to: {dst_width}x{dst_height}")
    
    # Convert to RGBA PNG for overlay
    # Land cover typically has class values - convert to colormap
    if num_bands == 1:
        # Single band - apply colormap
        lc_class = resampled[0]
        rgba = apply_landcover_colormap(lc_class)
    elif num_bands >= 3:
        # Already RGB(A) - data is float32 but values are in 0-255 range
        if num_bands == 3:
            rgba = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
            # Cast float to uint8 (values already in 0-255 range)
            rgb_float = np.moveaxis(resampled[:3], 0, -1)
            # Create mask for valid data: non-NaN AND at least one RGB channel > 50
            valid_mask = ~np.isnan(rgb_float).any(axis=2) & (rgb_float.max(axis=2) > 50.0)
            # Handle NaN values by replacing with 0
            rgb_float = np.nan_to_num(rgb_float, nan=0.0)
            rgba[..., :3] = np.clip(rgb_float, 0, 255).astype(np.uint8)
            # Set alpha=0 for no-data areas (transparent) so terrain shows through
            # Alpha=255 for valid land cover data
            rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)
        else:
            rgb_float = np.moveaxis(resampled[:4], 0, -1)
            # Create mask for valid data
            valid_mask = ~np.isnan(rgb_float[:, :, :3]).any(axis=2) & (rgb_float[:, :, :3].sum(axis=2) > 0.1)
            rgb_float = np.nan_to_num(rgb_float, nan=0.0)
            rgba = np.clip(rgb_float, 0, 255).astype(np.uint8)
            # Override alpha channel with mask
            rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected band count: {num_bands}")
    
    # Save as PNG
    if HAS_PIL:
        img = Image.fromarray(rgba)
        img.save(output_path)
        print(f"  Saved: {output_path}")
    else:
        # Fallback: save raw RGBA
        np.save(output_path.with_suffix('.npy'), rgba)
        print(f"  Saved: {output_path.with_suffix('.npy')}")
    
    return output_path


def apply_landcover_colormap(lc_class: np.ndarray) -> np.ndarray:
    """Apply a colormap to land cover class values.
    
    Uses Corine-style land cover colors:
    - 0: NoData (transparent)
    - 1xx: Artificial surfaces (red/pink)
    - 2xx: Agricultural areas (yellow/green)
    - 3xx: Forest/semi-natural (green)
    - 4xx: Wetlands (cyan)
    - 5xx: Water bodies (blue)
    """
    height, width = lc_class.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Default colormap for common land cover classes
    colormap = {
        0: (0, 0, 0, 0),          # NoData - transparent
        1: (230, 0, 77, 200),     # Urban fabric
        2: (255, 0, 0, 200),      # Industrial/commercial
        3: (204, 77, 242, 200),   # Transport
        4: (166, 0, 204, 200),    # Mine/dump/construction
        5: (255, 77, 255, 200),   # Artificial green
        6: (255, 166, 255, 200),  # Sport/leisure
        10: (255, 255, 168, 200), # Arable land
        11: (255, 255, 0, 200),   # Permanent crops
        12: (230, 230, 0, 200),   # Pastures
        13: (230, 128, 0, 200),   # Mixed agriculture
        20: (0, 166, 0, 200),     # Broad-leaved forest
        21: (77, 255, 0, 200),    # Coniferous forest
        22: (166, 255, 128, 200), # Mixed forest
        23: (166, 230, 77, 200),  # Natural grassland
        24: (255, 230, 166, 200), # Moors/heathland
        25: (166, 242, 0, 200),   # Sclerophyllous vegetation
        26: (128, 255, 0, 200),   # Transitional woodland
        27: (204, 204, 204, 200), # Bare rock
        28: (204, 255, 204, 200), # Sparse vegetation
        29: (166, 166, 255, 200), # Glaciers/perpetual snow
        30: (166, 230, 204, 200), # Inland marshes
        31: (0, 204, 242, 200),   # Peat bogs
        32: (128, 242, 230, 200), # Salt marshes
        33: (166, 166, 230, 200), # Intertidal flats
        40: (0, 204, 242, 200),   # Water courses
        41: (128, 242, 230, 200), # Water bodies
        42: (0, 255, 166, 200),   # Coastal lagoons
        43: (230, 242, 255, 200), # Estuaries
        44: (0, 128, 255, 200),   # Sea/ocean
    }
    
    # Apply colormap
    for class_val, color in colormap.items():
        mask = lc_class == class_val
        rgba[mask] = color
    
    # Handle values not in colormap (use modulo for broader coverage)
    unassigned = ~np.isin(lc_class, list(colormap.keys()))
    if np.any(unassigned):
        # Generate colors for unassigned classes
        unique_vals = np.unique(lc_class[unassigned])
        for val in unique_vals:
            if val == 0:
                continue
            # Generate pseudo-random color based on class value
            np.random.seed(int(val))
            r, g, b = np.random.randint(50, 230, 3)
            mask = lc_class == val
            rgba[mask] = (r, g, b, 180)
    
    return rgba


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Swiss terrain viewer with land cover overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dem", type=Path, default=None,
                        help="Path to DEM GeoTIFF (default: assets/tif/switzerland_dem.tif)")
    parser.add_argument("--landcover", type=Path, default=None,
                        help="Path to land cover GeoTIFF (default: assets/tif/switzerland_land_cover.tif)")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")
    parser.add_argument("--crs", type=str, default="EPSG:3035",
                        help="Target CRS for reprojection (default: EPSG:3035)")
    
    # High-quality presets
    parser.add_argument("--preset", choices=["hq1", "hq2", "hq3", "hq4"],
                        help="High-quality rendering preset (see --help for details)")
    
    # Overlay options
    overlay_group = parser.add_argument_group("Overlay Options", "Land cover overlay settings")
    overlay_group.add_argument("--overlay-opacity", type=float, default=1,
                               help="Overlay opacity 0.0-1.0 (default: 1)")
    overlay_group.add_argument("--no-overlay", action="store_true",
                               help="Disable land cover overlay (DEM only)")
    overlay_group.add_argument("--no-solid", action="store_true",
                               help="Hide base surface where overlay alpha=0 (like rayshader solid=FALSE)")
    overlay_group.add_argument("--legend", action="store_true", default=True,
                               help="Add land cover legend to snapshots (default: on)")
    overlay_group.add_argument("--no-legend", action="store_false", dest="legend",
                               help="Disable land cover legend on snapshots")
    
    # Background options
    bg_group = parser.add_argument_group("Background", "Background color settings")
    bg_group.add_argument("--background", type=str, default="#FFFFFF",
                          help="Background color as hex (e.g. #FFFFFF) or gradient as hex1,hex2 (default: #FFFFFF white)")
    
    # PBR rendering options (inherited from template)
    pbr_group = parser.add_argument_group("PBR Rendering", "High-quality terrain rendering options")
    pbr_group.add_argument("--exposure", type=float, default=1.0,
                           help="ACES exposure multiplier (default: 1.0)")
    pbr_group.add_argument("--msaa", type=int, choices=[1, 4, 8], default=1,
                           help="MSAA samples (default: 1)")
    pbr_group.add_argument("--shadows", choices=["none", "hard", "pcf", "pcss"], default="pcss",
                           help="Shadow technique (default: pcss)")
    
    # Sun/lighting options
    sun_group = parser.add_argument_group("Sun Lighting", "Directional sun light parameters")
    sun_group.add_argument("--sun-azimuth", type=float, default=135.0,
                           help="Sun azimuth angle in degrees (default: 135.0)")
    sun_group.add_argument("--sun-elevation", type=float, default=35.0,
                           help="Sun elevation angle in degrees (default: 35.0)")
    
    args = parser.parse_args()
    
    # Get default paths if not specified
    default_dem, default_landcover = get_default_paths()
    dem_path = args.dem if args.dem else default_dem
    landcover_path = args.landcover if args.landcover else default_landcover
    
    # Validate paths
    if not dem_path.exists():
        print(f"Error: DEM file not found: {dem_path}")
        return 1
    if not args.no_overlay and not landcover_path.exists():
        print(f"Error: Land cover file not found: {landcover_path}")
        return 1
    
    # Apply high-quality presets
    height_ao = False
    sun_vis = False
    snow = False
    rock = False
    dof = False
    lens_effects = False
    temperature = 6500.0
    
    if args.preset:
        if args.preset == "hq1":
            # Standard high quality
            args.width = 3840
            args.height = 2160
            args.msaa = 8
            height_ao = True
            sun_vis = True
            args.exposure = 1.2
            print("Preset: hq1 - Standard high quality (4K, MSAA 8x, AO, sun visibility)")
        elif args.preset == "hq2":
            # Alpine preset
            args.width = 3840
            args.height = 2160
            args.msaa = 4
            height_ao = True
            sun_vis = True
            snow = True
            rock = True
            temperature = 7000.0  # Cool for alpine
            args.exposure = 1.3
            print("Preset: hq2 - Alpine (4K, snow/rock layers, cool temperature)")
        elif args.preset == "hq3":
            # Cinematic
            args.width = 3840
            args.height = 2160
            args.msaa = 4
            height_ao = True
            dof = True
            lens_effects = True
            temperature = 5500.0  # Warm golden hour
            args.exposure = 1.4
            print("Preset: hq3 - Cinematic (4K, DoF, lens effects, warm tones)")
        elif args.preset == "hq4":
            # Maximum quality - all effects
            args.width = 3840
            args.height = 2160
            args.msaa = 8
            height_ao = True
            sun_vis = True
            snow = True
            rock = True
            dof = True
            lens_effects = True
            temperature = 6500.0
            args.exposure = 1.2
            print("Preset: hq4 - Maximum quality (4K, all effects enabled)")
    
    # Resample land cover to match DEM
    overlay_png_path = None
    if not args.no_overlay and HAS_RASTERIO:
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay_png_path = Path(tmpdir) / "landcover_resampled.png"
            try:
                overlay_png_path = resample_landcover_to_dem(
                    landcover_path, dem_path, overlay_png_path, args.crs
                )
                # Copy to persistent location for viewer
                persistent_path = dem_path.parent / "switzerland_landcover_overlay.png"
                if HAS_PIL:
                    import shutil
                    shutil.copy(overlay_png_path, persistent_path)
                    overlay_png_path = persistent_path
                    print(f"Overlay saved to: {overlay_png_path}")
            except Exception as e:
                print(f"Warning: Failed to resample land cover: {e}")
                overlay_png_path = None
    elif not args.no_overlay:
        # Try to use pre-existing overlay if available
        persistent_path = dem_path.parent / "switzerland_landcover_overlay.png"
        if persistent_path.exists():
            overlay_png_path = persistent_path
            print(f"Using existing overlay: {overlay_png_path}")
    
    # Start viewer
    binary = find_viewer_binary()
    cmd = [binary, "--ipc-port", "0", "--size", f"{args.width}x{args.height}"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    # Wait for READY
    ready_pattern = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
    port = None
    start = time.time()
    
    while time.time() - start < 30.0:
        if process.poll() is not None:
            print("Viewer exited unexpectedly")
            return 1
        line = process.stdout.readline()
        if line:
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                break
    
    if port is None:
        print("Timeout waiting for viewer")
        process.terminate()
        return 1
    
    # Connect and load terrain
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    
    print(f"Loading terrain: {dem_path}")
    resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
    if not resp.get("ok"):
        print(f"Failed to load terrain: {resp.get('error')}")
        sock.close()
        process.terminate()
        return 1
    
    # Parse background color(s)
    bg_colors = args.background.split(",")
    if len(bg_colors) == 1:
        # Single color
        bg_rgb = hex_to_rgb(bg_colors[0].strip())
        background = [bg_rgb[0] / 255.0, bg_rgb[1] / 255.0, bg_rgb[2] / 255.0]
    else:
        # Gradient: top,bottom colors
        bg_top = hex_to_rgb(bg_colors[0].strip())
        bg_bot = hex_to_rgb(bg_colors[1].strip())
        background = [
            bg_top[0] / 255.0, bg_top[1] / 255.0, bg_top[2] / 255.0,
            bg_bot[0] / 255.0, bg_bot[1] / 255.0, bg_bot[2] / 255.0,
        ]
    
    # Set initial camera and terrain params
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 45.0, "theta": 40.0, "radius": 4500.0, "fov": 35.0,
        "zscale": 0.05,  # Good for Swiss Alps
        "sun_azimuth": args.sun_azimuth,
        "sun_elevation": args.sun_elevation,
        "sun_intensity": 1.2,
        "background": background,
    })
    
    # Configure PBR rendering
    pbr_cmd = {
        "cmd": "set_terrain_pbr",
        "enabled": True,
        "shadow_technique": args.shadows,
        "shadow_map_res": 4096,
        "exposure": args.exposure,
        "msaa": args.msaa,
        "ibl_intensity": 0.8,
        "normal_strength": 1.2,
    }
    
    # Heightfield AO
    if height_ao:
        pbr_cmd["height_ao"] = {
            "enabled": True,
            "directions": 8,
            "steps": 24,
            "max_distance": 300.0,
            "strength": 1.0,
            "resolution_scale": 1.0,
        }
    
    # Sun visibility
    if sun_vis:
        pbr_cmd["sun_visibility"] = {
            "enabled": True,
            "mode": "soft",
            "samples": 6,
            "steps": 32,
            "max_distance": 500.0,
            "softness": 1.2,
            "bias": 0.01,
            "resolution_scale": 1.0,
        }
    
    # Material layers
    if snow or rock:
        pbr_cmd["materials"] = {
            "snow_enabled": snow,
            "snow_altitude_min": 2200.0,  # Swiss Alps snow line
            "snow_altitude_blend": 300.0,
            "snow_slope_max": 50.0,
            "rock_enabled": rock,
            "rock_slope_min": 40.0,
            "wetness_enabled": False,
        }
    
    # Tonemap with temperature
    pbr_cmd["tonemap"] = {
        "operator": "aces",
        "white_point": 4.0,
        "white_balance_enabled": True,
        "temperature": temperature,
        "tint": 0.0,
    }
    
    # Depth of field
    if dof:
        pbr_cmd["dof"] = {
            "enabled": True,
            "f_stop": 8.0,
            "focus_distance": 3000.0,
            "focal_length": 85.0,
            "tilt_pitch": 0.0,
            "tilt_yaw": 0.0,
            "quality": "high",
        }
    
    # Lens effects
    if lens_effects:
        pbr_cmd["lens_effects"] = {
            "enabled": True,
            "distortion": 0.0,
            "chromatic_aberration": 0.002,
            "vignette_strength": 0.25,
            "vignette_radius": 0.75,
            "vignette_softness": 0.4,
        }
    
    resp = send_ipc(sock, pbr_cmd)
    if not resp.get("ok"):
        print(f"Warning: PBR config failed: {resp.get('error')}")
    else:
        print("PBR rendering enabled")
    
    # Load overlay if available
    if overlay_png_path and overlay_png_path.exists():
        print(f"Loading land cover overlay: {overlay_png_path}")
        resp = send_ipc(sock, {
            "cmd": "load_overlay",
            "name": "landcover",
            "path": str(overlay_png_path),
            "extent": [0.0, 0.0, 1.0, 1.0],  # Full coverage
            "opacity": args.overlay_opacity,
            "z_order": 0,
        })
        if resp.get("ok"):
            print(f"Overlay loaded (opacity={args.overlay_opacity})")
            # Enable overlay system
            send_ipc(sock, {"cmd": "set_overlays_enabled", "enabled": True})
            # Set solid surface mode (default True, --no-solid sets to False)
            solid = not args.no_solid
            send_ipc(sock, {"cmd": "set_overlay_solid", "solid": solid})
            if not solid:
                print("Solid surface disabled (areas with overlay alpha=0 will be hidden)")
        else:
            print(f"Warning: Overlay load failed: {resp.get('error')}")
    
    # Snapshot mode
    if args.snapshot:
        time.sleep(1.0)  # Wait for render to stabilize
        resp = send_ipc(sock, {
            "cmd": "snapshot",
            "path": str(args.snapshot.resolve()),
            "width": args.width,
            "height": args.height,
        })
        send_ipc(sock, {"cmd": "close"})
        sock.close()
        process.wait()
        
        if args.snapshot.exists():
            # Add land cover legend to the image if enabled
            if args.legend and not args.no_overlay and HAS_PIL:
                add_legend_to_image(args.snapshot, legend_scale=0.35, position="southeast",
                                    max_legend_height=800, transparent_background=True)
                print(f"Saved with legend: {args.snapshot}")
            else:
                print(f"Saved: {args.snapshot}")
            return 0
        return 1
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("SWISS TERRAIN VIEWER WITH LAND COVER OVERLAY")
    print("=" * 60)
    print("Window controls:")
    print("  Mouse drag     - Orbit camera")
    print("  Scroll wheel   - Zoom in/out")
    print("  W/S or ↑/↓     - Tilt camera up/down")
    print("  A/D or ←/→     - Rotate camera left/right")
    print()
    print("Terminal commands:")
    print("  overlay on/off           - Toggle land cover overlay")
    print("  overlay opacity=0.5      - Set overlay opacity")
    print("  set phi=45 theta=60 radius=2000")
    print("  set sun_az=135 sun_el=45")
    print("  snap <path> [<width>x<height>]  - Take snapshot")
    print("  params                   - Show current parameters")
    print("  quit                     - Close viewer")
    print("=" * 60 + "\n")
    
    def parse_set_command(cmd_args: str) -> dict:
        """Parse 'set key=value key=value ...' into IPC params."""
        params = {"cmd": "set_terrain"}
        for pair in cmd_args.split():
            if "=" not in pair:
                continue
            key, val = pair.split("=", 1)
            key = key.lower().strip()
            val = val.strip()
            
            key_map = {
                "phi": "phi", "theta": "theta", "radius": "radius", "fov": "fov",
                "sun_az": "sun_azimuth", "sun_azimuth": "sun_azimuth",
                "sun_el": "sun_elevation", "sun_elevation": "sun_elevation",
                "intensity": "sun_intensity", "sun_intensity": "sun_intensity",
                "ambient": "ambient", "zscale": "zscale", "z_scale": "zscale",
                "shadow": "shadow", "bg": "background", "background": "background",
            }
            
            ipc_key = key_map.get(key, key)
            
            if "," in val:
                params[ipc_key] = [float(x) for x in val.split(",")]
            else:
                params[ipc_key] = float(val)
        
        return params
    
    try:
        while process.poll() is None:
            try:
                cmd_str = input("> ").strip()
            except EOFError:
                break
            
            if not cmd_str:
                continue
            
            parts = cmd_str.split(maxsplit=1)
            name = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""
            
            if name in ("quit", "exit", "q"):
                send_ipc(sock, {"cmd": "close"})
                break
            elif name == "overlay":
                if cmd_args.lower() in ("on", "true", "1"):
                    send_ipc(sock, {"cmd": "set_overlays_enabled", "enabled": True})
                    print("Overlay enabled")
                elif cmd_args.lower() in ("off", "false", "0"):
                    send_ipc(sock, {"cmd": "set_overlays_enabled", "enabled": False})
                    print("Overlay disabled")
                elif "opacity=" in cmd_args.lower():
                    opacity = float(cmd_args.split("=")[1])
                    send_ipc(sock, {"cmd": "set_global_overlay_opacity", "opacity": opacity})
                    print(f"Overlay opacity set to {opacity}")
                else:
                    print("Usage: overlay on|off|opacity=0.5")
            elif name == "set":
                if cmd_args:
                    params = parse_set_command(cmd_args)
                    resp = send_ipc(sock, params)
                    if not resp.get("ok"):
                        print(f"Error: {resp.get('error')}")
                else:
                    print("Usage: set key=value [key=value ...]")
            elif name == "params":
                send_ipc(sock, {"cmd": "get_terrain_params"})
            elif name == "snap":
                if cmd_args:
                    snap_parts = cmd_args.split()
                    path = str(Path(snap_parts[0]).resolve())
                    snap_cmd = {"cmd": "snapshot", "path": path}
                    if len(snap_parts) == 2:
                        m = re.match(r"^(\d+)x(\d+)$", snap_parts[1].lower())
                        if m:
                            snap_cmd["width"] = int(m.group(1))
                            snap_cmd["height"] = int(m.group(2))
                    resp = send_ipc(sock, snap_cmd)
                    if resp.get("ok"):
                        snap_path = Path(path)
                        # Add legend if enabled and overlay is active
                        if args.legend and not args.no_overlay and HAS_PIL and snap_path.exists():
                            # Wait for file to be fully written and retry if needed
                            legend_added = False
                            for attempt in range(5):
                                time.sleep(0.5)  # Wait for file to be ready
                                try:
                                    add_legend_to_image(snap_path, legend_scale=0.35, position="southeast",
                                                        max_legend_height=800, transparent_background=True)
                                    legend_added = True
                                    break
                                except Exception as e:
                                    if attempt < 4:
                                        continue
                                    print(f"Warning: Could not add legend: {e}")
                            if legend_added:
                                print(f"Saved with legend: {path}")
                            else:
                                print(f"Saved (legend failed): {path}")
                        else:
                            print(f"Saved: {path}")
                else:
                    print("Usage: snap <path> [<width>x<height>]")
            else:
                print("Unknown command. Type 'overlay', 'set', 'params', 'snap', or 'quit'")
    except KeyboardInterrupt:
        pass
    
    sock.close()
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
