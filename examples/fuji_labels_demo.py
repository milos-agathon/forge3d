#!/usr/bin/env python3
"""Mount Fuji terrain with place name labels demo.

This example demonstrates the new label functionality by loading a
Mount Fuji terrain DEM and overlaying place name labels from
OpenStreetMap data stored in a GeoPackage file.

The labels are rendered in screen-space with world-position anchoring.

Usage:
    # Interactive viewing
    python examples/fuji_labels_demo.py
    
    # Take snapshot and exit
    python examples/fuji_labels_demo.py --snapshot fuji_labels.png

Requirements:
    - assets/tif/Mount_Fuji_30m.tif (Terrain DEM)
    - assets/gpkg/Mount_Fuji_places.gpkg (OSM place names)
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from forge3d.viewer_ipc import (
    find_viewer_binary,
    send_ipc,
    add_label,
    add_line_label,
    add_curved_label,
    add_callout,
    clear_labels,
    set_labels_enabled,
    set_label_zoom,
    set_max_visible_labels,
    set_label_typography,
    set_declutter_algorithm,
    load_label_atlas,
)
from forge3d.interactive import run_interactive_loop

# Asset paths
ASSETS_DIR = Path(__file__).parent.parent / "assets"
DEM_PATH = ASSETS_DIR / "tif" / "Mount_Fuji_30m.tif"
PLACES_PATH = ASSETS_DIR / "gpkg" / "Mount_Fuji_places.gpkg"
FONT_ATLAS_PNG = ASSETS_DIR / "fonts" / "default_atlas.png"
FONT_ATLAS_JSON = ASSETS_DIR / "fonts" / "default_atlas.json"

# Terrain extent in WGS84 coordinates (from gdalinfo)
# Upper Left  ( 138.6278010,  35.4605292)
# Lower Right ( 138.8277887,  35.2605414)
TERRAIN_MIN_X = 138.6278010
TERRAIN_MAX_X = 138.8277887
TERRAIN_MIN_Y = 35.2605414
TERRAIN_MAX_Y = 35.4605292
TERRAIN_WIDTH_PX = 2594
TERRAIN_HEIGHT_PX = 2594


def world_to_terrain_local(world_x: float, world_y: float) -> tuple[float, float, float]:
    """Transform WGS84 coordinates to terrain-local coordinates.
    
    The terrain renderer normalizes coordinates to a local system where:
    - Origin is at (0, 0)
    - X spans 0 to terrain_width (max dimension)
    - Y spans 0 to terrain_height
    
    Returns (local_x, local_y, normalized_z) where z is a placeholder for height.
    """
    terrain_width = max(TERRAIN_WIDTH_PX, TERRAIN_HEIGHT_PX)
    
    # Normalize X (Longitude)
    # Min X (West) -> 0
    # Max X (East) -> 1
    norm_x = (world_x - TERRAIN_MIN_X) / (TERRAIN_MAX_X - TERRAIN_MIN_X)
    
    # Normalize Y (Latitude)
    # Texture row 0 is North (Max Y). Row H is South (Min Y).
    # We want North -> 0 (local_z=0), South -> 1 (local_z=max)
    # So we invert the latitude mapping.
    norm_y = (TERRAIN_MAX_Y - world_y) / (TERRAIN_MAX_Y - TERRAIN_MIN_Y)
    
    local_x = norm_x * terrain_width
    local_y = norm_y * terrain_width * (TERRAIN_HEIGHT_PX / TERRAIN_WIDTH_PX)
    
    # Height will be added based on terrain elevation + offset
    return local_x, local_y, 0.0


def parse_gpkg_point(blob: bytes) -> tuple[float, float] | tuple[None, None]:
    """Parse GeoPackage point geometry (WKB with GPKG header)."""
    import struct
    if not blob or len(blob) < 21:
        return None, None
    # Search for WKB point structure within the blob
    for i in range(len(blob) - 20):
        if blob[i] in (0, 1):
            byte_order = blob[i]
            if byte_order == 1:  # Little endian
                geom_type = struct.unpack_from('<I', blob, i+1)[0]
                if geom_type == 1:  # Point
                    x, y = struct.unpack_from('<dd', blob, i+5)
                    return x, y
            elif byte_order == 0:  # Big endian
                geom_type = struct.unpack_from('>I', blob, i+1)[0]
                if geom_type == 1:
                    x, y = struct.unpack_from('>dd', blob, i+5)
                    return x, y
    return None, None


def load_places(gpkg_path: Path) -> list[dict]:
    """Load place names and coordinates from GeoPackage using sqlite3.
    
    Returns list of dicts with 'name', 'x', 'y' keys.
    Coordinates are in the same CRS as the terrain (WGS84).
    """
    import sqlite3
    
    places = []
    try:
        conn = sqlite3.connect(str(gpkg_path))
        cur = conn.cursor()
        # Note: Table name typically matches the file name/layer name. 
        # For Mount_Fuji_places.gpkg, likely 'Mount_Fuji_places' or 'places' depending on how it was written.
        # But commonly 'Mount_Fuji_places' if written by sf.
        # Let's try to find the table name first.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'sqlite_%'")
        tables = cur.fetchall()
        table_name = "Mount_Fuji_places" # Default guess
        if tables:
             # Look for a likely candidate
             for t in tables:
                 if "places" in t[0] or "Mount_Fuji" in t[0]:
                     table_name = t[0]
                     break
        
        # Try to select name.en (R/sf export style) or name:en (OSM style)
        # We need to quote column names with special characters
        cols_to_check = ["name.en", "name:en"]
        en_col = None
        
        # Check table info to see which column exists
        cur.execute(f"PRAGMA table_info('{table_name}')")
        cols = [info[1] for info in cur.fetchall()]
        
        for c in cols_to_check:
            if c in cols:
                en_col = c
                break
        
        if en_col:
            cur.execute(f'SELECT name, "{en_col}", ele, geom FROM "{table_name}" WHERE name IS NOT NULL')
        else:
            # Fallback if english name doesn't exist
            cur.execute(f'SELECT name, name, ele, geom FROM "{table_name}" WHERE name IS NOT NULL')
            
        for name, name_en, ele_str, geom in cur.fetchall():
            x, y = parse_gpkg_point(geom)
            if x is not None and y is not None:
                # Prefer English name, fallback to local name
                # Also ensure we only use ASCII characters since our atlas is limited
                display_name = name_en if name_en else name
                
                # Parse elevation
                try:
                    ele = float(ele_str) if ele_str else 0.0
                except ValueError:
                    ele = 0.0
                
                # Check if display_name is ASCII
                if not all(ord(c) < 128 for c in display_name):
                    # If local name is not ASCII and we don't have English, skip or try to find ASCII in name
                    # (Sometimes name contains "Local (English)")
                    continue
                    
                places.append({"name": display_name, "x": x, "y": y, "ele": ele})
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to read places from {gpkg_path}: {e}")
        # Fallback hardcoded places (approx coords)
        places = [
            {"name": "Mount Fuji", "x": 138.7278, "y": 35.3606, "ele": 3776.0},
        ]
    
    return places


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mount Fuji terrain with place name labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")
    parser.add_argument("--dem", type=Path, default=DEM_PATH,
                        help="Path to terrain DEM")
    parser.add_argument("--places", type=Path, default=PLACES_PATH,
                        help="Path to places GeoPackage")
    parser.add_argument("--preset", choices=["default", "alpine", "cinematic", "high_quality"],
                        help="Rendering preset (overrides individual settings)")
    
    # PBR+POM rendering options
    pbr_group = parser.add_argument_group("PBR Rendering", "High-quality terrain rendering options")
    pbr_group.add_argument("--pbr", action="store_true",
                           help="Enable PBR+POM rendering mode (default: legacy simple shader)")
    pbr_group.add_argument("--hdr", type=Path,
                           help="HDR environment map for IBL lighting")
    pbr_group.add_argument("--shadows", choices=["none", "hard", "pcf", "pcss"], default="pcss",
                           help="Shadow technique (default: pcss)")
    pbr_group.add_argument("--shadow-map-res", type=int, default=2048,
                           help="Shadow map resolution (default: 2048)")
    pbr_group.add_argument("--exposure", type=float, default=1.0,
                           help="ACES exposure multiplier (default: 1.0)")
    pbr_group.add_argument("--msaa", type=int, choices=[1, 4, 8], default=1,
                           help="MSAA samples (default: 1)")
    pbr_group.add_argument("--ibl-intensity", type=float, default=1.0,
                           help="IBL intensity multiplier (default: 1.0)")
    pbr_group.add_argument("--normal-strength", type=float, default=1.0,
                           help="Terrain normal strength (default: 1.0)")
    # P0.1/M1: OIT
    pbr_group.add_argument("--oit", type=str, choices=["auto", "wboit", "dual_source", "off"],
                           default=None, help="OIT mode for transparent surfaces (default: off)")
    
    # Sun/lighting options
    sun_group = parser.add_argument_group("Sun Lighting", "Directional sun light parameters")
    sun_group.add_argument("--sun-azimuth", type=float, default=135.0,
                           help="Sun azimuth angle in degrees (default: 135.0)")
    sun_group.add_argument("--sun-elevation", type=float, default=35.0,
                           help="Sun elevation angle in degrees (default: 35.0)")
    sun_group.add_argument("--sun-intensity", type=float, default=1.0,
                           help="Sun light intensity (default: 1.0)")
    
    # Heightfield Ray AO options
    ao_group = parser.add_argument_group("Heightfield AO", "Terrain ambient occlusion from heightfield ray-tracing")
    ao_group.add_argument("--height-ao", action="store_true",
                          help="Enable heightfield ray-traced ambient occlusion")
    ao_group.add_argument("--height-ao-directions", type=int, default=6,
                          help="Number of ray directions around horizon [4-16] (default: 6)")
    ao_group.add_argument("--height-ao-steps", type=int, default=16,
                          help="Ray march steps per direction [8-64] (default: 16)")
    ao_group.add_argument("--height-ao-distance", type=float, default=200.0,
                          help="Maximum ray distance in world units (default: 200.0)")
    ao_group.add_argument("--height-ao-strength", type=float, default=1.0,
                          help="AO darkening intensity [0.0-2.0] (default: 1.0)")
    ao_group.add_argument("--height-ao-resolution", type=float, default=0.5,
                          help="AO texture resolution scale [0.1-1.0] (default: 0.5)")
    
    # Sun Visibility options
    sv_group = parser.add_argument_group("Sun Visibility", "Terrain self-shadowing from heightfield ray-tracing")
    sv_group.add_argument("--sun-vis", action="store_true",
                          help="Enable heightfield ray-traced sun visibility")
    sv_group.add_argument("--sun-vis-mode", choices=["hard", "soft"], default="soft",
                          help="Shadow mode: hard (binary) or soft (jittered) (default: soft)")
    sv_group.add_argument("--sun-vis-samples", type=int, default=4,
                          help="Number of jittered samples for soft shadows [1-16] (default: 4)")
    sv_group.add_argument("--sun-vis-steps", type=int, default=24,
                          help="Ray march steps toward sun [8-64] (default: 24)")
    sv_group.add_argument("--sun-vis-distance", type=float, default=400.0,
                          help="Maximum ray distance in world units (default: 400.0)")
    sv_group.add_argument("--sun-vis-softness", type=float, default=1.0,
                          help="Soft shadow penumbra size [0.0-4.0] (default: 1.0)")
    sv_group.add_argument("--sun-vis-bias", type=float, default=0.01,
                          help="Self-shadowing bias to reduce artifacts (default: 0.01)")
    sv_group.add_argument("--sun-vis-resolution", type=float, default=0.5,
                          help="Visibility texture resolution scale [0.1-1.0] (default: 0.5)")
    
    # M4: Material Layering options
    mat_group = parser.add_argument_group("Material Layering (M4)", "Terrain material layers: snow, rock, wetness")
    mat_group.add_argument("--snow", action="store_true",
                           help="Enable snow layer based on altitude and slope")
    mat_group.add_argument("--snow-altitude", type=float, default=2500.0,
                           help="Minimum altitude for snow in world units (default: 2500.0)")
    mat_group.add_argument("--snow-blend", type=float, default=200.0,
                           help="Altitude blend range for snow transition (default: 200.0)")
    mat_group.add_argument("--snow-slope", type=float, default=45.0,
                           help="Maximum slope for snow accumulation in degrees (default: 45.0)")
    mat_group.add_argument("--rock", action="store_true",
                           help="Enable exposed rock layer on steep slopes")
    mat_group.add_argument("--rock-slope", type=float, default=45.0,
                           help="Minimum slope for exposed rock in degrees (default: 45.0)")
    mat_group.add_argument("--wetness", action="store_true",
                           help="Enable wetness effect in low areas")
    mat_group.add_argument("--wetness-strength", type=float, default=0.3,
                           help="Wetness darkening strength [0.0-1.0] (default: 0.3)")
    
    # M5: Vector Overlay options
    vo_group = parser.add_argument_group("Vector Overlays (M5)", "Depth-correct vector overlays with halos")
    vo_group.add_argument("--overlay-depth", action="store_true",
                          help="Enable depth testing for vector overlays (occlude behind terrain)")
    vo_group.add_argument("--overlay-depth-bias", type=float, default=0.001,
                          help="Depth bias to prevent z-fighting (default: 0.001)")
    vo_group.add_argument("--overlay-halo", action="store_true",
                          help="Enable halo/outline around vector overlays")
    vo_group.add_argument("--overlay-halo-width", type=float, default=2.0,
                          help="Halo width in pixels (default: 2.0)")
    vo_group.add_argument("--overlay-halo-color", type=str, default="0,0,0,0.5",
                          help="Halo color as R,G,B,A (default: 0,0,0,0.5)")
    
    # M6: Tonemap options
    tm_group = parser.add_argument_group("Tonemap (M6)", "HDR tonemapping and color grading")
    tm_group.add_argument("--tonemap", choices=["reinhard", "reinhard_extended", "aces", "uncharted2", "exposure"],
                          default="aces", help="Tonemap operator (default: aces)")
    tm_group.add_argument("--tonemap-white-point", type=float, default=4.0,
                          help="White point for extended operators (default: 4.0)")
    tm_group.add_argument("--white-balance", action="store_true",
                          help="Enable white balance adjustment")
    tm_group.add_argument("--temperature", type=float, default=6500.0,
                          help="Color temperature in Kelvin [2000-12000] (default: 6500.0 = D65)")
    tm_group.add_argument("--tint", type=float, default=0.0,
                          help="Green-magenta tint [-1.0 to 1.0] (default: 0.0)")
    
    # M3: Depth of Field options
    dof_group = parser.add_argument_group("Depth of Field (M3)", "Camera DoF with tilt-shift support")
    dof_group.add_argument("--dof", action="store_true",
                           help="Enable depth of field effect")
    dof_group.add_argument("--dof-f-stop", type=float, default=5.6,
                           help="Aperture f-stop [1.0-22.0] (default: 5.6)")
    dof_group.add_argument("--dof-focus-distance", type=float, default=100.0,
                           help="Focus distance in world units (default: 100.0)")
    dof_group.add_argument("--dof-focal-length", type=float, default=50.0,
                           help="Lens focal length in mm (default: 50.0)")
    dof_group.add_argument("--dof-tilt-pitch", type=float, default=0.0,
                           help="Tilt-shift pitch angle in degrees (default: 0.0)")
    dof_group.add_argument("--dof-tilt-yaw", type=float, default=0.0,
                           help="Tilt-shift yaw angle in degrees (default: 0.0)")
    dof_group.add_argument("--dof-quality", choices=["low", "medium", "high", "ultra"],
                           default="medium", help="DoF quality preset (default: medium)")
    
    # M4: Motion Blur options
    mb_group = parser.add_argument_group("Motion Blur (M4)", "Camera shutter accumulation")
    mb_group.add_argument("--motion-blur", action="store_true",
                          help="Enable motion blur (camera shutter accumulation)")
    mb_group.add_argument("--mb-samples", type=int, default=8,
                          help="Number of sub-frames [1-64] (default: 8)")
    mb_group.add_argument("--mb-shutter-angle", type=float, default=180.0,
                          help="Shutter angle in degrees [0-360] (default: 180)")
    mb_group.add_argument("--mb-cam-phi-delta", type=float, default=0.0,
                          help="Camera azimuth change over shutter (default: 0.0)")
    mb_group.add_argument("--mb-cam-theta-delta", type=float, default=0.0,
                          help="Camera elevation change over shutter (default: 0.0)")
    mb_group.add_argument("--mb-cam-radius-delta", type=float, default=0.0,
                          help="Camera distance change over shutter (default: 0.0)")
    
    # M5: Lens Effects options
    le_group = parser.add_argument_group("Lens Effects (M5)", "Optical imperfections and sensor effects")
    le_group.add_argument("--lens-effects", action="store_true",
                          help="Enable lens effects (distortion, CA, vignette)")
    le_group.add_argument("--lens-distortion", type=float, default=0.0,
                          help="Barrel (+) / pincushion (-) distortion (default: 0.0)")
    le_group.add_argument("--lens-ca", type=float, default=0.0,
                          help="Chromatic aberration strength (default: 0.0)")
    le_group.add_argument("--lens-vignette", type=float, default=0.0,
                          help="Vignette strength [0.0-1.0] (default: 0.0)")
    le_group.add_argument("--lens-vignette-radius", type=float, default=0.7,
                          help="Vignette start radius [0.0-1.0] (default: 0.7)")
    le_group.add_argument("--lens-vignette-softness", type=float, default=0.3,
                          help="Vignette falloff softness (default: 0.3)")
    
    # M5: Denoise options
    dn_group = parser.add_argument_group("Denoise (M5)", "Noise reduction for rendered images")
    dn_group.add_argument("--denoise", action="store_true",
                          help="Enable CPU-based denoising")
    dn_group.add_argument("--denoise-method", choices=["atrous", "bilateral", "none"],
                          default="atrous", help="Denoise method (default: atrous)")
    dn_group.add_argument("--denoise-iterations", type=int, default=3,
                          help="Filter iterations [1-10] (default: 3)")
    dn_group.add_argument("--denoise-sigma-color", type=float, default=0.1,
                          help="Color similarity weight (default: 0.1)")
    
    # M6: Volumetrics options
    vol_group = parser.add_argument_group("Volumetrics (M6)", "Volumetric fog and light shafts")
    vol_group.add_argument("--volumetrics", action="store_true",
                           help="Enable volumetric fog")
    vol_group.add_argument("--vol-mode", choices=["uniform", "height", "exponential"],
                           default="uniform", help="Fog density mode (default: uniform)")
    vol_group.add_argument("--vol-density", type=float, default=0.01,
                           help="Global fog density (default: 0.01)")
    vol_group.add_argument("--vol-scattering", type=float, default=0.5,
                           help="In-scatter amount [0.0-1.0] (default: 0.5)")
    vol_group.add_argument("--vol-absorption", type=float, default=0.1,
                           help="Light absorption [0.0-1.0] (default: 0.1)")
    vol_group.add_argument("--vol-light-shafts", action="store_true",
                           help="Enable god rays / light shafts")
    vol_group.add_argument("--vol-shaft-intensity", type=float, default=1.0,
                           help="Light shaft brightness (default: 1.0)")
    vol_group.add_argument("--vol-half-res", action="store_true",
                           help="Render volumetrics at half resolution (faster)")
    
    # M6: Sky options
    sky_group = parser.add_argument_group("Sky (M6)", "Physically-based sky rendering")
    sky_group.add_argument("--sky", action="store_true",
                           help="Enable procedural sky")
    sky_group.add_argument("--sky-turbidity", type=float, default=2.0,
                           help="Atmospheric haziness [1.0-10.0] (default: 2.0)")
    sky_group.add_argument("--sky-ground-albedo", type=float, default=0.3,
                           help="Ground reflectance [0.0-1.0] (default: 0.3)")
    sky_group.add_argument("--sky-sun-intensity", type=float, default=1.0,
                           help="Sun disc brightness (default: 1.0)")
    sky_group.add_argument("--sky-aerial", action="store_true", default=True,
                           help="Enable aerial perspective (default: on)")
    sky_group.add_argument("--no-sky-aerial", action="store_false", dest="sky_aerial",
                           help="Disable aerial perspective")
    sky_group.add_argument("--sky-exposure", type=float, default=1.0,
                           help="Sky brightness adjustment (default: 1.0)")
    
    args = parser.parse_args()
    
    # Apply preset settings (overrides individual flags)
    if args.preset:
        if args.preset == "default":
            # All features disabled - use defaults
            pass
        elif args.preset == "alpine":
            # Alpine mountain scene with snow and rock layers
            args.pbr = True
            args.snow = True
            args.snow_altitude = 2500.0
            args.snow_blend = 300.0
            args.snow_slope = 50.0
            args.rock = True
            args.rock_slope = 40.0
            args.overlay_depth = True
            args.overlay_halo = True
            args.tonemap = "aces"
            args.white_balance = True
            args.temperature = 7000.0  # Slightly cool for snow
            print(f"Preset: alpine - Snow/rock layers, ACES tonemap, cool temperature")
        elif args.preset == "cinematic":
            # Cinematic warm-toned render with bloom
            args.pbr = True
            args.wetness = True
            args.wetness_strength = 0.4
            args.overlay_halo = True
            args.overlay_halo_width = 3.0
            args.tonemap = "uncharted2"
            args.tonemap_white_point = 6.0
            args.white_balance = True
            args.temperature = 5500.0  # Warm golden hour
            args.tint = 0.1
            print(f"Preset: cinematic - Warm tones, Uncharted2 tonemap")
        elif args.preset == "high_quality":
            # Maximum quality with all features
            args.pbr = True
            args.msaa = 8
            args.snow = True
            args.rock = True
            args.wetness = True
            args.overlay_depth = True
            args.overlay_halo = True
            args.height_ao = True
            args.sun_vis = True
            args.tonemap = "aces"
            args.white_balance = True
            print(f"Preset: high_quality - All features enabled, MSAA 8x")
    
    # Validate assets exist
    if not args.dem.exists():
        print(f"Error: DEM file not found: {args.dem}")
        return 1
    
    if not args.places.exists():
        print(f"Error: Places file not found: {args.places}")
        return 1
    
    # Load places
    places = load_places(args.places)
    print(f"Loaded {len(places)} places from {args.places.name}")
    for p in places[:10]:
        print(f"  - {p['name']} at ({p['x']:.4f}, {p['y']:.4f})")
    
    # Find and start viewer
    binary = find_viewer_binary()
    cmd = [binary, "--ipc-port", "0", "--size", f"{args.width}x{args.height}"]
    
    print(f"Starting viewer: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    # Wait for READY message
    ready_pattern = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
    port = None
    start = time.time()
    
    while time.time() - start < 30.0:
        if process.poll() is not None:
            print("Viewer exited unexpectedly")
            return 1
        line = process.stdout.readline()
        if line:
            print(f"  {line.rstrip()}")
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                break
    
    if port is None:
        print("Timeout waiting for viewer")
        process.terminate()
        return 1
    
    # Connect via IPC
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    print(f"Connected to viewer on port {port}")
    
    # Load terrain
    print(f"Loading terrain: {args.dem}")
    resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(args.dem.resolve())})
    if not resp.get("ok"):
        print(f"Failed to load terrain: {resp.get('error')}")
        sock.close()
        process.terminate()
        return 1
    
    # Terrain settings
    Z_SCALE = 0.25
    
    # Set initial camera and sun
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 0.0,           # Camera azimuth
        "theta": 25.0,        # Camera elevation
        "radius": 4000.0,     # Distance from center
        "fov": 35.0,          # Field of view
        "zscale": Z_SCALE,    # Height exaggeration
        "sun_azimuth": args.sun_azimuth,
        "sun_elevation": args.sun_elevation,
        "sun_intensity": args.sun_intensity,
    })
    
    # Configure PBR rendering if requested
    if args.pbr:
        pbr_cmd = {
            "cmd": "set_terrain_pbr",
            "enabled": True,
            "shadow_technique": args.shadows,
            "shadow_map_res": args.shadow_map_res,
            "exposure": args.exposure,
            "msaa": args.msaa,
            "ibl_intensity": args.ibl_intensity,
            "normal_strength": args.normal_strength,
        }
        if args.hdr:
            pbr_cmd["hdr_path"] = str(args.hdr.resolve())
        
        # Heightfield Ray AO settings
        if args.height_ao:
            pbr_cmd["height_ao"] = {
                "enabled": True,
                "directions": args.height_ao_directions,
                "steps": args.height_ao_steps,
                "max_distance": args.height_ao_distance,
                "strength": args.height_ao_strength,
                "resolution_scale": args.height_ao_resolution,
            }
        
        # Sun Visibility settings
        if args.sun_vis:
            pbr_cmd["sun_visibility"] = {
                "enabled": True,
                "mode": args.sun_vis_mode,
                "samples": args.sun_vis_samples,
                "steps": args.sun_vis_steps,
                "max_distance": args.sun_vis_distance,
                "softness": args.sun_vis_softness,
                "bias": args.sun_vis_bias,
                "resolution_scale": args.sun_vis_resolution,
            }
        
        # M4: Material Layering settings
        if args.snow or args.rock or args.wetness:
            pbr_cmd["materials"] = {
                "snow_enabled": args.snow,
                "snow_altitude_min": args.snow_altitude,
                "snow_altitude_blend": args.snow_blend,
                "snow_slope_max": args.snow_slope,
                "rock_enabled": args.rock,
                "rock_slope_min": args.rock_slope,
                "wetness_enabled": args.wetness,
                "wetness_strength": args.wetness_strength,
            }
        
        # M5: Vector Overlay settings
        if args.overlay_depth or args.overlay_halo:
            halo_color = [float(x) for x in args.overlay_halo_color.split(",")]
            pbr_cmd["vector_overlay"] = {
                "depth_test": args.overlay_depth,
                "depth_bias": args.overlay_depth_bias,
                "halo_enabled": args.overlay_halo,
                "halo_width": args.overlay_halo_width,
                "halo_color": halo_color,
            }
        
        # M6: Tonemap settings
        pbr_cmd["tonemap"] = {
            "operator": args.tonemap,
            "white_point": args.tonemap_white_point,
            "white_balance_enabled": args.white_balance,
            "temperature": args.temperature,
            "tint": args.tint,
        }
        
        # M3: Depth of Field settings
        if args.dof:
            pbr_cmd["dof"] = {
                "enabled": True,
                "f_stop": args.dof_f_stop,
                "focus_distance": args.dof_focus_distance,
                "focal_length": args.dof_focal_length,
                "tilt_pitch": args.dof_tilt_pitch,
                "tilt_yaw": args.dof_tilt_yaw,
                "quality": args.dof_quality,
            }
        
        # M4: Motion Blur settings
        if args.motion_blur:
            shutter_close = args.mb_shutter_angle / 360.0
            pbr_cmd["motion_blur"] = {
                "enabled": True,
                "samples": args.mb_samples,
                "shutter_open": 0.0,
                "shutter_close": shutter_close,
                "cam_phi_delta": args.mb_cam_phi_delta,
                "cam_theta_delta": args.mb_cam_theta_delta,
                "cam_radius_delta": args.mb_cam_radius_delta,
            }
        
        # M5: Lens Effects settings
        if args.lens_effects or args.lens_distortion != 0.0 or args.lens_ca != 0.0 or args.lens_vignette > 0.0:
            pbr_cmd["lens_effects"] = {
                "enabled": True,
                "distortion": args.lens_distortion,
                "chromatic_aberration": args.lens_ca,
                "vignette_strength": args.lens_vignette,
                "vignette_radius": args.lens_vignette_radius,
                "vignette_softness": args.lens_vignette_softness,
            }
        
        # M5: Denoise settings
        if args.denoise:
            pbr_cmd["denoise"] = {
                "enabled": True,
                "method": args.denoise_method,
                "iterations": args.denoise_iterations,
                "sigma_color": args.denoise_sigma_color,
            }
        
        # M6: Volumetrics settings
        if args.volumetrics:
            pbr_cmd["volumetrics"] = {
                "enabled": True,
                "mode": args.vol_mode,
                "density": args.vol_density,
                "scattering": args.vol_scattering,
                "absorption": args.vol_absorption,
                "light_shafts": args.vol_light_shafts,
                "shaft_intensity": args.vol_shaft_intensity,
                "half_res": args.vol_half_res,
            }
        
        # M6: Sky settings
        if args.sky:
            pbr_cmd["sky"] = {
                "enabled": True,
                "turbidity": args.sky_turbidity,
                "ground_albedo": args.sky_ground_albedo,
                "sun_intensity": args.sky_sun_intensity,
                "aerial_perspective": args.sky_aerial,
                "sky_exposure": args.sky_exposure,
            }
        
        resp = send_ipc(sock, pbr_cmd)
        if not resp.get("ok"):
            print(f"Warning: PBR config failed: {resp.get('error')}")
    
    # P0.1/M1: Enable OIT if requested
    if args.oit and args.oit != "off":
        resp = send_ipc(sock, {
            "cmd": "set_oit_enabled",
            "enabled": True,
            "mode": args.oit,
        })
        if resp.get("ok"):
            print(f"OIT enabled: mode={args.oit}")
        else:
            print(f"Warning: OIT config failed: {resp.get('error')}")
    
    # Wait for terrain to load and stabilize
    time.sleep(1.0)
    
    # Load font atlas for label rendering
    print("\nLoading font atlas...")
    if FONT_ATLAS_PNG.exists() and FONT_ATLAS_JSON.exists():
        resp = load_label_atlas(sock, str(FONT_ATLAS_PNG), str(FONT_ATLAS_JSON))
        if resp.get("ok"):
            print(f"  Font atlas loaded: {FONT_ATLAS_PNG.name}")
        else:
            print(f"  Warning: Failed to load font atlas: {resp.get('error')}")
        # Wait for atlas to be processed
        time.sleep(0.5)
    else:
        print(f"  Warning: Font atlas not found at {FONT_ATLAS_PNG}")
        print("  Labels will not render visible text without a font atlas.")
    
    # Open DEM to sample elevation
    import rasterio
    print(f"\nSampling elevation from {args.dem.name}...")
    try:
        dem_src = rasterio.open(args.dem)
    except Exception as e:
        print(f"Warning: Failed to open DEM with rasterio: {e}")
        dem_src = None

    print("\nAdding labels with Plan 2 features...")
    set_labels_enabled(sock, True)
    time.sleep(0.2)
    
    # Configure label system (Plan 2 features)
    set_label_zoom(sock, 1.0)  # Default zoom level
    set_max_visible_labels(sock, 100)  # Limit visible labels
    
    for i, place in enumerate(places):
        # Transform WGS84 coordinates to terrain-local coordinates
        local_x, local_z, _ = world_to_terrain_local(place["x"], place["y"])
        
        # Get elevation from DEM if available, else fallback to GPKG or 0
        elevation = place.get("ele", 0.0)
        
        if dem_src:
            try:
                # Sample elevation at (lon, lat)
                # rasterio.sample expects list of (x, y)
                vals = list(dem_src.sample([(place["x"], place["y"])]))
                if vals and len(vals) > 0:
                    val = vals[0][0]
                    # Check for nodata (often very negative or specific value)
                    if val > -10000:  # Simple sanity check
                        elevation = float(val)
            except Exception:
                pass
        
        # Place labels at correct elevation
        # Viewer visual height = elevation * zscale
        # Add a small offset (e.g. 50m scaled) so it doesn't clip into steep terrain
        local_y = (elevation * Z_SCALE) + (50.0 * Z_SCALE)
        
        # Assign priority based on elevation (higher places = higher priority)
        # This demonstrates Plan 2's priority-based collision resolution
        priority = int(elevation / 100) + 10
        
        # Demonstrate Plan 2 features: offset labels with leader lines for some places
        use_offset = (i % 3 == 0) and elevation > 1000  # Every 3rd high-altitude label
        offset_x = 30.0 if use_offset else 0.0
        offset_y = -20.0 if use_offset else 0.0
        
        # Scale-dependent visibility: show high-altitude labels at all zooms,
        # lower labels only when zoomed in
        min_zoom = 0.0 if elevation > 2000 else 0.5
        max_zoom = 10.0  # Always visible up to max zoom
        
        resp = add_label(
            sock,
            text=place["name"],
            world_pos=(local_x, local_y, local_z),
            size=20.0,
            color=(0.1, 0.1, 0.1, 1.0),  # Dark text
            halo_color=(1.0, 1.0, 1.0, 0.9),  # White halo
            halo_width=2.0,
            priority=priority,
            # Plan 2 features:
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            offset=(offset_x, offset_y) if use_offset else None,
            leader=use_offset,  # Show leader line when offset
            horizon_fade_angle=10.0,  # Fade labels near horizon
        )
        offset_info = f" [offset + leader]" if use_offset else ""
        print(f"  Added label: {place['name']} pri={priority} zoom=[{min_zoom:.1f},{max_zoom:.1f}]{offset_info}")
    
    if dem_src:
        dem_src.close()
    
    # === Plan 3 Features Demo ===
    print("\n--- Plan 3: Premium Label Features ---")
    
    # Set typography with tracking (letter-spacing)
    set_label_typography(sock, tracking=0.02, kerning=True, line_height=1.2)
    print("  Typography: tracking=0.02, kerning=True, line_height=1.2")
    
    # Set declutter algorithm to greedy for fast placement
    set_declutter_algorithm(sock, algorithm="greedy", seed=42)
    print("  Declutter: algorithm=greedy, seed=42")
    
    # 1. Add a callout for Mount Fuji summit (main peak)
    summit_local_x, summit_local_z, _ = world_to_terrain_local(138.7278, 35.3606)
    summit_local_y = 3776.0 * Z_SCALE + 100.0 * Z_SCALE
    
    add_callout(
        sock,
        text="Mt. Fuji Summit\n3776m\nHighest Peak",
        anchor=(summit_local_x, summit_local_y, summit_local_z),
        offset=(0.0, -70.0),
        background_color=(1.0, 1.0, 0.95, 0.95),
        border_color=(0.8, 0.5, 0.0, 1.0),
        border_width=2.0,
        corner_radius=6.0,
        padding=10.0,
        text_size=18.0,
        text_color=(0.0, 0.0, 0.0, 1.0),
    )
    print("  Added callout: Mt. Fuji Summit")
    
    # 2. Add special styled labels for different landmark types
    print("\n--- Advanced Label Styling Examples ---")
    
    # Example: Large underlined label for a major landmark
    major_landmark_x, major_landmark_z, _ = world_to_terrain_local(138.73, 35.35)
    major_landmark_y = 2500.0 * Z_SCALE + 80.0 * Z_SCALE
    add_label(
        sock,
        text="FIFTH STATION",
        world_pos=(major_landmark_x, major_landmark_y, major_landmark_z),
        size=24.0,
        color=(0.0, 0.2, 0.5, 1.0),  # Blue text
        halo_color=(1.0, 1.0, 1.0, 0.95),
        halo_width=3.0,
        priority=150,
        underline=True,  # Underlined style
        min_zoom=0.0,
        max_zoom=5.0,
    )
    print("  Added underlined label: FIFTH STATION")
    
    # Example: Small caps label for geographic features
    geo_feature_x, geo_feature_z, _ = world_to_terrain_local(138.75, 35.37)
    geo_feature_y = 1800.0 * Z_SCALE + 60.0 * Z_SCALE
    add_label(
        sock,
        text="Crater Lake",
        world_pos=(geo_feature_x, geo_feature_y, geo_feature_z),
        size=16.0,
        color=(0.0, 0.4, 0.6, 1.0),  # Teal color
        halo_color=(1.0, 1.0, 1.0, 0.9),
        halo_width=2.0,
        priority=100,
        small_caps=True,  # Small caps style
    )
    print("  Added small-caps label: Crater Lake")
    
    # 3. Demonstrate rotated labels
    print("\n--- Rotated Label Examples ---")
    import math
    
    # Add rotated labels at different angles
    for angle_deg, name in [(45, "Ridge NE"), (-30, "Valley SW"), (90, "North Face")]:
        angle_rad = math.radians(angle_deg)
        offset_dist = 0.02  # Offset in degrees
        rot_x, rot_z, _ = world_to_terrain_local(
            138.72 + offset_dist * math.cos(angle_rad),
            35.36 + offset_dist * math.sin(angle_rad)
        )
        rot_y = 2200.0 * Z_SCALE + 70.0 * Z_SCALE
        
        add_label(
            sock,
            text=name,
            world_pos=(rot_x, rot_y, rot_z),
            size=14.0,
            color=(0.3, 0.0, 0.3, 1.0),  # Purple
            halo_color=(1.0, 1.0, 1.0, 0.85),
            halo_width=1.5,
            priority=80,
            rotation=angle_rad,
        )
        print(f"  Added rotated label ({angle_deg}°): {name}")
    
    # 4. Line label examples (simulated paths)
    print("\n--- Line Label Examples ---")
    
    # Create a simulated hiking trail path
    trail_points = []
    base_lon, base_lat = 138.71, 35.34
    for i in range(8):
        t = i / 7.0
        lon = base_lon + 0.03 * t + 0.005 * math.sin(t * math.pi * 2)
        lat = base_lat + 0.04 * t + 0.003 * math.cos(t * math.pi * 3)
        elev = (1500.0 + 800.0 * t) * Z_SCALE + 60.0 * Z_SCALE
        x, z, _ = world_to_terrain_local(lon, lat)
        trail_points.append((x, elev, z))
    
    add_line_label(
        sock,
        text="Yoshida Trail",
        polyline=trail_points,
        size=16.0,
        color=(0.6, 0.3, 0.0, 1.0),  # Brown
        halo_color=(1.0, 1.0, 1.0, 0.9),
        halo_width=2.0,
        priority=120,
        placement="along",
        repeat_distance=0.0,
    )
    print("  Added line label: Yoshida Trail")
    
    # 5. Curved label along a terrain contour
    print("\n--- Curved Label Example ---")
    
    # Create a curved path for a contour line
    contour_points = []
    center_lon, center_lat = 138.73, 35.355
    radius_deg = 0.015
    for i in range(12):
        angle = (i / 11.0) * math.pi * 1.5  # 270 degrees
        lon = center_lon + radius_deg * math.cos(angle)
        lat = center_lat + radius_deg * math.sin(angle)
        elev = 3000.0 * Z_SCALE + 70.0 * Z_SCALE
        x, z, _ = world_to_terrain_local(lon, lat)
        contour_points.append((x, elev, z))
    
    add_curved_label(
        sock,
        text="3000m CONTOUR",
        polyline=contour_points,
        size=14.0,
        color=(0.4, 0.2, 0.0, 1.0),
        halo_color=(1.0, 1.0, 1.0, 0.85),
        halo_width=1.5,
        priority=90,
        tracking=0.05,
        center_on_path=True,
    )
    print("  Added curved label: 3000m CONTOUR")
    
    # 6. Multiple callouts with different styles
    print("\n--- Additional Callouts ---")
    
    # Information callout (white background)
    info_x, info_z, _ = world_to_terrain_local(138.74, 35.34)
    info_y = 1600.0 * Z_SCALE + 80.0 * Z_SCALE
    add_callout(
        sock,
        text="Visitor Center\nOpen May-Sep",
        anchor=(info_x, info_y, info_z),
        offset=(40.0, 20.0),
        background_color=(1.0, 1.0, 1.0, 0.92),
        border_color=(0.0, 0.5, 0.8, 1.0),
        border_width=1.5,
        corner_radius=5.0,
        padding=8.0,
        text_size=13.0,
        text_color=(0.0, 0.0, 0.0, 1.0),
    )
    print("  Added info callout: Visitor Center")
    
    # Warning callout (yellow background)
    warning_x, warning_z, _ = world_to_terrain_local(138.72, 35.38)
    warning_y = 3200.0 * Z_SCALE + 90.0 * Z_SCALE
    add_callout(
        sock,
        text="⚠ Steep Slope\nCaution Required",
        anchor=(warning_x, warning_y, warning_z),
        offset=(-40.0, -40.0),
        background_color=(1.0, 1.0, 0.0, 0.88),
        border_color=(0.8, 0.0, 0.0, 1.0),
        border_width=2.0,
        corner_radius=4.0,
        padding=9.0,
        text_size=14.0,
        text_color=(0.5, 0.0, 0.0, 1.0),
    )
    print("  Added warning callout: Steep Slope")
    
    # 7. Horizon fade demonstration with low-angle labels
    print("\n--- Horizon Fade Examples ---")
    
    # Add labels at various distances to show horizon fade
    for i, (distance_factor, fade_angle) in enumerate([(0.5, 15.0), (1.0, 10.0), (1.5, 5.0)]):
        far_x, far_z, _ = world_to_terrain_local(
            138.68 + distance_factor * 0.05,
            35.32 + distance_factor * 0.03
        )
        far_y = 800.0 * Z_SCALE + 50.0 * Z_SCALE
        
        add_label(
            sock,
            text=f"Distant Peak {i+1}",
            world_pos=(far_x, far_y, far_z),
            size=15.0,
            color=(0.4, 0.4, 0.5, 1.0),  # Gray-blue
            halo_color=(1.0, 1.0, 1.0, 0.8),
            halo_width=2.0,
            priority=60 - i * 10,
            horizon_fade_angle=fade_angle,
        )
        print(f"  Added horizon-fade label: Distant Peak {i+1} (fade_angle={fade_angle}°)")
    
    # 8. Scale-dependent visibility demonstration
    print("\n--- Scale-Dependent Visibility Examples ---")
    
    # Detail labels only visible when zoomed in
    for i in range(3):
        detail_x, detail_z, _ = world_to_terrain_local(
            138.725 + i * 0.005,
            35.355 + i * 0.003
        )
        detail_y = 2800.0 * Z_SCALE + 65.0 * Z_SCALE
        
        add_label(
            sock,
            text=f"Detail {i+1}",
            world_pos=(detail_x, detail_y, detail_z),
            size=12.0,
            color=(0.2, 0.5, 0.2, 1.0),  # Green
            halo_color=(1.0, 1.0, 1.0, 0.85),
            halo_width=1.0,
            priority=40,
            min_zoom=2.0,  # Only visible when zoomed in
            max_zoom=10.0,
        )
        print(f"  Added detail label (zoom 2-10): Detail {i+1}")
    
    # 9. Priority comparison
    print("\n--- Priority System Examples ---")
    
    # Add overlapping labels with different priorities
    overlap_base_x, overlap_base_z, _ = world_to_terrain_local(138.76, 35.36)
    overlap_base_y = 2000.0 * Z_SCALE + 75.0 * Z_SCALE
    
    for priority_val, offset_m, label_name in [(200, 0, "HIGH"), (100, 20, "MED"), (50, 40, "LOW")]:
        add_label(
            sock,
            text=f"{label_name} Priority",
            world_pos=(overlap_base_x, overlap_base_y, overlap_base_z + offset_m),
            size=16.0,
            color=(1.0 - priority_val/200.0, 0.0, priority_val/200.0, 1.0),
            halo_color=(1.0, 1.0, 1.0, 0.9),
            halo_width=2.0,
            priority=priority_val,
        )
        print(f"  Added priority={priority_val} label: {label_name} Priority")
    
    # Wait for all label commands to be processed
    time.sleep(1.5)
    
    # Take snapshot if requested
    if args.snapshot:
        time.sleep(2.0)  # Wait for render to settle
        print(f"\nTaking snapshot: {args.snapshot}")
        resp = send_ipc(sock, {
            "cmd": "snapshot",
            "path": str(args.snapshot.resolve()),
        })
        print(f"Snapshot result: {resp}")
        
        # Wait for snapshot to complete
        time.sleep(1.0)
        
        # Close viewer
        send_ipc(sock, {"cmd": "close"})
        sock.close()
        process.wait(timeout=5.0)
        
        if args.snapshot.exists():
            print(f"Snapshot saved to: {args.snapshot}")
            return 0
        else:
            print("Warning: Snapshot file not created")
            return 1
    
    # Interactive mode - use shared interactive loop
    print("\nWindow controls:")
    print("  Mouse drag     - Orbit camera")
    print("  Scroll wheel   - Zoom in/out")
    print("  W/S or ↑/↓     - Tilt camera up/down")
    print("  A/D or ←/→     - Rotate camera left/right")
    print("  Q/E            - Zoom out/in")
    
    run_interactive_loop(sock, process, title="MOUNT FUJI LABELS DEMO")
    
    # Cleanup
    try:
        send_ipc(sock, {"cmd": "close"})
    except:
        pass
    sock.close()
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
