#!/usr/bin/env python3
"""Feature Picking Demo - Interactive feature selection for vector overlays.

This example demonstrates the full picking system (Plan 1-3) by loading 
vector data and enabling advanced picking functionality.

Plan 3 (Premium) Features Demonstrated:
- BVH-accelerated ray intersection
- Ray-heightfield terrain picking
- Lasso/box selection
- Highlight styles (outline, glow, pulse)
- Rich picking callbacks

Usage:
    python examples/picking_demo.py
    python examples/picking_demo.py --gpkg assets/gpkg/Mount_Fuji_places.gpkg
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.viewer_ipc import (
    find_viewer_binary,
    send_ipc,
    add_vector_overlay,
    set_vector_overlays_enabled,
    poll_pick_events,
    set_lasso_mode,
    get_lasso_state,
    clear_selection,
    set_labels_enabled,
    add_label,
    load_label_atlas,
    set_label_zoom,
    set_max_visible_labels,
)

# P0.3/M2: Sun ephemeris - calculate realistic sun position from location and time
from forge3d import sun_position, sun_position_utc, SunPosition

# Asset paths
ASSETS_DIR = Path(__file__).parent.parent / "assets"
DEFAULT_GPKG = ASSETS_DIR / "gpkg" / "Mount_Fuji_places.gpkg"
DEM_PATH = ASSETS_DIR / "tif" / "Mount_Fuji_30m.tif"
FONT_ATLAS_PNG = ASSETS_DIR / "fonts" / "default_atlas.png"
FONT_ATLAS_JSON = ASSETS_DIR / "fonts" / "default_atlas.json"

# Terrain extent in WGS84 (from gdalinfo Mount_Fuji_30m.tif)
TERRAIN_MIN_X = 138.6278010
TERRAIN_MAX_X = 138.8277887
TERRAIN_MIN_Y = 35.2605414
TERRAIN_MAX_Y = 35.4605292
TERRAIN_WIDTH_PX = 2594
TERRAIN_HEIGHT_PX = 2594

# Z scale for terrain (must match viewer terrain settings)
Z_SCALE = 0.20

def world_to_terrain_local(world_x: float, world_y: float) -> tuple[float, float]:
    """Transform WGS84 coordinates to terrain-local coordinates."""
    terrain_width = max(TERRAIN_WIDTH_PX, TERRAIN_HEIGHT_PX)
    
    # Normalize to [0, 1]
    norm_x = (world_x - TERRAIN_MIN_X) / (TERRAIN_MAX_X - TERRAIN_MIN_X)
    norm_y = (TERRAIN_MAX_Y - world_y) / (TERRAIN_MAX_Y - TERRAIN_MIN_Y)
    
    local_x = norm_x * terrain_width
    local_z = norm_y * terrain_width * (TERRAIN_HEIGHT_PX / TERRAIN_WIDTH_PX)
    
    return local_x, local_z

def create_test_triangle_layer(sock: socket.socket):
    """Create a layer of floating triangles for BVH testing."""
    vertices = []
    indices = []
    
    # Create a grid of triangles above the terrain
    rows, cols = 10, 10
    spacing = 500.0
    start_x = 138.7
    start_y = 35.3
    
    # Convert lat/lon approx to local units (just distinct positions)
    # Actually the viewer uses local coordinates if not georeferenced?
    # No, it uses the DEM coordinates. Mount Fuji DEM is lat/lon.
    # Let's put them near the summit.
    center_x = 138.7278
    center_y = 35.3606
    
    # 0.01 deg is roughly 1km
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = center_x + (c - cols/2) * 0.005
            y = center_y + (r - rows/2) * 0.005
            z = 4000.0 # Above terrain
            
            size = 0.002
            
            # Triangle
            # v0, v1, v2. format: x, y, z, r, g, b, a, feature_id
            # Use ID range 1000+ to avoid collision with labels (1+)
            fid = 1000 + idx // 3 + 1
            vertices.append([x, z, y + size, 1.0, 0.0, 0.0, 0.8, fid])
            vertices.append([x + size, z, y - size, 0.0, 1.0, 0.0, 0.8, fid])
            vertices.append([x - size, z, y - size, 0.0, 0.0, 1.0, 0.8, fid])
            
            indices.extend([idx, idx+1, idx+2])
            idx += 3
            
    add_vector_overlay(
        sock, 
        "Floating Triangles", 
        vertices, 
        indices, 
        primitive="triangles", 
        drape=False, 
        opacity=0.9,
        line_width=2.0
    )
    print(f"Added 'Floating Triangles' layer with {len(indices)//3} triangles")

def load_gpkg_points(sock: socket.socket, gpkg_path: Path, dem_path: Path):
    """Load points from GPKG as a vector overlay (using small triangles or points)."""
    import sqlite3
    import struct
    
    if not gpkg_path.exists():
        print(f"Warning: {gpkg_path} not found. Skipping.")
        return

    print(f"Loading points from {gpkg_path.name}...")
    
    # Open DEM for elevation sampling
    dem_src = None
    try:
        import rasterio
        dem_src = rasterio.open(dem_path)
        print(f"  Opened DEM: {dem_path.name}")
    except Exception as e:
        print(f"  Warning: Could not open DEM: {e}")
    
    try:
        conn = sqlite3.connect(str(gpkg_path))
        cur = conn.cursor()
        
        # Find table
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'sqlite_%'")
        tables = cur.fetchall()
        table = tables[0][0] if tables else "places"
        # Try to find a better table name if multiple exist
        if tables:
            for t in tables:
                if "places" in t[0] or "Mount_Fuji" in t[0]:
                    table = t[0]
                    break
        
        # Check for English name column
        cur.execute(f"PRAGMA table_info('{table}')")
        cols = [info[1] for info in cur.fetchall()]
        
        en_col = None
        for c in ["name.en", "name:en"]:
            if c in cols:
                en_col = c
                break
        
        if en_col:
            cur.execute(f'SELECT geom, name, "{en_col}" FROM "{table}"')
        else:
            cur.execute(f'SELECT geom, name, name FROM "{table}"')
            
        rows = cur.fetchall()
        
        vertices = []
        indices = []
        labels = []
        idx = 0
        
        for (blob, name, name_en) in rows:
            # Parse GPKG WKB point
            if not blob or len(blob) < 21: continue
            
            # Simple parser for standard GPKG WKB point (assuming little endian)
            # Header is usually bytes 0-7, then WKB starts. 
            # GPKG header: magic(2), version(1), flags(1), srs_id(4)
            # Then WKB. 
            # If standard WKB point: byte order(1), type(4), x(8), y(8)
            # Header length variable? Fixed 8 bytes usually.
            # Let's try to find the coords.
            # Byte 0-1: 'GP'
            # Byte 2: version
            # Byte 3: flags
            # ...
            # Actually easier to scan for the doubles.
            # A point is doubles X and Y.
            
            # Use the same logic as fuji_labels_demo.py if available or simplified
            # Assuming standard GPKG point (flags=0 means no envelope) -> 8 byte header
            # offset 8: byte order (1)
            # offset 9: type (4) -> 1 for point
            # offset 13: x (8)
            # offset 21: y (8)
            
            try:
                x, y = struct.unpack_from('<dd', blob, 8 + 1 + 4)
                
                # Determine display name (prefer English)
                display_name = name_en if name_en else name
                
                # Store label info with WGS84 coords for elevation sampling
                # Only use ASCII labels as the default atlas has limited character support
                if display_name and display_name.strip():
                    if all(ord(c) < 128 for c in display_name):
                        labels.append((x, y, display_name.strip()))
                
                # Create a small diamond for the point
                z = 0.0 # Will be draped
                size = 0.0001
                
                # Diamond (2 tris)
                # Center x,y
                # Format: x, y, z, r, g, b, a, feature_id
                # Use ID range 2000+ to avoid collision with labels and other layers
                fid = 2000 + idx // 4 + 1
                v0 = [x, z, y + size, 1.0, 1.0, 0.0, 1.0, fid] # Top
                v1 = [x + size, z, y, 1.0, 1.0, 0.0, 1.0, fid] # Right
                v2 = [x, z, y - size, 1.0, 1.0, 0.0, 1.0, fid] # Bottom
                v3 = [x - size, z, y, 1.0, 1.0, 0.0, 1.0, fid] # Left
                
                vertices.extend([v0, v1, v2, v3])
                
                # Tri 1: 0, 1, 2
                indices.extend([idx, idx+1, idx+2])
                # Tri 2: 0, 2, 3
                indices.extend([idx, idx+2, idx+3])
                
                idx += 4
            except:
                continue
                
        conn.close()
        
        if vertices:
            add_vector_overlay(
                sock,
                "Places",
                vertices,
                indices,
                primitive="triangles",
                drape=True,
                drape_offset=20.0, # Lift off terrain
                opacity=1.0,
                depth_bias=0.0001
            )
            print(f"Added 'Places' layer with {len(indices)//3} triangles (draped markers)")
            
            # Load font atlas (required for labels to render)
            print("Loading font atlas...")
            if FONT_ATLAS_PNG.exists() and FONT_ATLAS_JSON.exists():
                resp = load_label_atlas(sock, str(FONT_ATLAS_PNG), str(FONT_ATLAS_JSON))
                if resp.get("ok"):
                    print(f"  Font atlas loaded: {FONT_ATLAS_PNG.name}")
                else:
                    print(f"  Warning: Failed to load font atlas: {resp.get('error')}")
                time.sleep(0.5)  # Wait for atlas to be processed
            else:
                print(f"  Warning: Font atlas not found at {FONT_ATLAS_PNG}")
                print("  Labels will not render without a font atlas.")
            
            # Add text labels for place names
            set_labels_enabled(sock, True)
            set_label_zoom(sock, 1.0)
            set_max_visible_labels(sock, 100)
            
            for x, y, name in labels:
                # Convert WGS84 to terrain-local coordinates
                local_x, local_z = world_to_terrain_local(x, y)
                
                # Sample elevation from DEM at this location
                elevation = 2000.0  # Default fallback
                if dem_src:
                    try:
                        vals = list(dem_src.sample([(x, y)]))
                        if vals and len(vals) > 0:
                            val = vals[0][0]
                            if val > -10000:  # Sanity check for nodata
                                elevation = float(val)
                    except Exception:
                        pass
                
                # Position label above terrain surface (matching fuji_labels_demo.py)
                # Viewer visual height = elevation * zscale
                # Add offset scaled by zscale
                local_y = (elevation * Z_SCALE) + (50.0 * Z_SCALE)
                
                add_label(
                    sock,
                    text=name,
                    world_pos=(local_x, local_y, local_z),
                    size=20.0,
                    color=(0.1, 0.1, 0.1, 1.0),  # Dark text
                    halo_color=(1.0, 1.0, 1.0, 0.9),  # White halo
                    halo_width=2.0,
                    priority=int(elevation / 100) + 10,
                    horizon_fade_angle=10.0
                )
            print(f"Added {len(labels)} text labels for place names")
            
        if dem_src:
            dem_src.close()
            
    except Exception as e:
        print(f"Error loading GPKG: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plan 3 Picking Demo")
    parser.add_argument("--gpkg", type=Path, default=DEFAULT_GPKG)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    # P0.1/M1: OIT
    parser.add_argument("--oit", type=str, choices=["auto", "wboit", "dual_source", "off"],
                        default=None, help="OIT mode for transparent surfaces (default: off)")
    args = parser.parse_args()

    # Launch viewer
    binary = find_viewer_binary()
    cmd = [binary, "--ipc-port", "0", "--size", f"{args.width}x{args.height}"]
    
    print(f"Starting viewer: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    # Wait for READY
    port = None
    start = time.time()
    while time.time() - start < 10.0:
        line = process.stdout.readline()
        if not line: break
        print(f"  {line.rstrip()}")
        if "FORGE3D_VIEWER_READY" in line:
            import re
            m = re.search(r"port=(\d+)", line)
            if m: port = int(m.group(1))
            break
            
    if not port:
        print("Failed to start viewer")
        return 1
        
    print(f"Connected on port {port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(5.0)
    
    # Load terrain
    print("Loading terrain...")
    send_ipc(sock, {"cmd": "load_terrain", "path": str(DEM_PATH.resolve())})
    
    # Set terrain with zscale and camera (matching fuji_labels_demo.py)
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 135.0,
        "theta": 30.0,
        "radius": 15000.0,
        "fov": 45.0,
        "zscale": Z_SCALE,
        "sun_azimuth": 180.0,
        "sun_elevation": 45.0,
        "sun_intensity": 1.0
    })
    
    # Add layers
    create_test_triangle_layer(sock)
    if args.gpkg.exists():
        load_gpkg_points(sock, args.gpkg, DEM_PATH)
        
    print("\n" + "="*60)
    print("INTERACTIVE PICKING DEMO")
    print("="*60)
    print("Commands:")
    print("  pick       - Poll for recent pick events")
    print("  lasso on   - Enable lasso selection mode")
    print("  lasso off  - Disable lasso selection mode")
    print("  clear      - Clear current selection")
    print("  quit       - Exit demo")
    print("  (or just click in the viewer!)")
    print("="*60)
    
    # Interactive loop
    import os
    
    # Set non-blocking on stdout
    os.set_blocking(process.stdout.fileno(), False)
    
    try:
        while process.poll() is None:
            # Read any output from viewer
            try:
                while True:
                    line = process.stdout.readline()
                    if not line: break
                    print(f"[VIEWER] {line.rstrip()}")
            except:
                pass

            # Poll picking events periodically
            resp = poll_pick_events(sock)
            if resp.get("ok") and resp.get("pick_events"):
                events = resp["pick_events"]
                for e in events:
                    print(f"\n[EVENT] {e['event_type']}")
                    for r in e.get("results", []):
                        print(f"  HIT: Layer '{r.get('layer_name')}' ID={r.get('feature_id')}")
                        pos = r.get("world_pos", [0,0,0])
                        print(f"       Pos: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                        if "terrain_info" in r and r["terrain_info"]:
                            ti = r["terrain_info"]
                            print(f"       Terrain: Ele={ti.get('elevation',0):.1f}m Slope={ti.get('slope',0):.1f}°")
                        if "attributes" in r and r["attributes"]:
                            print(f"       Attrs: {r['attributes']}")
            
            # Check for user input (non-blocking if possible)
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "quit" or cmd == "q":
                    break
                elif cmd == "pick":
                    print("Polled events (see above)")
                elif cmd == "lasso on":
                    set_lasso_mode(sock, True)
                    print("Lasso mode ENABLED - Draw with Left Mouse Button")
                elif cmd == "lasso off":
                    set_lasso_mode(sock, False)
                    print("Lasso mode DISABLED")
                elif cmd == "clear":
                    clear_selection(sock)
                    print("Selection cleared")
                elif cmd:
                    print(f"Unknown command: {cmd}")
                    
    except KeyboardInterrupt:
        pass
    finally:
        send_ipc(sock, {"cmd": "close"})
        sock.close()
        process.terminate()

if __name__ == "__main__":
    main()
