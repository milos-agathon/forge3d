#!/usr/bin/env python3
"""Luxembourg Terrain with Rail Network Vector Overlay.

This example demonstrates the Option B vector overlay feature:
- Loads luxembourg_dem.tif as the terrain
- Loads luxembourg_rail.gpkg and converts rail lines to triangle quads (thick lines)
- Reprojects coordinates from GeoPackage CRS to EPSG:3035 (ETRS89-LAEA)
- Drapes the rail lines onto the terrain surface
- Renders with high-quality PBR lighting, shadows, and AO
- Uses terrain colormap (green valleys, brown slopes, white peaks)

Usage:
    # Basic usage with default settings
    python examples/luxembourg_rail_overlay.py --dem assets/tif/luxembourg_dem.tif \\
        --gpkg assets/gpkg/luxembourg_rail.gpkg --snapshot output.png
    
    # Custom line color and width
    python examples/luxembourg_rail_overlay.py --dem assets/tif/luxembourg_dem.tif \\
        --line-color "#FF0000" --line-width 25 --snapshot red_rails.png
    
    # Available line color options (hex format):
    #   --line-color "#E64A19"  (Deep orange, default)
    #   --line-color "#FF4444"  (Bright red)
    #   --line-color "#2196F3"  (Blue)
    #   --line-color "#4CAF50"  (Green)

Requirements:
    - geopandas (for loading .gpkg files)
    - fiona (for GeoPackage support)
    - pyproj (for coordinate reprojection to EPSG:3035)
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
except ImportError:
    print("numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from forge3d.terrain_params import (
        VectorOverlayConfig,
        VectorVertex,
        PrimitiveType,
    )
    HAS_VECTOR_API = True
except ImportError:
    HAS_VECTOR_API = False
    VectorOverlayConfig = None
    VectorVertex = None
    PrimitiveType = None

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    gpd = None
    HAS_GPD = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False

try:
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except ImportError:
    CRS = None
    Transformer = None
    HAS_PYPROJ = False


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
    msg_bytes = msg.encode()
    
    # Log message size for debugging large payloads
    msg_size = len(msg_bytes)
    if msg_size > 100000:
        print(f"  Sending large IPC message: {msg_size / 1024 / 1024:.1f} MB")
    
    try:
        sock.sendall(msg_bytes)
    except Exception as e:
        return {"ok": False, "error": f"Send failed: {e}"}
    
    data = b""
    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        except socket.timeout:
            if not data:
                return {"ok": False, "error": "Timeout waiting for response"}
            break
        except Exception as e:
            return {"ok": False, "error": f"Receive failed: {e}"}
    
    line = data.decode().strip()
    if not line:
        return {"ok": False, "error": "Empty response from viewer"}
    
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid JSON response: {e}"}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> List[float]:
    """Convert hex color to RGBA list (0.0-1.0 range).
    
    Args:
        hex_color: Color in hex format, e.g. '#FF5500' or 'FF5500'
        alpha: Alpha value (0.0-1.0)
        
    Returns:
        List of [R, G, B, A] values in 0.0-1.0 range
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return [r, g, b, alpha]
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return [r, g, b, a]
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def load_gpkg_lines(gpkg_path: Path, dem_path: Path, color: List[float], line_width: float = 10.0) -> Tuple[List[List[float]], List[int]]:
    """Load line geometries from GeoPackage and convert to triangle quads.
    
    The coordinates are reprojected to EPSG:3035 and mapped from GeoPackage bounds 
    to terrain world space, where terrain is centered at origin.
    
    Args:
        gpkg_path: Path to GeoPackage file
        dem_path: Path to DEM file (for reading dimensions)
        color: RGBA color as list [r, g, b, a] in 0.0-1.0 range
        line_width: Width of lines in world units
        
    Returns:
        vertices: List of [x, y, z, r, g, b, a] for each vertex
        indices: List of triangle indices
    """
    if gpd is None:
        print("geopandas is required for .gpkg files. Install with: pip install geopandas fiona")
        return [], []
    
    gdf = gpd.read_file(str(gpkg_path))
    print(f"  Original CRS: {gdf.crs}")
    
    # Reproject to EPSG:3035 (ETRS89-LAEA) for European data
    target_crs = "EPSG:3035"
    if gdf.crs is not None and str(gdf.crs) != target_crs:
        print(f"  Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    else:
        print(f"  Using target CRS: {target_crs}")
    
    # Get terrain dimensions and bounds from DEM
    terrain_width = 1000.0  # Default
    dem_min_x, dem_min_y, dem_max_x, dem_max_y = 0.0, 0.0, 1.0, 1.0
    
    if HAS_RASTERIO and dem_path.exists():
        try:
            with rasterio.open(dem_path) as dem:
                terrain_width = float(max(dem.width, dem.height))
                print(f"  DEM dimensions: {dem.width}x{dem.height}, terrain_width={terrain_width}")
                
                # Get DEM bounds in its native CRS
                dem_bounds = dem.bounds
                dem_crs = dem.crs
                print(f"  DEM CRS: {dem_crs}")
                print(f"  DEM bounds (native): left={dem_bounds.left:.6f}, bottom={dem_bounds.bottom:.6f}, right={dem_bounds.right:.6f}, top={dem_bounds.top:.6f}")
                
                # Reproject DEM bounds to target CRS if needed
                if HAS_PYPROJ and dem_crs is not None:
                    try:
                        from pyproj import CRS as ProjCRS, Transformer
                        dem_proj_crs = ProjCRS.from_user_input(dem_crs)
                        target_proj_crs = ProjCRS.from_user_input(target_crs)
                        
                        transformer = Transformer.from_crs(dem_proj_crs, target_proj_crs, always_xy=True)
                        
                        # Transform all four corners of DEM and get bounding box
                        corners = [
                            (dem_bounds.left, dem_bounds.bottom),
                            (dem_bounds.left, dem_bounds.top),
                            (dem_bounds.right, dem_bounds.bottom),
                            (dem_bounds.right, dem_bounds.top),
                        ]
                        transformed_corners = [transformer.transform(x, y) for x, y in corners]
                        
                        dem_min_x = min(c[0] for c in transformed_corners)
                        dem_max_x = max(c[0] for c in transformed_corners)
                        dem_min_y = min(c[1] for c in transformed_corners)
                        dem_max_y = max(c[1] for c in transformed_corners)
                        
                        print(f"  DEM bounds (EPSG:3035): X=[{dem_min_x:.1f}, {dem_max_x:.1f}], Y=[{dem_min_y:.1f}, {dem_max_y:.1f}]")
                    except Exception as e:
                        print(f"  Warning: Could not reproject DEM bounds: {e}")
                        # Fall back to using GeoPackage bounds
                        total_bounds = gdf.total_bounds
                        dem_min_x, dem_min_y, dem_max_x, dem_max_y = total_bounds
                else:
                    # If no pyproj, use DEM bounds directly (assumes same CRS)
                    dem_min_x, dem_min_y = dem_bounds.left, dem_bounds.bottom
                    dem_max_x, dem_max_y = dem_bounds.right, dem_bounds.top
                    print(f"  Warning: pyproj not available, using DEM bounds directly")
        except Exception as e:
            print(f"  Warning: Could not read DEM: {e}")
            # Fall back to GeoPackage bounds
            total_bounds = gdf.total_bounds
            dem_min_x, dem_min_y, dem_max_x, dem_max_y = total_bounds
    else:
        print(f"  Using default terrain_width={terrain_width} (rasterio not available or DEM not found)")
        # Fall back to GeoPackage bounds
        total_bounds = gdf.total_bounds
        dem_min_x, dem_min_y, dem_max_x, dem_max_y = total_bounds
    
    # Also print GeoPackage bounds for reference
    gpkg_bounds = gdf.total_bounds
    print(f"  GeoPackage bounds (EPSG:3035): X=[{gpkg_bounds[0]:.1f}, {gpkg_bounds[2]:.1f}], Y=[{gpkg_bounds[1]:.1f}, {gpkg_bounds[3]:.1f}]")
    
    vertices = []
    indices = []
    
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            _add_linestring_as_quads(coords, color, vertices, indices, 0,
                          dem_min_x, dem_max_x, dem_min_y, dem_max_y, 
                          terrain_width, line_width)
            
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                _add_linestring_as_quads(coords, color, vertices, indices, 0,
                              dem_min_x, dem_max_x, dem_min_y, dem_max_y, 
                              terrain_width, line_width)
    
    print(f"Loaded {len(vertices)} vertices, {len(indices)//3} triangles from {gpkg_path}")
    return vertices, indices


def _add_linestring_as_quads(coords: list, color: list, vertices: list, indices: list, 
                              vertex_offset: int, min_x: float, max_x: float, 
                              min_y: float, max_y: float, terrain_width: float,
                              line_width: float = 10.0):
    """Add a linestring as continuous triangle strip for thick line rendering.
    
    WebGPU/wgpu doesn't support wide lines, so we convert the linestring
    into a continuous triangle strip with the specified width. This creates
    solid lines without gaps between segments.
    """
    if len(coords) < 2:
        return
    
    # First convert all coords to world space (0 to terrain_width)
    world_coords = []
    for x, y, *rest in coords:
        u = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
        v = (max_y - y) / (max_y - min_y) if max_y != min_y else 0.5
        world_x = u * terrain_width
        world_z = v * terrain_width
        world_coords.append((world_x, world_z))
    
    # Half width for offset calculation
    half_w = line_width / 2.0
    
    # Generate continuous strip vertices along the entire linestring
    # For each point, we create 2 vertices (left and right of the line)
    strip_vertices = []
    
    for i in range(len(world_coords)):
        x, z = world_coords[i]
        
        # Calculate tangent direction (average of prev and next segments)
        if i == 0:
            # First point: use direction to next point
            nx, nz = world_coords[i + 1]
            dx, dz = nx - x, nz - z
        elif i == len(world_coords) - 1:
            # Last point: use direction from previous point
            px, pz = world_coords[i - 1]
            dx, dz = x - px, z - pz
        else:
            # Middle point: average of incoming and outgoing directions
            px, pz = world_coords[i - 1]
            nx, nz = world_coords[i + 1]
            dx1, dz1 = x - px, z - pz
            dx2, dz2 = nx - x, nz - z
            # Normalize both and average
            len1 = (dx1*dx1 + dz1*dz1) ** 0.5
            len2 = (dx2*dx2 + dz2*dz2) ** 0.5
            if len1 > 0.001 and len2 > 0.001:
                dx = dx1/len1 + dx2/len2
                dz = dz1/len1 + dz2/len2
            elif len1 > 0.001:
                dx, dz = dx1, dz1
            else:
                dx, dz = dx2, dz2
        
        # Normalize tangent
        length = (dx * dx + dz * dz) ** 0.5
        if length < 0.001:
            # Skip degenerate points
            continue
        dx /= length
        dz /= length
        
        # Perpendicular vector (rotated 90 degrees)
        perp_x = -dz * half_w
        perp_z = dx * half_w
        
        # Add left and right vertices
        strip_vertices.append((x - perp_x, z - perp_z))  # Left
        strip_vertices.append((x + perp_x, z + perp_z))  # Right
    
    if len(strip_vertices) < 4:
        return
    
    # Add vertices and create triangle strip indices
    base_idx = len(vertices)
    
    # Add all strip vertices
    for vx, vz in strip_vertices:
        vertices.append([vx, 0.0, vz] + color)
    
    # Create triangles connecting the strip
    # Strip vertices are: L0, R0, L1, R1, L2, R2, ...
    # We need triangles: (L0,R0,L1), (R0,R1,L1), (L1,R1,L2), (R1,R2,L2), ...
    num_points = len(strip_vertices) // 2
    for i in range(num_points - 1):
        # Indices for this quad
        l0 = base_idx + i * 2      # Left vertex of current point
        r0 = base_idx + i * 2 + 1  # Right vertex of current point
        l1 = base_idx + (i+1) * 2  # Left vertex of next point
        r1 = base_idx + (i+1) * 2 + 1  # Right vertex of next point
        
        # Two triangles forming a quad
        indices.extend([l0, r0, l1])  # Triangle 1
        indices.extend([r0, r1, l1])  # Triangle 2


def create_demo_lines(terrain_width: float, line_width: float = 15.0) -> Tuple[List[List[float]], List[int]]:
    """Create simple demo lines as triangle quads when no GeoPackage is available."""
    vertices = []
    indices = []
    
    # Create a simple grid of lines for demonstration
    color = [0.9, 0.3, 0.1, 1.0]  # Red-orange
    half_w = terrain_width * 0.4
    half_line = line_width / 2.0
    
    # Create horizontal lines as quads
    for i in range(-3, 4):
        z = i * terrain_width * 0.1
        points = list(np.linspace(-half_w, half_w, 10))
        for j in range(len(points) - 1):
            x1, x2 = points[j], points[j + 1]
            # Create quad vertices (perpendicular to line direction)
            v_idx = len(vertices)
            vertices.extend([
                [x1, 0.0, z - half_line] + color,  # BL
                [x1, 0.0, z + half_line] + color,  # BR
                [x2, 0.0, z - half_line] + color,  # TL
                [x2, 0.0, z + half_line] + color,  # TR
            ])
            indices.extend([v_idx, v_idx + 1, v_idx + 2, v_idx + 1, v_idx + 3, v_idx + 2])
    
    # Create vertical lines as quads
    for i in range(-3, 4):
        x = i * terrain_width * 0.1
        points = list(np.linspace(-half_w, half_w, 10))
        for j in range(len(points) - 1):
            z1, z2 = points[j], points[j + 1]
            # Create quad vertices (perpendicular to line direction)
            v_idx = len(vertices)
            vertices.extend([
                [x - half_line, 0.0, z1] + color,  # BL
                [x + half_line, 0.0, z1] + color,  # BR
                [x - half_line, 0.0, z2] + color,  # TL
                [x + half_line, 0.0, z2] + color,  # TR
            ])
            indices.extend([v_idx, v_idx + 1, v_idx + 2, v_idx + 1, v_idx + 3, v_idx + 2])
    
    print(f"Created demo grid with {len(vertices)} vertices, {len(indices)//3} triangles")
    return vertices, indices


def load_gpkg_lines_sqlite(gpkg_path: Path, dem_path: Path, color: List[float], line_width: float = 10.0) -> Tuple[List[List[float]], List[int]]:
    """Load lines from GeoPackage using sqlite3 (no geopandas required)."""
    import sqlite3
    import struct
    
    conn = sqlite3.connect(str(gpkg_path))
    c = conn.cursor()
    
    # helper to find table name
    c.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")
    rows = c.fetchall()
    if not rows:
        print("No feature table found in GPKG")
        conn.close()
        return [], []
    
    table_name = rows[0][0]
    
    # Discover the geometry column name from gpkg_geometry_columns
    geom_col = "geom"  # Default fallback
    try:
        c.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", (table_name,))
        geom_row = c.fetchone()
        if geom_row:
            geom_col = geom_row[0]
            print(f"  Found geometry column: {geom_col}")
    except Exception as e:
        print(f"  Could not query gpkg_geometry_columns: {e}, using default 'geom'")
    
    # Get terrain dimensions for scaling
    terrain_width = 1000.0
    if HAS_RASTERIO and dem_path.exists():
        try:
            with rasterio.open(dem_path) as dem:
                terrain_width = float(max(dem.width, dem.height))
        except:
            pass
            
    # Read geometries
    raw_lines = []
    
    try:
        c.execute(f"SELECT [{geom_col}] FROM {table_name}")
        for (blob,) in c:
            if not blob: continue
            
            # Parse GPKG Header: Magic(2) + Version(1) + Flags(1)
            if blob[:2] != b'GP': continue
            flags = blob[3]
            # Envelope code (bits 1-3)
            env_code = (flags >> 1) & 0x07
            env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env_code, 0)
            
            offset = 8 + env_len
            wkb = blob[offset:]
            
            # Helper to read WKB LineString
            byte_order = wkb[0]
            endian = '<' if byte_order == 1 else '>'
            
            # Geometry Type (4 bytes)
            geom_type = struct.unpack(endian + 'I', wkb[1:5])[0]
            base_type = geom_type % 1000
            
            if base_type == 2: # LineString
                num_points = struct.unpack(endian + 'I', wkb[5:9])[0]
                ptr = 9
                pts = []
                for _ in range(num_points):
                    x, y = struct.unpack(endian + 'dd', wkb[ptr:ptr+16])
                    pts.append((x,y))
                    ptr += 16
                raw_lines.append(pts)
                
            elif base_type == 5: # MultiLineString
                num_parts = struct.unpack(endian + 'I', wkb[5:9])[0]
                ptr = 9
                for _ in range(num_parts):
                    if ptr >= len(wkb): break
                    part_bo = wkb[ptr]
                    part_end = '<' if part_bo == 1 else '>'
                    ptr += 1
                    part_type = struct.unpack(part_end + 'I', wkb[ptr:ptr+4])[0]
                    ptr += 4
                    
                    if part_type % 1000 == 2: # LineString
                        n_pts = struct.unpack(part_end + 'I', wkb[ptr:ptr+4])[0]
                        ptr += 4
                        part_pts = []
                        for _ in range(n_pts):
                            x, y = struct.unpack(part_end + 'dd', wkb[ptr:ptr+16])
                            part_pts.append((x,y))
                            ptr += 16
                        raw_lines.append(part_pts)
                        
    except Exception as e:
        print(f"Error parsing GPKG geometry: {e}")
        conn.close()
        return [], []
        
    conn.close()
    
    if not raw_lines:
        return [], []
        
    # Compute bounds and normalize
    all_x = [p[0] for l in raw_lines for p in l]
    all_y = [p[1] for l in raw_lines for p in l]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    print(f"  Native Bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
    
    # Generate quads
    vertices = []
    indices = []
    
    for pts in raw_lines:
        _add_linestring_as_quads(pts, color, vertices, indices, 0,
                      min_x, max_x, min_y, max_y, 
                      terrain_width, line_width)
    return vertices, indices


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Luxembourg terrain viewer with rail network vector overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Basic options
    parser.add_argument("--dem", type=Path, 
                        default=Path("assets/tif/luxembourg_dem.tif"),
                        help="Path to GeoTIFF DEM file")
    parser.add_argument("--gpkg", type=Path,
                        default=Path("assets/gpkg/luxembourg_rail.gpkg"),
                        help="Path to GeoPackage vector file")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="Take snapshot at this path and exit")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--preset", choices=["default", "alpine", "cinematic", "high_quality"],
                        help="Rendering preset (overrides individual settings)")
    
    # Camera parameters
    cam_group = parser.add_argument_group("Camera", "Camera position and view settings")
    cam_group.add_argument("--phi", type=float, default=135.0, help="Camera azimuth (degrees)")
    cam_group.add_argument("--theta", type=float, default=35.0, help="Camera elevation (degrees)")
    cam_group.add_argument("--radius", type=float, default=3000.0, help="Camera distance")
    cam_group.add_argument("--fov", type=float, default=55.0, help="Field of view (degrees)")
    cam_group.add_argument("--z-scale", dest="z_scale", type=float, default=0.15,
                           help="Vertical exaggeration factor (default: 0.15)")
    
    # PBR+POM rendering options
    pbr_group = parser.add_argument_group("PBR Rendering", "High-quality terrain rendering options")
    pbr_group.add_argument("--pbr", action="store_true", default=True,
                           help="Enable PBR+POM rendering mode (default: on)")
    pbr_group.add_argument("--no-pbr", action="store_false", dest="pbr",
                           help="Disable PBR rendering")
    pbr_group.add_argument("--hdr", type=Path,
                           help="HDR environment map for IBL lighting")
    pbr_group.add_argument("--shadows", choices=["none", "hard", "pcf", "pcss"], default="pcss",
                           help="Shadow technique (default: pcss)")
    pbr_group.add_argument("--shadow-map-res", type=int, default=4096,
                           help="Shadow map resolution (default: 4096)")
    pbr_group.add_argument("--exposure", type=float, default=1.2,
                           help="ACES exposure multiplier (default: 1.2)")
    pbr_group.add_argument("--msaa", type=int, choices=[1, 4, 8], default=1,
                           help="MSAA samples (default: 1)")
    pbr_group.add_argument("--ibl-intensity", type=float, default=0.8,
                           help="IBL intensity multiplier (default: 0.8)")
    pbr_group.add_argument("--normal-strength", type=float, default=1.2,
                           help="Terrain normal strength (default: 1.2)")
    
    # Sun/lighting options
    sun_group = parser.add_argument_group("Sun Lighting", "Directional sun light parameters")
    sun_group.add_argument("--sun-azimuth", type=float, default=135.0,
                           help="Sun azimuth angle in degrees (default: 135.0)")
    sun_group.add_argument("--sun-elevation", type=float, default=35.0,
                           help="Sun elevation angle in degrees (default: 35.0)")
    sun_group.add_argument("--sun-intensity", type=float, default=1.2,
                           help="Sun light intensity (default: 1.2)")
    
    # Heightfield Ray AO options
    ao_group = parser.add_argument_group("Heightfield AO", "Terrain ambient occlusion from heightfield ray-tracing")
    ao_group.add_argument("--height-ao", action="store_true", default=True,
                          help="Enable heightfield ray-traced ambient occlusion (default: on)")
    ao_group.add_argument("--no-height-ao", action="store_false", dest="height_ao",
                          help="Disable heightfield AO")
    ao_group.add_argument("--height-ao-directions", type=int, default=8,
                          help="Number of ray directions around horizon [4-16] (default: 8)")
    ao_group.add_argument("--height-ao-steps", type=int, default=24,
                          help="Ray march steps per direction [8-64] (default: 24)")
    ao_group.add_argument("--height-ao-distance", type=float, default=300.0,
                          help="Maximum ray distance in world units (default: 300.0)")
    ao_group.add_argument("--height-ao-strength", type=float, default=1.0,
                          help="AO darkening intensity [0.0-2.0] (default: 1.0)")
    ao_group.add_argument("--height-ao-resolution", type=float, default=1.0,
                          help="AO texture resolution scale [0.1-1.0] (default: 1.0)")
    
    # Sun Visibility options
    sv_group = parser.add_argument_group("Sun Visibility", "Terrain self-shadowing from heightfield ray-tracing")
    sv_group.add_argument("--sun-vis", action="store_true", default=True,
                          help="Enable heightfield ray-traced sun visibility (default: on)")
    sv_group.add_argument("--no-sun-vis", action="store_false", dest="sun_vis",
                          help="Disable sun visibility")
    sv_group.add_argument("--sun-vis-mode", choices=["hard", "soft"], default="soft",
                          help="Shadow mode: hard (binary) or soft (jittered) (default: soft)")
    sv_group.add_argument("--sun-vis-samples", type=int, default=6,
                          help="Number of jittered samples for soft shadows [1-16] (default: 6)")
    sv_group.add_argument("--sun-vis-steps", type=int, default=32,
                          help="Ray march steps toward sun [8-64] (default: 32)")
    sv_group.add_argument("--sun-vis-distance", type=float, default=500.0,
                          help="Maximum ray distance in world units (default: 500.0)")
    sv_group.add_argument("--sun-vis-softness", type=float, default=1.2,
                          help="Soft shadow penumbra size [0.0-4.0] (default: 1.2)")
    sv_group.add_argument("--sun-vis-bias", type=float, default=0.01,
                          help="Self-shadowing bias to reduce artifacts (default: 0.01)")
    sv_group.add_argument("--sun-vis-resolution", type=float, default=1.0,
                          help="Visibility texture resolution scale [0.1-1.0] (default: 1.0)")
    
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
    
    # M5: Vector Overlay options (depth/halo)
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
    
    # Vector overlay parameters (rail-specific)
    rail_group = parser.add_argument_group("Rail Overlay", "Rail network overlay parameters")
    rail_group.add_argument("--no-overlay", action="store_true",
                            help="Disable rail overlay (terrain only)")
    rail_group.add_argument("--overlay-opacity", type=float, default=1.0,
                            help="Rail overlay opacity 0.0-1.0 (default: 1.0)")
    rail_group.add_argument("--no-solid", action="store_true",
                            help="Hide terrain where raster overlay alpha=0 (NOTE: only works with raster overlays like swiss_terrain_landcover_viewer.py; for vector overlays, terrain always remains visible)")
    rail_group.add_argument("--drape", action="store_true", default=True, help="Drape lines onto terrain")
    rail_group.add_argument("--no-drape", action="store_false", dest="drape", help="Don't drape lines")
    rail_group.add_argument("--drape-offset", type=float, default=50.0, help="Height above terrain (default: 50)")
    rail_group.add_argument("--line-width", type=float, default=15.0, 
                            help="Line width in world units (default: 15, visible range: 5-50)")
    rail_group.add_argument("--line-color", type=str, default="#E64A19",
                            help="Line color in hex format, e.g. #FF5500 or #E64A19 (default: deep orange)")
    rail_group.add_argument("--max-vertices", type=int, default=20000,
                            help="Maximum vertices to send (decimates if needed, default: 20000)")
    
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
    
    # Find viewer binary
    try:
        viewer_bin = find_viewer_binary()
    except FileNotFoundError as e:
        print(e)
        return 1
    
    # Check DEM exists
    if not args.dem.exists():
        print(f"DEM file not found: {args.dem}")
        return 1
    
    # Launch viewer
    print(f"Launching viewer: {viewer_bin}")
    cmd = [viewer_bin, "--ipc-port", "0", "--size", f"{args.width}x{args.height}"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Wait for FORGE3D_VIEWER_READY message with port
    ready_pattern = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
    port = None
    start = time.time()
    
    while time.time() - start < 30.0:
        if proc.poll() is not None:
            print("Viewer exited unexpectedly")
            return 1
        line = proc.stdout.readline()
        if line:
            print(f"  {line.rstrip()}")  # Show viewer output
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                print(f"Viewer ready on port {port}")
                break
    
    if port is None:
        print("Timeout waiting for viewer to start")
        proc.terminate()
        return 1
    
    # Connect to IPC
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    
    # Load terrain
    print(f"Loading terrain: {args.dem}")
    resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(args.dem.absolute())})
    if not resp.get("ok"):
        print(f"Failed to load terrain: {resp.get('error')}")
    
    # Set camera
    resp = send_ipc(sock, {
        "cmd": "set_terrain_camera",
        "phi_deg": args.phi,
        "theta_deg": args.theta,
        "radius": args.radius,
        "fov_deg": args.fov
    })
    
    # Set terrain parameters including z-scale
    resp = send_ipc(sock, {
        "cmd": "set_terrain",
        "zscale": args.z_scale,
    })
    
    # Set sun
    resp = send_ipc(sock, {
        "cmd": "set_terrain_sun",
        "azimuth_deg": args.sun_azimuth,
        "elevation_deg": args.sun_elevation,
        "intensity": args.sun_intensity
    })
    
    # Enable PBR mode with high-quality effects
    if args.pbr:
        pbr_config = {
            "cmd": "set_terrain_pbr",
            "enabled": True,
            "exposure": args.exposure,
            "shadow_technique": args.shadows,
            "shadow_map_res": args.shadow_map_res,
            "ibl_intensity": args.ibl_intensity,
            "normal_strength": args.normal_strength,
            "msaa": args.msaa,
        }
        
        if args.hdr:
            pbr_config["hdr_path"] = str(args.hdr.resolve())
        
        # Heightfield Ray AO settings
        if args.height_ao:
            pbr_config["height_ao"] = {
                "enabled": True,
                "directions": args.height_ao_directions,
                "steps": args.height_ao_steps,
                "max_distance": args.height_ao_distance,
                "strength": args.height_ao_strength,
                "resolution_scale": args.height_ao_resolution,
            }
        
        # Sun Visibility settings
        if args.sun_vis:
            pbr_config["sun_visibility"] = {
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
            pbr_config["materials"] = {
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
            pbr_config["vector_overlay"] = {
                "depth_test": args.overlay_depth,
                "depth_bias": args.overlay_depth_bias,
                "halo_enabled": args.overlay_halo,
                "halo_width": args.overlay_halo_width,
                "halo_color": halo_color,
            }
        
        # M6: Tonemap settings
        pbr_config["tonemap"] = {
            "operator": args.tonemap,
            "white_point": args.tonemap_white_point,
            "white_balance_enabled": args.white_balance,
            "temperature": args.temperature,
            "tint": args.tint,
        }
        
        # M3: Depth of Field settings
        if args.dof:
            pbr_config["dof"] = {
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
            pbr_config["motion_blur"] = {
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
            pbr_config["lens_effects"] = {
                "enabled": True,
                "distortion": args.lens_distortion,
                "chromatic_aberration": args.lens_ca,
                "vignette_strength": args.lens_vignette,
                "vignette_radius": args.lens_vignette_radius,
                "vignette_softness": args.lens_vignette_softness,
            }
        
        # M5: Denoise settings
        if args.denoise:
            pbr_config["denoise"] = {
                "enabled": True,
                "method": args.denoise_method,
                "iterations": args.denoise_iterations,
                "sigma_color": args.denoise_sigma_color,
            }
        
        # M6: Volumetrics settings
        if args.volumetrics:
            pbr_config["volumetrics"] = {
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
            pbr_config["sky"] = {
                "enabled": True,
                "turbidity": args.sky_turbidity,
                "ground_albedo": args.sky_ground_albedo,
                "sun_intensity": args.sky_sun_intensity,
                "aerial_perspective": args.sky_aerial,
                "sky_exposure": args.sky_exposure,
            }
        
        resp = send_ipc(sock, pbr_config)
        if not resp.get("ok"):
            print(f"PBR config warning: {resp.get('error')}")
        else:
            features = [f"shadows={args.shadows}", f"exposure={args.exposure}"]
            if args.height_ao:
                features.append("height_ao=on")
            if args.sun_vis:
                features.append(f"sun_vis={args.sun_vis_mode}")
            if args.snow:
                features.append("snow=on")
            if args.rock:
                features.append("rock=on")
            if args.overlay_depth or args.overlay_halo:
                features.append("overlay=on")
            features.append(f"tonemap={args.tonemap}")
            if args.dof:
                features.append("dof=on")
            if args.motion_blur:
                features.append(f"motion_blur={args.mb_samples}spp")
            if args.lens_effects or args.lens_distortion != 0.0 or args.lens_ca != 0.0 or args.lens_vignette > 0.0:
                features.append("lens=on")
            if args.denoise:
                features.append(f"denoise={args.denoise_method}")
            if args.volumetrics:
                features.append("volumetrics=on")
            if args.sky:
                features.append("sky=on")
            print(f"PBR mode enabled: {', '.join(features)}")
    
    # Parse line color from hex
    try:
        line_color = hex_to_rgba(args.line_color, args.overlay_opacity)
    except ValueError as e:
        print(f"Invalid line color: {e}, using default orange")
        line_color = [0.9, 0.26, 0.1, args.overlay_opacity]
    
    # Load vector overlay - use triangles (quads) for visible thick lines
    vertices = []
    indices = []
    
    # Load rail overlay (unless disabled)
    if not args.no_overlay:
        if args.gpkg.exists() and gpd is not None and HAS_PYPROJ:
            print(f"Loading vector overlay (via geopandas): {args.gpkg}")
            print(f"  Line color: {args.line_color} -> RGBA {line_color}")
            print(f"  Line width: {args.line_width} world units")
            print(f"  Target CRS: EPSG:3035 (ETRS89-LAEA)")
            vertices, indices = load_gpkg_lines(args.gpkg, args.dem, line_color, args.line_width)
        elif args.gpkg.exists():
            # Fallback to native sqlite3 reader
            print(f"Geopandas not available or partial dependencies. Using native SQLite reader for: {args.gpkg}")
            try:
                vertices, indices = load_gpkg_lines_sqlite(args.gpkg, args.dem, line_color, args.line_width)
            except Exception as e:
                print(f"SQLite fallback failed: {e}")
                vertices, indices = [], []

        if not vertices:
            if not args.gpkg.exists():
                print(f"GeoPackage not found: {args.gpkg}, using demo lines")
            elif gpd is None:
                 print("Using demo lines (failed to load GeoPackage)")
            
            # Create demo lines with terrain_width from DEM
            terrain_width = 1000.0  # Default
            if HAS_RASTERIO and args.dem.exists():
                try:
                    with rasterio.open(args.dem) as dem:
                        terrain_width = float(max(dem.width, dem.height))
                except:
                    pass
            vertices, indices = create_demo_lines(terrain_width, args.line_width)
            # Update colors for demo lines
            for v in vertices:
                v[3:7] = line_color
    
    # Note: Decimation removed - IPC now handles large messages (tested up to 6MB+)
    # The continuous triangle strip format doesn't support simple decimation
    # without breaking line connectivity
    
    # Add vector overlay if we have data (rail overlay is visible here)
    if vertices and indices:
        print(f"Adding vector overlay: {len(vertices)} vertices, {len(indices)//3} triangles")
        
        # For large overlays (>10000 vertices), use raw dict directly to avoid
        # memory overhead of VectorVertex object conversion
        use_raw_dict = len(vertices) > 10000 or not HAS_VECTOR_API or VectorOverlayConfig is None
        
        if not use_raw_dict:
            # Use the new Python API classes (demonstrates Option B API from plan)
            try:
                vertex_objects = [
                    VectorVertex(x=v[0], y=v[1], z=v[2], r=v[3], g=v[4], b=v[5], a=v[6])
                    for v in vertices
                ]
                
                overlay_config = VectorOverlayConfig(
                    name="rail_network",
                    vertices=vertex_objects,
                    indices=indices,
                    primitive=PrimitiveType.TRIANGLES,
                    drape=args.drape,
                    drape_offset=args.drape_offset,
                    opacity=args.overlay_opacity,
                    depth_bias=0.5,
                    line_width=args.line_width,
                    point_size=5.0,
                    z_order=0,
                )
                ipc_cmd = overlay_config.to_ipc_dict()
            except Exception as e:
                print(f"  VectorOverlayConfig failed ({e}), using raw dict")
                use_raw_dict = True
        
        if use_raw_dict:
            # Send raw dict directly (more efficient for large overlays)
            ipc_cmd = {
                "cmd": "add_vector_overlay",
                "name": "rail_network",
                "vertices": vertices,
                "indices": indices,
                "primitive": "triangles",
                "drape": args.drape,
                "drape_offset": args.drape_offset,
                "opacity": args.overlay_opacity,
                "depth_bias": 5.0,
                "line_width": args.line_width,
                "point_size": 5.0,
                "z_order": 0
            }
        
        # Send with extended timeout for large payloads
        old_timeout = sock.gettimeout()
        sock.settimeout(120.0)  # 2 minutes for large overlays
        resp = send_ipc(sock, ipc_cmd)
        sock.settimeout(old_timeout)
        
        if not resp.get("ok"):
            print(f"Vector overlay warning: {resp.get('error')}")
        else:
            print(f"Vector overlay added successfully (opacity={args.overlay_opacity})")
            
            # Handle --no-solid: note that this only works with RASTER overlays
            # For vector overlays (triangle geometry), the terrain shader always renders
            # the full DEM surface. The set_overlay_solid IPC command is sent but will
            # only have effect when a raster overlay (load_overlay) is also present.
            if args.no_solid:
                # Send the IPC command (follows swiss_terrain_landcover_viewer.py pattern)
                send_ipc(sock, {"cmd": "set_overlay_solid", "solid": False})
                print("Note: --no-solid flag set, but terrain remains visible because")
                print("      vector overlays (rail lines) don't mask the terrain.")
                print("      For terrain masking, use a raster overlay like in swiss_terrain_landcover_viewer.py")
    elif not args.no_overlay:
        print("Rail overlay requested but failed to load")
    else:
        print("Rail overlay disabled (--no-overlay flag)")
    
    # Wait for rendering to complete
    time.sleep(3.0)
    
    # Take snapshot if requested
    if args.snapshot:
        print(f"Taking snapshot: {args.snapshot}")
        resp = send_ipc(sock, {
            "cmd": "snapshot",
            "path": str(Path(args.snapshot).absolute()),
            "width": args.width,
            "height": args.height
        })
        if resp.get("ok"):
            # Wait for snapshot to be written
            time.sleep(2.0)
            print(f"Snapshot saved to {args.snapshot}")
        else:
            print(f"Snapshot failed: {resp.get('error')}")
        
        # Close viewer
        send_ipc(sock, {"cmd": "close"})
        sock.close()
        proc.wait(timeout=5.0)
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("LUXEMBOURG RAIL VIEWER")
        print("=" * 60)
        print()
        print("Terminal commands (set any combination of parameters):")
        print("  set phi=45 theta=60 radius=2000 fov=55")
        print("  set sun_az=135 sun_el=45 intensity=1.5 ambient=0.3")
        print("  set zscale=2.0 shadow=0.5")
        print("  set background=0.2,0.3,0.5")
        print("  set water=1500 water_color=0.1,0.3,0.5")
        print()
        print("Other commands:")
        print("  params         - Show current parameters")
        print("  snap <path> [<width>x<height>]  - Take snapshot")
        print("  pbr on/off     - Toggle PBR rendering mode")
        print("  pbr shadows=pcss exposure=1.5 ibl=2.0")
        print("  quit           - Close viewer")
        print("=" * 60 + "\n")
        
        def parse_set_command(args: str) -> dict:
            """Parse 'set key=value key=value ...' into IPC params."""
            params = {"cmd": "set_terrain"}
            for pair in args.split():
                if "=" not in pair:
                    continue
                key, val = pair.split("=", 1)
                key = key.lower().strip()
                val = val.strip()
                
                # Map friendly names to IPC param names
                key_map = {
                    "phi": "phi", "theta": "theta", "radius": "radius", "fov": "fov",
                    "sun_az": "sun_azimuth", "sun_azimuth": "sun_azimuth",
                    "sun_el": "sun_elevation", "sun_elevation": "sun_elevation",
                    "intensity": "sun_intensity", "sun_intensity": "sun_intensity",
                    "ambient": "ambient", "zscale": "zscale", "z_scale": "zscale",
                    "shadow": "shadow", "bg": "background", "background": "background",
                    "water": "water_level", "water_level": "water_level",
                    "water_color": "water_color",
                }
                
                ipc_key = key_map.get(key, key)
                
                # Parse value
                if "," in val:  # RGB color
                    params[ipc_key] = [float(x) for x in val.split(",")]
                else:
                    params[ipc_key] = float(val)
            
            return params

        try:
            while proc.poll() is None:
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
                            if not m:
                                print("Usage: snap <path> [<width>x<height>]")
                                continue
                            snap_cmd["width"] = int(m.group(1))
                            snap_cmd["height"] = int(m.group(2))
                        elif len(snap_parts) == 3:
                            try:
                                snap_cmd["width"] = int(snap_parts[1])
                                snap_cmd["height"] = int(snap_parts[2])
                            except ValueError:
                                print("Usage: snap <path> [<width>x<height>]")
                                continue
                        elif len(snap_parts) > 3:
                            print("Usage: snap <path> [<width>x<height>]")
                            continue

                        resp = send_ipc(sock, snap_cmd)
                        if resp.get("ok"):
                            print(f"Saved: {path}")
                    else:
                        print("Usage: snap <path> [<width>x<height>]")
                # Legacy commands for compatibility
                elif name == "cam" and len(parts) > 1:
                    try:
                        vals = [float(x) for x in cmd_args.split()]
                        if len(vals) >= 4:
                            send_ipc(sock, {"cmd": "set_terrain_camera",
                                "phi_deg": vals[0], "theta_deg": vals[1], 
                                "radius": vals[2], "fov_deg": vals[3]})
                    except ValueError:
                        print("Usage: cam <phi> <theta> <radius> <fov>")
                elif name == "sun" and len(parts) > 1:
                    try:
                        vals = [float(x) for x in cmd_args.split()]
                        if len(vals) >= 3:
                            send_ipc(sock, {"cmd": "set_terrain_sun",
                                "azimuth_deg": vals[0], "elevation_deg": vals[1], 
                                "intensity": vals[2]})
                    except ValueError:
                        print("Usage: sun <azimuth> <elevation> <intensity>")
                elif name == "pbr":
                    pbr_cmd = {"cmd": "set_terrain_pbr"}
                    if cmd_args.lower() in ("on", "true", "1"):
                        pbr_cmd["enabled"] = True
                    elif cmd_args.lower() in ("off", "false", "0"):
                        pbr_cmd["enabled"] = False
                    else:
                        # Parse key=value pairs
                        for pair in cmd_args.split():
                            if "=" not in pair:
                                continue
                            key, val = pair.split("=", 1)
                            key = key.lower().strip()
                            val = val.strip()
                            if key == "shadows":
                                pbr_cmd["shadow_technique"] = val
                            elif key == "exposure":
                                pbr_cmd["exposure"] = float(val)
                            elif key in ("ibl", "ibl_intensity"):
                                pbr_cmd["ibl_intensity"] = float(val)
                            elif key in ("normal", "normal_strength"):
                                pbr_cmd["normal_strength"] = float(val)
                            elif key in ("msaa",):
                                pbr_cmd["msaa"] = int(val)
                            elif key in ("shadow_res", "shadow_map_res"):
                                pbr_cmd["shadow_map_res"] = int(val)
                            elif key in ("hdr", "hdr_path"):
                                pbr_cmd["hdr_path"] = str(Path(val).resolve())
                    resp = send_ipc(sock, pbr_cmd)
                    if not resp.get("ok"):
                        print(f"Error: {resp.get('error')}")
                else:
                    print("Unknown command. Type 'set', 'params', 'snap', 'pbr', or 'quit'")
        except KeyboardInterrupt:
            pass
        
        sock.close()
        proc.terminate()
        proc.wait()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
