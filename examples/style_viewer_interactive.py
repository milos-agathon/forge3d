#!/usr/bin/env python3
"""Interactive style viewer - apply Mapbox styles to vector overlays.

This script demonstrates Mapbox Style Spec import in the interactive viewer.
Load a style.json file and apply it to vector overlays on terrain.

**User Journeys:**

1. **Interactive Session:** Load style and vectors, orbit camera, adjust
   via terminal, call `snap` to save.
2. **One-shot Snapshot:** Load style and take snapshot with `--snapshot out.png`.

Usage:
    # Load style and vector data interactively
    python examples/style_viewer_interactive.py --dem terrain.tif \\
        --style mapbox-streets.json --vectors roads.geojson

    # One-shot snapshot
    python examples/style_viewer_interactive.py --dem terrain.tif \\
        --style mapbox-streets.json --vectors roads.geojson \\
        --snapshot output.png

    # Apply style to specific source-layer
    python examples/style_viewer_interactive.py --dem terrain.tif \\
        --style mapbox-streets.json --vectors roads.geojson \\
        --source-layer road

Interactive Commands:
    style <path>              Load a new style.json
    layer <id>                Show info about a style layer
    layers                    List all style layers
    snap <path>               Take screenshot
    camera phi=N theta=N      Set camera position
    quit                      Exit viewer

See docs/api/style.md for Style Spec documentation.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from forge3d.viewer_ipc import find_viewer_binary, send_ipc
from forge3d.style import (
    load_style,
    layer_to_vector_style,
    layer_to_label_style,
    StyleSpec,
)

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
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


def generate_nodata_mask(dem_path: Path, output_path: Path) -> bool:
    """Generate a transparency mask from DEM nodata values.
    
    Creates an RGBA image where:
    - Valid DEM pixels -> white with alpha=255
    - NoData pixels -> black with alpha=0
    """
    if not HAS_RASTERIO or not HAS_NUMPY or not HAS_PIL:
        print("  Warning: rasterio, numpy, or PIL not available for mask generation")
        return False
    
    try:
        with rasterio.open(dem_path) as src:
            data = src.read(1)
            nodata = src.nodata
            
            # Create mask: 255 where valid, 0 where nodata
            if nodata is not None:
                mask = np.where(
                    (data == nodata) | ~np.isfinite(data),
                    0, 255
                ).astype(np.uint8)
            else:
                # No nodata defined - check for NaN/Inf
                mask = np.where(np.isfinite(data), 255, 0).astype(np.uint8)
            
            # Create RGBA: white where valid, transparent where nodata
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = 255  # R
            rgba[:, :, 1] = 255  # G
            rgba[:, :, 2] = 255  # B
            rgba[:, :, 3] = mask  # A = mask
            
            img = Image.fromarray(rgba, 'RGBA')
            img.save(output_path)
            print(f"  Generated nodata mask: {output_path.name}")
            return True
    except Exception as e:
        print(f"  Warning: Failed to generate mask: {e}")
        return False


def load_vectors_from_file(
    vector_path: Path,
    dem_path: Path,
    color: Tuple[float, float, float, float],
    line_width: float = 15.0
) -> Tuple[List[List[float]], List[int]]:
    """Load vector geometries and convert to triangle vertices.
    
    Returns:
        vertices: List of [x, y, z, r, g, b, a, feature_id]
        indices: List of triangle indices
    """
    if not HAS_GPD:
        print("  Warning: geopandas not installed. Install with: pip install geopandas")
        return [], []
    
    gdf = gpd.read_file(str(vector_path))
    print(f"  Loaded {len(gdf)} features from {vector_path.name}")
    
    # Get terrain dimensions
    terrain_width = 1000.0
    dem_bounds = None
    
    if HAS_RASTERIO and dem_path.exists():
        with rasterio.open(dem_path) as dem:
            terrain_width = float(max(dem.width, dem.height))
            dem_bounds = dem.bounds
            dem_crs = dem.crs
            
            # Reproject vector to DEM CRS if needed
            if gdf.crs and dem_crs and gdf.crs != dem_crs:
                print(f"  Reprojecting from {gdf.crs} to {dem_crs}")
                gdf = gdf.to_crs(dem_crs)
    
    # Get bounds
    if dem_bounds:
        min_x, min_y = dem_bounds.left, dem_bounds.bottom
        max_x, max_y = dem_bounds.right, dem_bounds.top
    else:
        bounds = gdf.total_bounds
        min_x, min_y, max_x, max_y = bounds
    
    vertices = []
    indices = []
    half_w = line_width / 2.0
    color_list = list(color)
    
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        lines = []
        if geom.geom_type == 'LineString':
            lines = [list(geom.coords)]
        elif geom.geom_type == 'MultiLineString':
            lines = [list(line.coords) for line in geom.geoms]
        elif geom.geom_type == 'Polygon':
            lines = [list(geom.exterior.coords)]
        elif geom.geom_type == 'MultiPolygon':
            lines = [list(poly.exterior.coords) for poly in geom.geoms]
        
        for coords in lines:
            if len(coords) < 2:
                continue
            
            # Convert to world space
            world_coords = []
            for pt in coords:
                x, y = pt[0], pt[1]
                u = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
                v = (max_y - y) / (max_y - min_y) if max_y != min_y else 0.5
                world_coords.append((u * terrain_width, v * terrain_width))
            
            # Generate strip vertices
            strip_verts = []
            for i in range(len(world_coords)):
                wx, wz = world_coords[i]
                
                # Calculate perpendicular
                if i == 0:
                    dx = world_coords[1][0] - wx
                    dz = world_coords[1][1] - wz
                elif i == len(world_coords) - 1:
                    dx = wx - world_coords[i-1][0]
                    dz = wz - world_coords[i-1][1]
                else:
                    dx = world_coords[i+1][0] - world_coords[i-1][0]
                    dz = world_coords[i+1][1] - world_coords[i-1][1]
                
                length = (dx*dx + dz*dz) ** 0.5
                if length < 0.001:
                    continue
                dx, dz = dx/length, dz/length
                
                # Perpendicular
                px, pz = -dz * half_w, dx * half_w
                strip_verts.append((wx - px, wz - pz))
                strip_verts.append((wx + px, wz + pz))
            
            if len(strip_verts) < 4:
                continue
            
            # Add vertices
            base = len(vertices)
            for vx, vz in strip_verts:
                vertices.append([vx, 0.0, vz] + color_list + [0])
            
            # Add triangle indices
            for i in range(len(strip_verts) // 2 - 1):
                l0 = base + i * 2
                r0 = base + i * 2 + 1
                l1 = base + (i+1) * 2
                r1 = base + (i+1) * 2 + 1
                indices.extend([l0, r0, l1, r0, r1, l1])
    
    print(f"  Generated {len(vertices)} vertices, {len(indices)//3} triangles")
    return vertices, indices


def _show_layer_info(layer) -> str:
    """Format layer info for display."""
    lines = [f"Layer: {layer.id}"]
    lines.append(f"  Type: {layer.layer_type}")
    if layer.source_layer:
        lines.append(f"  Source-layer: {layer.source_layer}")
    if layer.filter:
        lines.append(f"  Filter: {json.dumps(layer.filter)}")
    if layer.minzoom or layer.maxzoom:
        lines.append(f"  Zoom: {layer.minzoom or 0} - {layer.maxzoom or 22}")
    
    # Show paint props
    paint = layer.paint
    paint_items = []
    if paint.fill_color:
        paint_items.append(f"fill-color: {paint.fill_color}")
    if paint.line_color:
        paint_items.append(f"line-color: {paint.line_color}")
    if paint.line_width:
        paint_items.append(f"line-width: {paint.line_width}")
    if paint.text_color:
        paint_items.append(f"text-color: {paint.text_color}")
    if paint_items:
        lines.append(f"  Paint: {', '.join(paint_items)}")
    
    return "\n".join(lines)


def run_style_interactive_loop(sock, process, args, spec: StyleSpec):
    """Run interactive command loop with style support."""
    print(f"\n{'='*60}")
    print(f" STYLE VIEWER")
    print(f"{'='*60}")
    print(f"\nLoaded style: {spec.name} ({len(spec.layers)} layers)")
    print(f"  Fill layers: {len(spec.fill_layers())}")
    print(f"  Line layers: {len(spec.line_layers())}")
    print(f"  Symbol layers: {len(spec.symbol_layers())}")
    
    print("\nCommands:")
    print("  style <path>       - Load new style.json")
    print("  layers             - List all layers")
    print("  layer <id>         - Show layer details")
    print("  snap <path>        - Take screenshot")
    print("  camera phi=N ...   - Set camera")
    print("  quit               - Exit")
    print()
    
    current_spec = spec
    
    try:
        while True:
            if process.poll() is not None:
                print("Viewer closed")
                break
            
            try:
                cmd = input("> ").strip()
            except EOFError:
                break
            
            if not cmd:
                continue
            
            parts = cmd.split(maxsplit=1)
            verb = parts[0].lower()
            
            if verb in ("quit", "exit", "q"):
                break
            
            elif verb == "style":
                if len(parts) < 2:
                    print("Usage: style <path>")
                    continue
                try:
                    new_spec = load_style(Path(parts[1]))
                    current_spec = new_spec
                    print(f"Loaded: {new_spec.name} ({len(new_spec.layers)} layers)")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif verb == "layers":
                for layer in current_spec.layers:
                    vis = "✓" if layer.is_visible() else "✗"
                    print(f"  [{vis}] {layer.id} ({layer.layer_type})")
            
            elif verb == "layer":
                if len(parts) < 2:
                    print("Usage: layer <id>")
                    continue
                layer = current_spec.layer_by_id(parts[1])
                if layer:
                    print(_show_layer_info(layer))
                else:
                    print(f"Layer not found: {parts[1]}")
            
            elif verb == "snap":
                path = parts[1] if len(parts) > 1 else "snapshot.png"
                resp = send_ipc(sock, {"cmd": "snapshot", "path": str(Path(path).resolve())})
                if resp.get("ok"):
                    print(f"Saved: {path}")
                else:
                    print(f"Failed: {resp.get('error')}")
            
            elif verb == "camera":
                params = {}
                for match in re.finditer(r"(\w+)=([0-9.]+)", cmd):
                    params[match.group(1)] = float(match.group(2))
                if params:
                    resp = send_ipc(sock, {"cmd": "set_terrain", **params})
                    if resp.get("ok"):
                        print(f"Camera: {params}")
                    else:
                        print(f"Failed: {resp.get('error')}")
            
            else:
                print(f"Unknown: {cmd}")
                
    except KeyboardInterrupt:
        print("\nInterrupted")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive style viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dem", type=Path, required=True,
                        help="Path to GeoTIFF DEM file")
    parser.add_argument("--style", type=Path, required=True,
                        help="Path to Mapbox style.json file")
    parser.add_argument("--vectors", type=Path,
                        help="Path to vector file (GeoJSON, GPKG, SHP)")
    parser.add_argument("--source-layer", type=str,
                        help="Filter to specific source-layer name")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot and exit")
    
    args = parser.parse_args()
    
    # Load style
    if not args.style.exists():
        print(f"Error: Style not found: {args.style}")
        return 1
    
    try:
        spec = load_style(args.style)
        print(f"Loaded style: {spec.name}")
    except Exception as e:
        print(f"Error loading style: {e}")
        return 1
    
    # Validate DEM
    dem_path = args.dem.resolve()
    if not dem_path.exists():
        print(f"Error: DEM not found: {dem_path}")
        return 1
    
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
    
    # Connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    
    # Load terrain
    print(f"Loading terrain: {dem_path}")
    resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
    if not resp.get("ok"):
        print(f"Failed: {resp.get('error')}")
        sock.close()
        process.terminate()
        return 1
    
    # Set camera
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 30.0, "theta": 45.0, "radius": 3800.0, "fov": 30.0,
        "zscale": 0.1,
    })
    
    # Generate and load nodata mask to hide areas outside DEM
    import tempfile
    mask_path = Path(tempfile.gettempdir()) / "dem_mask.png"
    if generate_nodata_mask(dem_path, mask_path):
        # Load mask as overlay
        resp = send_ipc(sock, {
            "cmd": "load_overlay",
            "name": "nodata_mask",
            "path": str(mask_path),
            "opacity": 1.0,
        })
        if resp.get("ok"):
            print("  ✓ Loaded nodata mask overlay")
            # Set solid=false to hide terrain where mask alpha is 0
            send_ipc(sock, {"cmd": "set_overlay_solid", "solid": False})
            print("  ✓ Set solid=false (hiding nodata areas)")
        else:
            print(f"  Warning: Failed to load mask: {resp.get('error')}")
    
    # Load vectors if provided
    if args.vectors:
        vector_path = args.vectors.resolve()
        if not vector_path.exists():
            print(f"Warning: Vector file not found: {vector_path}")
        else:
            print(f"Loading vectors: {vector_path}")
            
            # Get style for vectors from first matching layer
            color = (0.9, 0.3, 0.1, 1.0)  # Default orange
            line_width = 15.0
            
            for layer in spec.layers:
                if layer.layer_type in ("line", "fill", "circle"):
                    vs = layer_to_vector_style(layer)
                    if vs:
                        color = vs.stroke_color if layer.layer_type == "line" else vs.fill_color
                        line_width = vs.stroke_width if vs.stroke_width > 0 else 15.0
                        print(f"  Using style from layer: {layer.id}")
                        break
            
            # Load and convert vectors to vertices
            vertices, indices = load_vectors_from_file(
                vector_path, dem_path, color, line_width
            )
            
            if vertices:
                # Send via IPC
                resp = send_ipc(sock, {
                    "cmd": "add_vector_overlay",
                    "name": vector_path.stem,
                    "vertices": vertices,
                    "indices": indices,
                    "primitive": "triangles",
                    "drape": True,
                    "drape_offset": 5.0,
                    "line_width": line_width,
                    "point_size": 5.0,
                    "z_order": 0,
                })
                if resp.get("ok"):
                    print(f"  ✓ Added vector overlay: {vector_path.stem}")
                else:
                    print(f"  Warning: {resp.get('error', 'Failed to add overlay')}")
            else:
                print(f"  Warning: No geometry loaded from {vector_path.name}")
    
    # Snapshot mode
    if args.snapshot:
        time.sleep(0.5)
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
            print(f"Saved: {args.snapshot}")
            return 0
        return 1
    
    # Interactive mode
    run_style_interactive_loop(sock, process, args, spec)
    
    sock.close()
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
