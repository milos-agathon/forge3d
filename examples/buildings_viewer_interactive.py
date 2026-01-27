#!/usr/bin/env python3
"""Interactive 3D buildings viewer - load terrain + buildings, orbit camera, take snapshots.

This script launches the interactive viewer and loads terrain with 3D building
overlays. Supports GeoJSON and CityJSON building formats with automatic roof
type inference and PBR material assignment.

**Building Sources:**

- **GeoJSON**: Load building footprints and extrude to 3D with configurable heights
- **CityJSON**: Load pre-modeled 3D buildings with LOD support
- **3D Tiles**: Load building metadata from 3D Tiles tileset

**Features (P4: 3D Buildings Pipeline):**

- Roof type inference from OSM tags (flat, gabled, hipped, pyramidal, etc.)
- PBR material presets (brick, glass, concrete, steel, wood)
- Material inference from building attributes
- CityJSON 1.1 parsing with LOD selection
- Draping buildings onto terrain surface

Usage:
    # Quick demo: Mount Fuji with sample buildings
    python examples/buildings_viewer_interactive.py --fuji

    # Or equivalently:
    python examples/buildings_viewer_interactive.py \\
        --dem assets/tif/Mount_Fuji_30m.tif \\
        --buildings assets/geojson/mount_fuji_buildings.geojson --pbr

    # CityJSON buildings with LOD support
    python examples/buildings_viewer_interactive.py \\
        --dem assets/tif/Mount_Fuji_30m.tif \\
        --cityjson assets/geojson/sample_buildings.city.json

    # Custom building height and material
    python examples/buildings_viewer_interactive.py \\
        --dem assets/tif/Mount_Fuji_30m.tif \\
        --buildings assets/geojson/mount_fuji_buildings.geojson \\
        --default-height 15 --material brick

    # With PBR rendering and shadows
    python examples/buildings_viewer_interactive.py \\
        --dem assets/tif/Mount_Fuji_30m.tif \\
        --buildings assets/geojson/mount_fuji_buildings.geojson \\
        --pbr --shadow-technique pcss

    # Take snapshot and exit
    python examples/buildings_viewer_interactive.py --fuji --snapshot fuji_buildings.png

Interactive Commands:
    camera phi=45 theta=30 radius=2000   Set camera position
    sun azimuth=135 elevation=45         Set sun direction
    buildings on/off                     Toggle building visibility
    snapshot output.png                  Take screenshot
    quit                                 Exit viewer

See docs/buildings.md for full documentation.
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
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

# Import shared utilities from forge3d package
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from forge3d.viewer_ipc import find_viewer_binary, send_ipc
from forge3d.interactive import run_interactive_loop

# P4: Buildings pipeline
from forge3d.buildings import (
    add_buildings,
    add_buildings_cityjson,
    Building,
    BuildingLayer,
    BuildingMaterial,
    infer_roof_type,
    material_from_name,
)


def create_demo_buildings() -> Tuple[List[List[float]], List[int]]:
    """Create demo building geometry when no file is provided.

    Returns a simple city block with varied building heights.
    """
    vertices = []
    indices = []

    # Grid of buildings
    grid_size = 5
    building_size = 50.0
    spacing = 70.0
    base_height = 10.0

    heights = [
        15, 25, 12, 30, 18,
        20, 35, 14, 22, 28,
        16, 40, 11, 24, 19,
        32, 13, 26, 17, 45,
        21, 15, 33, 20, 27,
    ]

    colors = [
        (0.55, 0.25, 0.18),  # Brick red
        (0.6, 0.58, 0.55),   # Concrete gray
        (0.04, 0.08, 0.12),  # Glass dark
        (0.5, 0.35, 0.2),    # Wood brown
        (0.88, 0.86, 0.82),  # Plaster white
    ]

    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            x = col * spacing - (grid_size * spacing) / 2
            z = row * spacing - (grid_size * spacing) / 2
            h = heights[idx]
            color = colors[idx % len(colors)]

            # Create box vertices (8 corners)
            base = len(vertices)
            half = building_size / 2

            # Bottom face
            for corner in [(-half, 0, -half), (half, 0, -half),
                          (half, 0, half), (-half, 0, half)]:
                vertices.append([
                    x + corner[0], corner[1], z + corner[2],
                    color[0], color[1], color[2], 1.0,
                    idx  # feature_id
                ])

            # Top face
            for corner in [(-half, h, -half), (half, h, -half),
                          (half, h, half), (-half, h, half)]:
                vertices.append([
                    x + corner[0], corner[1], z + corner[2],
                    color[0], color[1], color[2], 1.0,
                    idx
                ])

            # Indices for box faces (12 triangles = 36 indices)
            # Bottom
            indices.extend([base+0, base+2, base+1, base+0, base+3, base+2])
            # Top
            indices.extend([base+4, base+5, base+6, base+4, base+6, base+7])
            # Front
            indices.extend([base+0, base+1, base+5, base+0, base+5, base+4])
            # Back
            indices.extend([base+2, base+3, base+7, base+2, base+7, base+6])
            # Left
            indices.extend([base+0, base+4, base+7, base+0, base+7, base+3])
            # Right
            indices.extend([base+1, base+2, base+6, base+1, base+6, base+5])

    return vertices, indices


def buildings_to_overlay(
    layer: BuildingLayer,
    dem_bounds: Optional[Tuple[float, float, float, float]] = None,
    terrain_width: float = 1000.0,
    height_scale: float = 1.0,
    default_color: Tuple[float, float, float, float] = (0.6, 0.58, 0.55, 1.0),
    use_materials: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    """Convert BuildingLayer to vector overlay format.

    Args:
        layer: BuildingLayer with building geometries
        dem_bounds: DEM bounds as (west, south, east, north) for coordinate transform
        terrain_width: Width of terrain in world units (default: 1000)
        height_scale: Scale factor for building heights (default: 1.0)
        default_color: Default RGBA color for buildings
        use_materials: Whether to use material colors when available

    Returns vertices in 8-component format: [x, y, z, r, g, b, a, feature_id]
    and triangle indices.
    """
    vertices = []
    indices = []

    # Get bounds for coordinate transformation
    if dem_bounds is not None:
        min_x, min_y, max_x, max_y = dem_bounds
        dx = max_x - min_x if max_x != min_x else 1.0
        dy = max_y - min_y if max_y != min_y else 1.0
    else:
        min_x, min_y, max_x, max_y = 0, 0, 1, 1
        dx, dy = 1.0, 1.0

    for i, building in enumerate(layer.buildings):
        if building.positions.size == 0:
            continue

        # Get color from material or default
        if use_materials and building.material:
            r, g, b = building.material.albedo
            a = 1.0
        else:
            r, g, b, a = default_color

        # Add vertices with color and feature_id
        base_idx = len(vertices)
        pos = building.positions.reshape(-1, 3)

        for p in pos:
            # p[0] = lon (X), p[1] = height (Y), p[2] = lat (Z)
            if dem_bounds is not None:
                # Transform from geographic to local world space
                # X: lon -> normalized -> world_x
                u = (float(p[0]) - min_x) / dx
                world_x = u * terrain_width
                # Z: lat -> normalized -> world_z (flip for Y-up coord system)
                v = (max_y - float(p[2])) / dy
                world_z = v * terrain_width
                # Y: height scaled for visibility (will be added to terrain height if draping)
                world_y = float(p[1]) * height_scale
            else:
                world_x, world_y, world_z = float(p[0]), float(p[1]) * height_scale, float(p[2])

            vertices.append([
                float(world_x), float(world_y), float(world_z),
                float(r), float(g), float(b), float(a),
                int(i)  # feature_id
            ])

        # Add indices (offset by base)
        for idx in building.indices:
            indices.append(base_idx + int(idx))

    return vertices, indices


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive 3D buildings viewer (P4: Buildings Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Quick presets
    parser.add_argument("--fuji", action="store_true",
                        help="Quick start: Mount Fuji DEM + sample buildings with PBR")
    parser.add_argument("--rainier", action="store_true",
                        help="Quick start: Mount Rainier DEM + demo buildings with PBR")

    # Terrain options
    parser.add_argument("--dem", type=Path, default=None,
                        help="Path to GeoTIFF DEM file (optional for flat ground)")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")

    # Building source options
    bldg_group = parser.add_argument_group("Building Sources", "Load buildings from various formats")
    bldg_group.add_argument("--buildings", type=Path, default=None,
                            help="GeoJSON file with building footprints (e.g., assets/geojson/mount_fuji_buildings.geojson)")
    bldg_group.add_argument("--cityjson", type=Path, default=None,
                            help="CityJSON file with 3D building models (e.g., assets/geojson/sample_buildings.city.json)")
    bldg_group.add_argument("--demo", action="store_true",
                            help="Use demo buildings (grid of boxes)")

    # Building extrusion options
    ext_group = parser.add_argument_group("Extrusion Options", "Control building height and appearance")
    ext_group.add_argument("--default-height", type=float, default=10.0,
                           help="Default building height in meters (default: 10.0)")
    ext_group.add_argument("--height-key", type=str, default=None,
                           help="GeoJSON property key for building height")
    ext_group.add_argument("--material", type=str, default=None,
                           choices=["brick", "concrete", "glass", "steel", "wood", "plaster"],
                           help="Override material for all buildings")
    ext_group.add_argument("--color", type=float, nargs=4, default=None,
                           metavar=("R", "G", "B", "A"),
                           help="Override color for all buildings (RGBA 0-1)")

    # Draping options
    drape_group = parser.add_argument_group("Draping Options", "Control how buildings sit on terrain")
    drape_group.add_argument("--drape", action="store_true", default=True,
                             help="Drape buildings onto terrain surface (default: on)")
    drape_group.add_argument("--no-drape", action="store_false", dest="drape",
                             help="Don't drape - use original Z coordinates")
    drape_group.add_argument("--drape-offset", type=float, default=50.0,
                             help="Height above terrain when draped (default: 50.0)")
    drape_group.add_argument("--height-scale", type=float, default=10.0,
                             help="Scale factor for building heights (default: 10.0)")

    # PBR rendering options
    pbr_group = parser.add_argument_group("PBR Rendering", "High-quality rendering options")
    pbr_group.add_argument("--pbr", action="store_true",
                           help="Enable PBR rendering mode")
    pbr_group.add_argument("--shadow-technique", dest="shadow_technique",
                           choices=["none", "hard", "pcf", "pcss", "vsm", "evsm", "msm"],
                           default="pcss", help="Shadow technique (default: pcss)")
    pbr_group.add_argument("--exposure", type=float, default=1.0,
                           help="Exposure multiplier (default: 1.0)")
    pbr_group.add_argument("--ambient", type=float, default=0.1,
                           help="Ambient light intensity (default: 0.1)")

    # Sun options
    sun_group = parser.add_argument_group("Sun Lighting", "Directional sun light")
    sun_group.add_argument("--sun-azimuth", type=float, default=135.0,
                           help="Sun azimuth in degrees (default: 135.0)")
    sun_group.add_argument("--sun-elevation", type=float, default=45.0,
                           help="Sun elevation in degrees (default: 45.0)")

    args = parser.parse_args()

    # Handle quick-start presets
    script_dir = Path(__file__).parent.parent

    if args.fuji:
        args.dem = args.dem or script_dir / "assets" / "tif" / "Mount_Fuji_30m.tif"
        args.buildings = args.buildings or script_dir / "assets" / "geojson" / "mount_fuji_buildings.geojson"
        args.pbr = True
        args.shadow_technique = args.shadow_technique or "pcss"
        print("Using --fuji preset: Mount Fuji DEM + sample buildings")

    if args.rainier:
        args.dem = args.dem or script_dir / "assets" / "tif" / "dem_rainier.tif"
        args.demo = True
        args.pbr = True
        args.shadow_technique = args.shadow_technique or "pcss"
        print("Using --rainier preset: Mount Rainier DEM + demo buildings")

    # Validate building source
    if not args.buildings and not args.cityjson and not args.demo:
        print("No building source specified. Using --demo mode.")
        args.demo = True

    # Find viewer binary
    binary = find_viewer_binary()

    # Load buildings
    print("=" * 60)
    print("P4: 3D Buildings Pipeline")
    print("=" * 60)

    # Read DEM bounds for coordinate transformation
    dem_bounds = None
    terrain_width = 1000.0  # Default terrain width in world units

    if args.dem and HAS_RASTERIO:
        dem_path = args.dem if args.dem.is_absolute() else script_dir / args.dem
        if dem_path.exists():
            try:
                with rasterio.open(dem_path) as ds:
                    bounds = ds.bounds
                    dem_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
                    # Use larger dimension for terrain_width (square terrain)
                    terrain_width = float(max(ds.width, ds.height))
                    print(f"\nDEM: {dem_path.name}")
                    print(f"  Bounds: lon[{bounds.left:.4f}, {bounds.right:.4f}], lat[{bounds.bottom:.4f}, {bounds.top:.4f}]")
                    print(f"  Size: {ds.width}x{ds.height} pixels")
                    print(f"  Terrain width: {terrain_width:.0f} units")
            except Exception as e:
                print(f"Warning: Could not read DEM bounds: {e}")

    vertices = []
    indices = []
    layer = None

    if args.demo:
        print("\nLoading demo buildings (5x5 grid)...")
        vertices, indices = create_demo_buildings()
        print(f"  Created {len(vertices)} vertices, {len(indices)//3} triangles")

    elif args.cityjson:
        if not args.cityjson.exists():
            print(f"Error: CityJSON file not found: {args.cityjson}")
            return 1

        print(f"\nLoading CityJSON: {args.cityjson}")
        try:
            layer = add_buildings_cityjson(args.cityjson)
            print(f"  Buildings: {layer.building_count}")
            print(f"  Max LOD: {layer.max_lod}")
            if layer.crs_epsg:
                print(f"  CRS: EPSG:{layer.crs_epsg}")

            # Convert to overlay format
            override_color = tuple(args.color) if args.color else None
            vertices, indices = buildings_to_overlay(
                layer,
                dem_bounds=dem_bounds,
                terrain_width=terrain_width,
                height_scale=args.height_scale,
                default_color=override_color or (0.6, 0.58, 0.55, 1.0),
                use_materials=args.material is None and args.color is None,
            )
        except Exception as e:
            print(f"Error loading CityJSON: {e}")
            return 1

    elif args.buildings:
        if not args.buildings.exists():
            print(f"Error: GeoJSON file not found: {args.buildings}")
            return 1

        print(f"\nLoading GeoJSON buildings: {args.buildings}")
        try:
            layer = add_buildings(
                args.buildings,
                default_height=args.default_height,
                height_key=args.height_key,
            )
            print(f"  Buildings: {layer.building_count}")
            print(f"  Total vertices: {layer.total_vertices}")

            # If layer has geometry, convert to overlay
            if layer.total_vertices > 0:
                override_color = tuple(args.color) if args.color else None
                vertices, indices = buildings_to_overlay(
                    layer,
                    dem_bounds=dem_bounds,
                    terrain_width=terrain_width,
                    height_scale=args.height_scale,
                    default_color=override_color or (0.6, 0.58, 0.55, 1.0),
                    use_materials=args.material is None and args.color is None,
                )
            else:
                # Fallback: create boxes from building metadata
                print("  Note: GeoJSON geometry requires native module for extrusion")
                print("  Using simplified box representation")
                vertices, indices = create_demo_buildings()

        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Apply material override if specified
    if args.material and vertices:
        mat = material_from_name(args.material)
        r, g, b = mat.albedo
        print(f"  Material override: {args.material} (albedo={r:.2f},{g:.2f},{b:.2f})")
        for v in vertices:
            v[3], v[4], v[5] = r, g, b

    print(f"\nBuilding overlay: {len(vertices)} vertices, {len(indices)//3} triangles")
    print(f"  Height scale: {args.height_scale}x, Drape offset: {args.drape_offset}")

    # Check for large payload
    if len(vertices) > 50000:
        print(f"  Warning: Large vertex count may cause IPC issues")
        print(f"  Consider decimating or using LOD")

    # Start viewer
    print("\nStarting interactive viewer...")
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
    sock.settimeout(60.0)  # Extended timeout for large payloads

    # Load terrain if provided
    if args.dem:
        dem_path = args.dem.resolve()
        if not dem_path.exists():
            print(f"Error: DEM file not found: {dem_path}")
            sock.close()
            process.terminate()
            return 1

        print(f"Loading terrain: {dem_path}")
        resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
        if not resp.get("ok"):
            print(f"Failed to load terrain: {resp.get('error')}")
            sock.close()
            process.terminate()
            return 1

    # Set initial camera and sun
    ambient = args.ambient if args.pbr else 0.3
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 45.0, "theta": 35.0, "radius": 800.0, "fov": 45.0,
        "zscale": 0.1,
        "sun_azimuth": args.sun_azimuth,
        "sun_elevation": args.sun_elevation,
        "sun_intensity": 1.0,
        "ambient": ambient,
    })

    # Enable PBR if requested
    if args.pbr:
        pbr_cmd = {
            "cmd": "set_terrain_pbr",
            "enabled": True,
            "shadow_technique": args.shadow_technique,
            "exposure": args.exposure,
        }
        resp = send_ipc(sock, pbr_cmd)
        if resp.get("ok"):
            print(f"PBR enabled: shadows={args.shadow_technique}, exposure={args.exposure}")
        else:
            print(f"Warning: PBR config failed: {resp.get('error')}")

    # Add building overlay
    if vertices and indices:
        print(f"Adding building overlay ({len(vertices)} vertices)...")

        overlay_cmd = {
            "cmd": "add_vector_overlay",
            "name": "buildings",
            "vertices": vertices,
            "indices": indices,
            "primitive": "triangles",
            "drape": args.drape,
            "drape_offset": args.drape_offset,
            "line_width": 1.0,
        }

        resp = send_ipc(sock, overlay_cmd)
        if resp.get("ok"):
            print("  Buildings added successfully")
        else:
            print(f"  Warning: Failed to add buildings: {resp.get('error')}")

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
            print(f"\nSaved: {args.snapshot}")
            return 0
        return 1

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE 3D BUILDINGS VIEWER")
    print("=" * 60)
    print("\nWindow controls:")
    print("  Mouse drag     - Orbit camera")
    print("  Scroll wheel   - Zoom in/out")
    print("  W/S or Up/Down - Tilt camera")
    print("  A/D or L/R     - Rotate camera")
    print("  Q/E            - Zoom out/in")
    print("\nTerminal commands:")
    print("  camera phi=45 theta=30 radius=500")
    print("  sun azimuth=180 elevation=60")
    print("  snapshot buildings.png")
    print("  quit")

    # Build dem_info for interactive loop
    dem_info = None
    if layer and layer.bounds():
        bounds = layer.bounds()
        from forge3d.map_plate import BBox
        dem_info = {
            "bbox": BBox(
                west=bounds[0], south=bounds[1],
                east=bounds[3], north=bounds[4],
                crs="EPSG:4326"
            ),
            "domain": (bounds[2], bounds[5]),  # z min/max as elevation
        }

    run_interactive_loop(sock, process, title="3D BUILDINGS VIEWER", dem_info=dem_info)

    sock.close()
    process.terminate()
    process.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
