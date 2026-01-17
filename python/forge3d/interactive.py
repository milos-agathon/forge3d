"""Interactive command loop utilities for terrain viewer examples.

This module provides reusable components for building interactive
terminal-based viewer control interfaces.
"""

from __future__ import annotations

import re
import socket
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .viewer_ipc import send_ipc


# Common key mapping for terrain parameters
TERRAIN_KEY_MAP = {
    "phi": "phi",
    "theta": "theta",
    "radius": "radius",
    "fov": "fov",
    "sun_az": "sun_azimuth",
    "sun_azimuth": "sun_azimuth",
    "sun_el": "sun_elevation",
    "sun_elevation": "sun_elevation",
    "intensity": "sun_intensity",
    "sun_intensity": "sun_intensity",
    "ambient": "ambient",
    "zscale": "zscale",
    "z_scale": "zscale",
    "shadow": "shadow",
    "bg": "background",
    "background": "background",
    "water": "water_level",
    "water_level": "water_level",
    "water_color": "water_color",
}


def parse_set_command(args: str, key_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Parse 'set key=value key=value ...' into IPC params.
    
    Args:
        args: Command arguments string, e.g. "phi=45 theta=60"
        key_map: Optional custom key mapping, defaults to TERRAIN_KEY_MAP
        
    Returns:
        Dictionary with cmd="set_terrain" and parsed parameters
    """
    if key_map is None:
        key_map = TERRAIN_KEY_MAP
    
    params: Dict[str, Any] = {"cmd": "set_terrain"}
    
    for pair in args.split():
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.lower().strip()
        val = val.strip()
        
        ipc_key = key_map.get(key, key)
        
        # Parse value - check for RGB color (comma-separated)
        if "," in val:
            params[ipc_key] = [float(x) for x in val.split(",")]
        else:
            try:
                params[ipc_key] = float(val)
            except ValueError:
                params[ipc_key] = val
    
    return params


def handle_map_plate_command(
    sock: socket.socket,
    args: str,
    dem_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """Handle map_plate command: map_plate <path> [title="..."] [width=1600] [height=1200]
    
    Captures a high-quality snapshot and composes it into a map plate with
    title, legend, scale bar, and north arrow.
    
    Args:
        sock: Connected socket
        args: Command arguments
        dem_info: Optional DEM metadata (bbox, domain, meters_per_pixel)
        
    Returns:
        True if command was handled
    """
    from PIL import Image
    from .map_plate import MapPlate, MapPlateConfig, BBox
    from .legend import Legend, LegendConfig
    from .scale_bar import ScaleBar, ScaleBarConfig
    from .north_arrow import NorthArrow, NorthArrowConfig
    
    if not args:
        print("Usage: map_plate <output.png> [title=\"...\"] [width=1600] [height=1200]")
        return True
    
    # Parse arguments
    parts = args.split()
    output_path = Path(parts[0]).resolve()
    
    # Default settings
    title = "3D Terrain Map"
    plate_width = 1600
    plate_height = 1200
    
    # Parse key=value or title="..." arguments
    i = 1
    while i < len(parts):
        part = parts[i]
        if part.startswith('title="'):
            # Handle quoted title
            title_parts = [part[7:]]
            while i < len(parts) - 1 and not title_parts[-1].endswith('"'):
                i += 1
                title_parts.append(parts[i])
            title = " ".join(title_parts).rstrip('"')
        elif "=" in part:
            key, val = part.split("=", 1)
            if key == "width":
                plate_width = int(val)
            elif key == "height":
                plate_height = int(val)
            elif key == "title":
                title = val.strip('"')
        i += 1
    
    # Calculate map region size (leave margins for legend, title, etc.)
    margin_top, margin_right, margin_bottom, margin_left = 70, 200, 80, 40
    map_width = plate_width - margin_left - margin_right
    map_height = plate_height - margin_top - margin_bottom
    
    # Take high-res snapshot to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    print(f"Capturing {map_width}x{map_height} snapshot...")
    snap_cmd = {
        "cmd": "snapshot",
        "path": tmp_path,
        "width": map_width,
        "height": map_height,
    }
    resp = send_ipc(sock, snap_cmd)
    if not resp.get("ok"):
        print(f"Snapshot failed: {resp.get('error')}")
        return True
    
    # Wait for file to be written (viewer writes asynchronously)
    import time
    tmp_file = Path(tmp_path)
    for _ in range(50):  # Wait up to 5 seconds
        time.sleep(0.1)
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
            break
    else:
        print(f"Snapshot timed out - file not created")
        return True
    
    # Small additional delay to ensure file is fully written
    time.sleep(0.2)
    
    # Load the snapshot
    try:
        snap_img = Image.open(tmp_path)
        snap_rgba = np.array(snap_img)
        tmp_file.unlink()  # Clean up temp file
    except Exception as e:
        print(f"Failed to load snapshot: {e}")
        return True
    
    print("Composing map plate...")
    
    # Create map plate
    plate = MapPlate(MapPlateConfig(
        width=plate_width,
        height=plate_height,
        margin=(margin_top, margin_right, margin_bottom, margin_left),
        background=(245, 245, 245, 255),
    ))
    
    # Use DEM info if available, otherwise use defaults
    if dem_info:
        bbox = dem_info.get("bbox", BBox(west=-121.9, south=46.7, east=-121.6, north=46.9))
        domain = dem_info.get("domain", (0.0, 4000.0))
        mpp = dem_info.get("meters_per_pixel")
    else:
        bbox = BBox(west=-121.9, south=46.7, east=-121.6, north=46.9)
        domain = (0.0, 4000.0)
        mpp = None
    
    plate.set_map_region(snap_rgba, bbox)
    plate.add_title(title, font_size=28)
    
    # Extract terrain colors from the snapshot, excluding background
    h, w = snap_rgba.shape[:2]
    img_float = snap_rgba[:, :, :3].astype(np.float32) / 255.0
    
    # Detect background color from top corners (sky)
    # We ignore bottom corners as they might be terrain
    corner_size = 30
    top_corners = [
        img_float[:corner_size, :corner_size],           # top-left
        img_float[:corner_size, -corner_size:],          # top-right
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in top_corners], axis=0)
    
    # Calculate distance from background color
    diff = img_float - bg_color
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Check if background is "blue sky" (B > R and B > G)
    bg_is_blue = (bg_color[2] > bg_color[0] + 0.05) and (bg_color[2] > bg_color[1] + 0.05)
    
    if bg_is_blue:
        # If sky is blue, we can use a simpler color distance
        # But we must be careful not to exclude water if it looks like sky
        # For map plates, we assume main subject is terrain
        is_terrain = dist_sq > 0.04  # 0.2^2
    else:
        # If sky is not clearly blue (e.g. white/grey), it's harder.
        # Use position heuristic: terrain is usually not at the very top
        # and has different texture/variance.
        # For now, use a tighter threshold
        is_terrain = dist_sq > 0.01
        
    # Explicitly preserve SNOW: High brightness, low saturation
    # (unless sky is also exactly white, but even then snow has shading)
    lum = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
    sat = np.max(img_float, axis=2) - np.min(img_float, axis=2)
    is_snow = (lum > 0.8) & (sat < 0.1)
    
    # If background is NOT white (high lum, low sat), then keep snow
    bg_lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    bg_sat = np.max(bg_color) - np.min(bg_color)
    bg_is_white = (bg_lum > 0.9) and (bg_sat < 0.1)
    
    if not bg_is_white:
        is_terrain = is_terrain | is_snow

    # Get all terrain pixel colors
    terrain_pixels = img_float[is_terrain]
    
    if len(terrain_pixels) < 1000:
        # Fallback to center crop if masking failed
        h, w = img_float.shape[:2]
        terrain_pixels = img_float[h//3:2*h//3, w//3:2*w//3].reshape(-1, 3)

    # Sort terrain colors by luminance (dark=low elevation, bright=high/snow)
    luminance = 0.299 * terrain_pixels[:, 0] + 0.587 * terrain_pixels[:, 1] + 0.114 * terrain_pixels[:, 2]
    sorted_idx = np.argsort(luminance)
    sorted_pixels = terrain_pixels[sorted_idx]
    sorted_lum = luminance[sorted_idx]
    
    # "Luminance-Linear Sampling"
    # Instead of binning (which is biased by frequency), we define N target luminances
    # linearly spaced from the absolute darkest to absolute brightest terrain pixel.
    # We then find the specific pixels that match these luminance values.
    # This forces the legend to show the full spectral range regardless of how rare a color is.
    
    # Robust min/max: use 0.1% and 99.9% percentiles to avoid outliers
    n_pix = len(sorted_pixels)
    if n_pix > 1000:
        idx_min = int(n_pix * 0.001)
        idx_max = int(n_pix * 0.999)
        l_min = sorted_lum[idx_min]
        l_max = sorted_lum[idx_max]
    elif n_pix > 0:
        l_min = sorted_lum[0]
        l_max = sorted_lum[-1]
    else:
        l_min, l_max = 0.0, 1.0
        
    if l_max <= l_min:
        l_max = l_min + 1e-6
        
    n_stops = 64  # High resolution sampling
    target_lums = np.linspace(l_min, l_max, n_stops)
    stops_colors = []
    
    # For each target luminance, find the index in sorted array using binary search
    # This is fast and accurate
    import bisect
    for t in target_lums:
        idx = bisect.bisect_left(sorted_lum, t)
        idx = min(idx, n_pix - 1)
        
        # Smooth: take average of a small window around the found index
        # Window size inversely proportional to slope? Just use fixed small window.
        w_size = max(1, n_pix // 1000) # e.g. 1000 pixels -> window 1
        s = max(0, idx - w_size)
        e = min(n_pix, idx + w_size + 1)
        
        stops_colors.append(np.mean(sorted_pixels[s:e], axis=0))
    
    stops_colors = np.array(stops_colors)
    
    # Interpolate these stops to 256 values for the legend texture
    legend_rgba = np.zeros((256, 4), dtype=np.float32)
    x_stops = np.linspace(0, 1, n_stops)
    x_vals = np.linspace(0, 1, 256)
    
    for c in range(3):
        legend_rgba[:, c] = np.interp(x_vals, x_stops, stops_colors[:, c])
    legend_rgba[:, 3] = 1.0
    
    # Apply slight smoothing to the legend to remove banding
    # (since we sampled specific pixels)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    for c in range(3):
        legend_rgba[:, c] = np.convolve(legend_rgba[:, c], kernel, mode='same')
        # Fix edges
        legend_rgba[:kernel_size, c] = legend_rgba[kernel_size, c]
        legend_rgba[-kernel_size:, c] = legend_rgba[-kernel_size-1, c]

    legend = Legend(
        colormap_rgba=legend_rgba,
        domain=domain,
        config=LegendConfig(
            bar_height=350,
            bar_width=30,
            tick_count=6,
            label_format="{:.0f}",
            label_suffix=" m",
            title="Elevation",
            title_font_size=16,
            font_size=13,
        ),
    )
    plate.add_legend(legend.render(), position="right")
    
    # Generate scale bar
    if mpp is not None:
        meters_per_px = mpp
    else:
        meters_per_px = ScaleBar.compute_meters_per_pixel(bbox, map_width)
    
    scale_bar = ScaleBar(
        meters_per_pixel=meters_per_px,
        config=ScaleBarConfig(
            units="km",
            style="alternating",
            width_px=200,
            divisions=4,
        ),
    )
    plate.add_scale_bar(scale_bar.render(), position="bottom-left")
    
    # Generate north arrow (larger and more visible)
    north_arrow = NorthArrow(NorthArrowConfig(
        style="compass",
        size=80,
        rotation_deg=0.0,
    ))
    plate.add_north_arrow(north_arrow.render(), position="top-right")
    
    # Export
    plate.export_png(output_path)
    print(f"Map plate saved: {output_path}")
    
    return True


def handle_snapshot_command(
    sock: socket.socket,
    args: str,
    post_snapshot_callback: Optional[Callable[[Path], None]] = None,
) -> bool:
    """Handle snapshot command: snap <path> [<width>x<height>]
    
    Args:
        sock: Connected socket
        args: Command arguments
        post_snapshot_callback: Optional callback after successful snapshot
        
    Returns:
        True if command was handled (even if failed)
    """
    if not args:
        print("Usage: snap <path> [<width>x<height>]")
        return True
    
    parts = args.split()
    path = str(Path(parts[0]).resolve())
    snap_cmd: Dict[str, Any] = {"cmd": "snapshot", "path": path}
    
    if len(parts) == 2:
        m = re.match(r"^(\d+)x(\d+)$", parts[1].lower())
        if m:
            snap_cmd["width"] = int(m.group(1))
            snap_cmd["height"] = int(m.group(2))
        else:
            print("Usage: snap <path> [<width>x<height>]")
            return True
    elif len(parts) == 3:
        try:
            snap_cmd["width"] = int(parts[1])
            snap_cmd["height"] = int(parts[2])
        except ValueError:
            print("Usage: snap <path> [<width>x<height>]")
            return True
    elif len(parts) > 3:
        print("Usage: snap <path> [<width>x<height>]")
        return True
    
    resp = send_ipc(sock, snap_cmd)
    if resp.get("ok"):
        print(f"Saved: {path}")
        if post_snapshot_callback:
            post_snapshot_callback(Path(path))
    else:
        print(f"Snapshot failed: {resp.get('error')}")
    
    return True


def handle_pbr_command(sock: socket.socket, args: str) -> bool:
    """Handle PBR command: pbr on/off or pbr key=value ...
    
    Args:
        sock: Connected socket
        args: Command arguments
        
    Returns:
        True if command was handled
    """
    pbr_cmd: Dict[str, Any] = {"cmd": "set_terrain_pbr"}
    
    if args.lower() in ("on", "true", "1"):
        pbr_cmd["enabled"] = True
    elif args.lower() in ("off", "false", "0"):
        pbr_cmd["enabled"] = False
    else:
        # Parse key=value pairs
        for pair in args.split():
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
            elif key == "msaa":
                pbr_cmd["msaa"] = int(val)
            elif key in ("shadow_res", "shadow_map_res"):
                pbr_cmd["shadow_map_res"] = int(val)
            elif key in ("hdr", "hdr_path"):
                pbr_cmd["hdr_path"] = str(Path(val).resolve())
    
    resp = send_ipc(sock, pbr_cmd)
    if not resp.get("ok"):
        print(f"Error: {resp.get('error')}")
    
    return True


def print_terrain_help(title: str = "TERRAIN VIEWER") -> None:
    """Print standard help text for terrain viewer interactive mode.
    
    Args:
        title: Title to display in header
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print()
    print("Terminal commands (set any combination of parameters):")
    print("  set phi=45 theta=60 radius=2000 fov=55")
    print("  set sun_az=135 sun_el=45 intensity=1.5 ambient=0.3")
    print("  set zscale=2.0 shadow=0.5")
    print("  set background=0.2,0.3,0.5")
    print("  set water=1500 water_color=0.1,0.3,0.5")
    print()
    print("Snapshot commands:")
    print("  snap <path> [<width>x<height>]  - Take raw screenshot")
    print("  map_plate <path> [title=\"...\"] [width=1600] [height=1200]")
    print("                                  - Export with legend, scale bar, north arrow")
    print()
    print("Other commands:")
    print("  params         - Show current parameters")
    print("  pbr on/off     - Toggle PBR rendering mode")
    print("  pbr shadows=pcss exposure=1.5 ibl=2.0")
    print("  quit           - Close viewer")
    print("=" * 60 + "\n")


def run_interactive_loop(
    sock: socket.socket,
    process,
    title: str = "TERRAIN VIEWER",
    extra_commands: Optional[Dict[str, Callable[[socket.socket, str], bool]]] = None,
    post_snapshot_callback: Optional[Callable[[Path], None]] = None,
    dem_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Run interactive command loop for terrain viewer.
    
    Handles standard commands: quit, set, params, snap, map_plate, pbr, cam, sun
    
    Args:
        sock: Connected socket
        process: Viewer subprocess (for poll check)
        title: Title for help display
        extra_commands: Dict of command_name -> handler(sock, args) for custom commands
        post_snapshot_callback: Optional callback after successful snapshot
        dem_info: Optional DEM metadata for map plate (bbox, domain, meters_per_pixel)
    """
    print_terrain_help(title)
    
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
            
            # Check for quit
            if name in ("quit", "exit", "q"):
                send_ipc(sock, {"cmd": "close"})
                break
            
            # Check extra commands first
            if extra_commands and name in extra_commands:
                extra_commands[name](sock, cmd_args)
                continue
            
            # Standard commands
            if name == "set":
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
                handle_snapshot_command(sock, cmd_args, post_snapshot_callback)
            
            elif name == "map_plate":
                handle_map_plate_command(sock, cmd_args, dem_info)
            
            elif name == "pbr":
                handle_pbr_command(sock, cmd_args)
            
            # Legacy camera command
            elif name == "cam" and cmd_args:
                try:
                    vals = [float(x) for x in cmd_args.split()]
                    if len(vals) >= 4:
                        send_ipc(sock, {
                            "cmd": "set_terrain_camera",
                            "phi_deg": vals[0],
                            "theta_deg": vals[1],
                            "radius": vals[2],
                            "fov_deg": vals[3],
                        })
                except ValueError:
                    print("Usage: cam <phi> <theta> <radius> <fov>")
            
            # Legacy sun command
            elif name == "sun" and cmd_args:
                try:
                    vals = [float(x) for x in cmd_args.split()]
                    if len(vals) >= 3:
                        send_ipc(sock, {
                            "cmd": "set_terrain_sun",
                            "azimuth_deg": vals[0],
                            "elevation_deg": vals[1],
                            "intensity": vals[2],
                        })
                except ValueError:
                    print("Usage: sun <azimuth> <elevation> <intensity>")
            
            else:
                print("Unknown command. Type 'set', 'params', 'snap', 'map_plate', 'pbr', or 'quit'")
    
    except KeyboardInterrupt:
        pass
