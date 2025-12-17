#!/usr/bin/env python3
"""Interactive terrain viewer - load DEM, orbit camera, take snapshots.

This script launches the interactive viewer and loads a terrain DEM for
real-time viewing. You can orbit the camera around the terrain, adjust
lighting, and take snapshots when you're happy with the view.

**Rendering Modes:**

- **Legacy mode** (default): Simple Lambertian shading with height-based colors
- **PBR mode** (`--pbr`): Enhanced Blinn-Phong lighting, ACES tonemapping,
  height+slope materials, configurable exposure

Usage:
    # Basic usage - load terrain with legacy rendering
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif
    
    # PBR mode - enhanced lighting and materials
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr
    
    # PBR with custom exposure (brighter)
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr --exposure 1.5
    
    # PBR with all options
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr --exposure 1.2 --normal-strength 1.5 --ibl-intensity 1.0
    
    # Take automatic snapshot and exit
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --snapshot output.png
    
    # Custom window size
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --width 1920 --height 1080

Interactive Commands:
    camera phi=45 theta=30 radius=2000   Set camera position
    sun azimuth=135 elevation=45         Set sun direction
    pbr on                               Enable PBR mode
    pbr off                              Disable PBR mode
    pbr exposure=2.0                     Adjust PBR exposure
    snapshot output.png                  Take screenshot
    quit                                 Exit viewer

See docs/pbm_pom_viewer.md for full documentation.
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


def find_viewer_binary() -> str:
    """Find the interactive_viewer binary."""
    candidates = [
        Path(__file__).parent.parent / "target" / "release" / "interactive_viewer",
        Path(__file__).parent.parent / "target" / "debug" / "interactive_viewer",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive terrain viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dem", type=Path, required=True,
                        help="Path to GeoTIFF DEM file")
    parser.add_argument("--width", type=int, default=3840, help="Window width")
    parser.add_argument("--height", type=int, default=2160, help="Window height")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")
    
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
    
    # Sun/lighting options
    sun_group = parser.add_argument_group("Sun Lighting", "Directional sun light parameters")
    sun_group.add_argument("--sun-azimuth", type=float, default=135.0,
                           help="Sun azimuth angle in degrees (default: 135.0)")
    sun_group.add_argument("--sun-elevation", type=float, default=35.0,
                           help="Sun elevation angle in degrees (default: 35.0)")
    sun_group.add_argument("--sun-intensity", type=float, default=1.0,
                           help="Sun light intensity (default: 1.0)")
    
    args = parser.parse_args()
    
    binary = find_viewer_binary()
    dem_path = args.dem.resolve()
    
    if not dem_path.exists():
        print(f"Error: DEM file not found: {dem_path}")
        return 1
    
    # Start viewer with IPC
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
    
    # Set initial camera and terrain params
    # z_scale controls height exaggeration: world_y = h_norm * terrain_width * z_scale * 0.1
    # z_scale=1.0 gives 10% height-to-width ratio - good balance of relief
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 30.0, "theta": 45.0, "radius": 3800.0, "fov": 30.0,
        "zscale": 0.1,
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
        resp = send_ipc(sock, pbr_cmd)
        if not resp.get("ok"):
            print(f"Warning: PBR config failed: {resp.get('error')}")
        else:
            print(f"PBR mode enabled: shadows={args.shadows}, exposure={args.exposure}")
    
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
    print("\n" + "=" * 60)
    print("INTERACTIVE TERRAIN VIEWER")
    print("=" * 60)
    print("Window controls:")
    print("  Mouse drag     - Orbit camera")
    print("  Scroll wheel   - Zoom in/out")
    print("  W/S or ↑/↓     - Tilt camera up/down")
    print("  A/D or ←/→     - Rotate camera left/right")
    print("  Q/E            - Zoom out/in")
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
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
