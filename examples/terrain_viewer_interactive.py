#!/usr/bin/env python3
"""Interactive terrain viewer - load DEM, orbit camera, take snapshots.

This script launches the interactive viewer and loads a terrain DEM for
real-time viewing. You can orbit the camera around the terrain, adjust
lighting, and take snapshots when you're happy with the view.

Usage:
    # Basic usage - load Mt. Rainier DEM
    python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif
    
    # Take automatic snapshot and exit
    python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif \\
        --snapshot output.png
    
    # Custom window size
    python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif \\
        --width 1920 --height 1080
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
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")
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
    
    # Set initial camera and terrain params (zoomed out with low z-scale)
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 30.0, "theta": 135.0, "radius": 3800.0, "fov": 45.0,
        "zscale": 0.1
    })
    
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
    print("  snap <path>    - Take snapshot")
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
                    path = str(Path(cmd_args.strip()).resolve())
                    resp = send_ipc(sock, {"cmd": "snapshot", "path": path})
                    if resp.get("ok"):
                        print(f"Saved: {path}")
                else:
                    print("Usage: snap <path>")
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
            else:
                print("Unknown command. Type 'set', 'params', 'snap', or 'quit'")
    except KeyboardInterrupt:
        pass
    
    sock.close()
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
