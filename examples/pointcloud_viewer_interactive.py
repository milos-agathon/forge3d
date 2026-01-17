#!/usr/bin/env python3
"""Interactive point cloud viewer - load LAZ/LAS, orbit camera, take snapshots.

This script launches the interactive viewer and loads a point cloud for
real-time viewing. You can orbit the camera around the point cloud, adjust
rendering parameters, and take snapshots.

Usage:
    # Basic usage - load point cloud
    python examples/pointcloud_viewer_interactive.py --input assets/lidar/MtStHelens.laz
    
    # Custom point size and max points
    python examples/pointcloud_viewer_interactive.py --input assets/lidar/MtStHelens.laz \\
        --point-size 3.0 --max-points 1000000
    
    # Different color modes
    python examples/pointcloud_viewer_interactive.py --input assets/lidar/MtStHelens.laz \\
        --color-mode rgb
    
    # Take automatic snapshot and exit
    python examples/pointcloud_viewer_interactive.py --input assets/lidar/MtStHelens.laz \\
        --snapshot output.png
    
    # Custom window size
    python examples/pointcloud_viewer_interactive.py --input assets/lidar/MtStHelens.laz \\
        --width 1920 --height 1080

Interactive Commands:
    set phi=45 theta=30 radius=2.0     Set camera position
    set point_size=3.0                 Adjust point size
    set color_mode=elevation           Change color mode (elevation/rgb/intensity)
    snap output.png [1920x1080]        Take screenshot
    params                             Show current parameters
    quit                               Exit viewer

Window Controls:
    Mouse drag     - Orbit camera
    Scroll wheel   - Zoom in/out
    W/S or ↑/↓     - Tilt camera up/down
    A/D or ←/→     - Rotate camera left/right
    Q/E            - Zoom out/in
    P              - Take snapshot (saves to pointcloud_snapshot_<timestamp>.png)
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Import shared utilities from forge3d package
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from forge3d.viewer_ipc import find_viewer_binary, send_ipc


# Point cloud specific key mapping
POINTCLOUD_KEY_MAP = {
    "phi": "phi",
    "theta": "theta",
    "radius": "radius",
    "point_size": "point_size",
    "color_mode": "color_mode",
    "visible": "visible",
}


def parse_set_command(args: str) -> Dict[str, Any]:
    """Parse 'set key=value key=value ...' into IPC params."""
    params: Dict[str, Any] = {"cmd": "set_point_cloud_params"}
    
    for pair in args.split():
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.lower().strip()
        val = val.strip()
        
        ipc_key = POINTCLOUD_KEY_MAP.get(key, key)
        
        # Parse value
        if val.lower() in ("true", "on", "yes"):
            params[ipc_key] = True
        elif val.lower() in ("false", "off", "no"):
            params[ipc_key] = False
        else:
            try:
                params[ipc_key] = float(val)
            except ValueError:
                params[ipc_key] = val
    
    return params


def handle_snapshot_command(sock: socket.socket, args: str) -> bool:
    """Handle snapshot command: snap <path> [<width>x<height>]"""
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
    
    resp = send_ipc(sock, snap_cmd)
    if resp.get("ok"):
        print(f"Saved: {path}")
    else:
        print(f"Snapshot failed: {resp.get('error')}")
    
    return True


def print_help(title: str = "POINT CLOUD VIEWER") -> None:
    """Print help text for interactive mode."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print()
    print("Terminal commands:")
    print("  set phi=45 theta=30 radius=2.0    - Set camera orbit position")
    print("  set point_size=3.0                - Adjust point size")
    print("  set color_mode=elevation          - Color mode (elevation/rgb/intensity)")
    print("  set visible=true                  - Toggle visibility")
    print()
    print("Snapshot commands:")
    print("  snap <path> [<width>x<height>]    - Take screenshot")
    print()
    print("Other commands:")
    print("  params         - Show current parameters")
    print("  reload         - Reload point cloud from file")
    print("  clear          - Clear point cloud")
    print("  quit           - Close viewer")
    print("=" * 60 + "\n")


def run_interactive_loop(
    sock: socket.socket,
    process,
    input_path: Path,
    title: str = "POINT CLOUD VIEWER",
) -> None:
    """Run interactive command loop for point cloud viewer."""
    print_help(title)
    
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
                resp = send_ipc(sock, {"cmd": "get_point_cloud_params"})
                if resp.get("ok"):
                    for k, v in resp.items():
                        if k != "ok":
                            print(f"  {k}: {v}")
                else:
                    print(f"Error: {resp.get('error')}")
            
            elif name == "snap":
                handle_snapshot_command(sock, cmd_args)
            
            elif name == "reload":
                resp = send_ipc(sock, {
                    "cmd": "load_point_cloud",
                    "path": str(input_path),
                })
                if resp.get("ok"):
                    print("Point cloud reloaded")
                else:
                    print(f"Error: {resp.get('error')}")
            
            elif name == "clear":
                resp = send_ipc(sock, {"cmd": "clear_point_cloud"})
                if resp.get("ok"):
                    print("Point cloud cleared")
                else:
                    print(f"Error: {resp.get('error')}")
            
            elif name == "help":
                print_help(title)
            
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive point cloud viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to LAZ/LAS point cloud file")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")
    
    # Point cloud options
    pc_group = parser.add_argument_group("Point Cloud", "Point cloud rendering options")
    pc_group.add_argument("--point-size", type=float, default=2.0,
                          help="Point size in pixels (default: 2.0)")
    pc_group.add_argument("--max-points", type=int, default=500_000,
                          help="Maximum points to load (default: 500000)")
    pc_group.add_argument("--color-mode", choices=["elevation", "rgb", "intensity"],
                          default="elevation", help="Color mode (default: elevation)")
    
    # Camera options
    cam_group = parser.add_argument_group("Camera", "Initial camera position")
    cam_group.add_argument("--phi", type=float, default=0.0,
                           help="Camera azimuth angle in radians (default: 0.0)")
    cam_group.add_argument("--theta", type=float, default=0.5,
                           help="Camera elevation angle in radians (default: 0.5)")
    cam_group.add_argument("--radius", type=float, default=1.0,
                           help="Camera distance multiplier (default: 1.0)")
    
    # Advanced options
    adv_group = parser.add_argument_group("Advanced", "Advanced options")
    adv_group.add_argument("--port", type=int, default=0,
                           help="IPC port (default: 0 = auto-assign)")
    
    args = parser.parse_args()
    
    binary = find_viewer_binary()
    input_path = args.input.resolve()
    
    if not input_path.exists():
        print(f"Error: Point cloud file not found: {input_path}")
        return 1
    
    # Start viewer with IPC
    cmd = [binary, "--ipc-port", str(args.port), "--size", f"{args.width}x{args.height}"]
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
    
    # Connect and load point cloud
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    
    print(f"Loading point cloud: {input_path}")
    resp = send_ipc(sock, {
        "cmd": "load_point_cloud",
        "path": str(input_path),
        "point_size": args.point_size,
        "max_points": args.max_points,
        "color_mode": args.color_mode,
    })
    if not resp.get("ok"):
        print(f"Failed to load point cloud: {resp.get('error')}")
        sock.close()
        process.terminate()
        return 1
    
    print(f"Loaded point cloud: {resp.get('point_count', 'unknown')} points")
    
    # Set initial camera if specified
    if args.phi != 0.0 or args.theta != 0.5 or args.radius != 1.0:
        send_ipc(sock, {
            "cmd": "set_point_cloud_params",
            "phi": args.phi,
            "theta": args.theta,
            "radius": args.radius,
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
    print("\nWindow controls:")
    print("  Mouse drag     - Orbit camera")
    print("  Scroll wheel   - Zoom in/out")
    print("  W/S or ↑/↓     - Tilt camera up/down")
    print("  A/D or ←/→     - Rotate camera left/right")
    print("  Q/E            - Zoom out/in")
    
    run_interactive_loop(sock, process, input_path, title="INTERACTIVE POINT CLOUD VIEWER")
    
    sock.close()
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
