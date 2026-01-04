"""Interactive command loop utilities for terrain viewer examples.

This module provides reusable components for building interactive
terminal-based viewer control interfaces.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
    print("Other commands:")
    print("  params         - Show current parameters")
    print("  snap <path> [<width>x<height>]  - Take snapshot")
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
) -> None:
    """Run interactive command loop for terrain viewer.
    
    Handles standard commands: quit, set, params, snap, pbr, cam, sun
    
    Args:
        sock: Connected socket
        process: Viewer subprocess (for poll check)
        title: Title for help display
        extra_commands: Dict of command_name -> handler(sock, args) for custom commands
        post_snapshot_callback: Optional callback after successful snapshot
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
                print("Unknown command. Type 'set', 'params', 'snap', 'pbr', or 'quit'")
    
    except KeyboardInterrupt:
        pass
