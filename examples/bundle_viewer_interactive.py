#!/usr/bin/env python3
"""Interactive scene bundle viewer - load, edit, save .forge3d bundles.

This script demonstrates the Scene Bundle (.forge3d) feature in the interactive
viewer. You can load an existing bundle, adjust camera and rendering parameters,
add camera bookmarks, and save the scene to a new bundle.

**User Journeys:**

1. **Interactive Session:** Load a bundle, orbit the camera, adjust settings
   via terminal commands, then call `save_bundle` to save changes.
2. **One-shot Snapshot:** Load a bundle and take a snapshot immediately
   with `--snapshot out.png`.

Usage:
    # Load an existing bundle and view interactively
    python examples/bundle_viewer_interactive.py --load-bundle my_scene.forge3d

    # Load bundle and take snapshot immediately
    python examples/bundle_viewer_interactive.py --load-bundle my_scene.forge3d \\
        --snapshot output.png

    # Load bundle with camera override
    python examples/bundle_viewer_interactive.py --load-bundle my_scene.forge3d \\
        --cam-phi 45 --cam-theta 30

    # Create new bundle from DEM interactively
    python examples/bundle_viewer_interactive.py --dem terrain.tif \\
        --save-bundle new_scene.forge3d

Interactive Commands:
    save_bundle path/to/scene.forge3d    Save current scene to bundle
    load_bundle path/to/scene.forge3d    Load scene from bundle
    bookmark name                        Save current camera as bookmark
    bookmarks                            List camera bookmarks
    goto bookmark_name                   Jump to camera bookmark
    camera phi=45 theta=30 radius=2000   Set camera position
    sun azimuth=135 elevation=45         Set sun direction
    snap output.png                      Take screenshot
    quit                                 Exit viewer

See docs/api/bundle.md for Scene Bundle documentation.
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from forge3d.viewer_ipc import find_viewer_binary, send_ipc
from forge3d.bundle import (
    save_bundle,
    load_bundle,
    is_bundle,
    BundleManifest,
    CameraBookmark,
    BUNDLE_VERSION,
)


def _collect_scene_state(args, camera_state: dict, bookmarks: list[CameraBookmark]) -> dict:
    """Collect current scene state for saving to bundle."""
    preset = {
        "z_scale": getattr(args, "z_scale", 2.0),
        "exposure": getattr(args, "exposure", 1.0),
        "ibl_intensity": getattr(args, "ibl_intensity", 1.0),
        "colormap": getattr(args, "colormap", "terrain"),
        "colormap_strength": getattr(args, "colormap_strength", 0.5),
        "normal_strength": getattr(args, "normal_strength", 1.0),
        "sun_azimuth": getattr(args, "sun_azimuth", 135.0),
        "sun_elevation": getattr(args, "sun_elevation", 35.0),
    }
    # Add camera state
    if camera_state:
        preset.update({
            "cam_phi": camera_state.get("phi"),
            "cam_theta": camera_state.get("theta"),
            "cam_radius": camera_state.get("radius"),
            "cam_fov": camera_state.get("fov"),
        })
    # Remove None values
    return {k: v for k, v in preset.items() if v is not None}


def _apply_bundle_to_args(bundle, args):
    """Apply loaded bundle settings to args, respecting CLI overrides."""
    if bundle.preset:
        for key, val in bundle.preset.items():
            arg_name = key.replace("-", "_")
            if hasattr(args, arg_name) and getattr(args, arg_name) is None:
                setattr(args, arg_name, val)
    if bundle.manifest.terrain:
        if bundle.manifest.terrain.colormap and args.colormap is None:
            args.colormap = bundle.manifest.terrain.colormap


def _handle_bundle_command(cmd: str, args, sock, camera_state: dict, bookmarks: list) -> str | None:
    """Handle bundle-related interactive commands. Returns response message or None."""
    parts = cmd.strip().split(maxsplit=1)
    if not parts:
        return None
    
    verb = parts[0].lower()
    
    if verb == "save_bundle":
        if len(parts) < 2:
            return "Usage: save_bundle <path>"
        bundle_path = Path(parts[1])
        try:
            preset = _collect_scene_state(args, camera_state, bookmarks)
            save_bundle(
                bundle_path,
                name=bundle_path.stem,
                dem_path=args.dem if hasattr(args, "dem") and args.dem else None,
                colormap_name=getattr(args, "colormap", None),
                preset=preset,
                camera_bookmarks=bookmarks if bookmarks else None,
                hdr_path=args.hdr if hasattr(args, "hdr") and args.hdr else None,
            )
            return f"Saved bundle: {bundle_path.with_suffix('.forge3d')}"
        except Exception as e:
            return f"Error saving bundle: {e}"
    
    elif verb == "load_bundle":
        if len(parts) < 2:
            return "Usage: load_bundle <path>"
        bundle_path = Path(parts[1])
        if not is_bundle(bundle_path):
            return f"Not a valid bundle: {bundle_path}"
        try:
            bundle = load_bundle(bundle_path)
            _apply_bundle_to_args(bundle, args)
            # Update DEM if bundle has one
            if bundle.dem_path:
                args.dem = bundle.dem_path
                resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(bundle.dem_path)})
                if not resp.get("ok"):
                    return f"Loaded bundle but terrain failed: {resp.get('error')}"
            # Load bookmarks
            bookmarks.clear()
            bookmarks.extend(bundle.manifest.camera_bookmarks)
            return f"Loaded bundle: {bundle_path} ({len(bookmarks)} bookmarks)"
        except Exception as e:
            return f"Error loading bundle: {e}"
    
    elif verb == "bookmark":
        if len(parts) < 2:
            return "Usage: bookmark <name>"
        name = parts[1]
        if not camera_state:
            return "No camera state available"
        bm = CameraBookmark(
            name=name,
            eye=(camera_state.get("eye_x", 0), camera_state.get("eye_y", 0), camera_state.get("eye_z", 0)),
            target=(camera_state.get("target_x", 0), camera_state.get("target_y", 0), camera_state.get("target_z", 0)),
            fov_deg=camera_state.get("fov", 45.0),
        )
        bookmarks.append(bm)
        return f"Saved bookmark '{name}' ({len(bookmarks)} total)"
    
    elif verb == "bookmarks":
        if not bookmarks:
            return "No bookmarks saved"
        lines = ["Camera bookmarks:"]
        for i, bm in enumerate(bookmarks):
            lines.append(f"  {i+1}. {bm.name} (fov={bm.fov_deg:.1f}°)")
        return "\n".join(lines)
    
    elif verb == "goto":
        if len(parts) < 2:
            return "Usage: goto <bookmark_name>"
        name = parts[1]
        for bm in bookmarks:
            if bm.name == name:
                # Send camera command to viewer
                resp = send_ipc(sock, {
                    "cmd": "set_camera",
                    "eye": list(bm.eye) if bm.eye else None,
                    "target": list(bm.target) if bm.target else None,
                    "fov": bm.fov_deg,
                })
                if resp.get("ok"):
                    return f"Jumped to bookmark '{name}'"
                return f"Failed to set camera: {resp.get('error')}"
        return f"Bookmark not found: {name}"
    
    return None


def run_bundle_interactive_loop(sock, process, args, title="BUNDLE VIEWER"):
    """Run interactive command loop with bundle support."""
    camera_state = {}
    bookmarks = list(args.initial_bookmarks) if hasattr(args, "initial_bookmarks") else []
    
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print("\nBundle commands:")
    print("  save_bundle <path>   - Save scene to .forge3d bundle")
    print("  load_bundle <path>   - Load scene from .forge3d bundle")
    print("  bookmark <name>      - Save current camera as bookmark")
    print("  bookmarks            - List all camera bookmarks")
    print("  goto <name>          - Jump to camera bookmark")
    print("\nOther commands:")
    print("  camera phi=N theta=N radius=N - Set camera")
    print("  sun azimuth=N elevation=N     - Set sun")
    print("  snap <path>                   - Take screenshot")
    print("  quit                          - Exit viewer")
    print()
    
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
            
            # Handle bundle commands
            result = _handle_bundle_command(cmd, args, sock, camera_state, bookmarks)
            if result:
                print(result)
                continue
            
            # Handle standard commands
            if cmd.lower() in ("quit", "exit", "q"):
                break
            
            if cmd.lower().startswith("snap"):
                parts = cmd.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else "snapshot.png"
                resp = send_ipc(sock, {"cmd": "snapshot", "path": str(Path(path).resolve())})
                if resp.get("ok"):
                    print(f"Saved: {path}")
                else:
                    print(f"Snapshot failed: {resp.get('error')}")
                continue
            
            if cmd.lower().startswith("camera"):
                # Parse camera phi=N theta=N radius=N
                params = {}
                for match in re.finditer(r"(\w+)=([0-9.]+)", cmd):
                    params[match.group(1)] = float(match.group(2))
                if params:
                    camera_state.update(params)
                    resp = send_ipc(sock, {"cmd": "set_terrain", **params})
                    if resp.get("ok"):
                        print(f"Camera updated: {params}")
                    else:
                        print(f"Camera update failed: {resp.get('error')}")
                continue
            
            if cmd.lower().startswith("sun"):
                params = {}
                for match in re.finditer(r"(\w+)=([0-9.]+)", cmd):
                    key = match.group(1)
                    val = float(match.group(2))
                    if key == "azimuth":
                        params["sun_azimuth"] = val
                    elif key == "elevation":
                        params["sun_elevation"] = val
                if params:
                    resp = send_ipc(sock, {"cmd": "set_terrain", **params})
                    if resp.get("ok"):
                        print(f"Sun updated: {params}")
                    else:
                        print(f"Sun update failed: {resp.get('error')}")
                continue
            
            print(f"Unknown command: {cmd}")
            
    except KeyboardInterrupt:
        print("\nInterrupted")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive scene bundle viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Bundle options
    bundle_group = parser.add_argument_group("Bundle Options", "Scene bundle save/load")
    bundle_group.add_argument("--load-bundle", type=Path,
                              help="Load scene from .forge3d bundle")
    bundle_group.add_argument("--save-bundle", type=Path,
                              help="Save scene to .forge3d bundle on exit")
    
    # Standard options
    parser.add_argument("--dem", type=Path,
                        help="Path to GeoTIFF DEM file (optional if loading bundle)")
    parser.add_argument("--hdr", type=Path,
                        help="HDR environment map for IBL lighting")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--snapshot", type=Path,
                        help="Take snapshot at this path and exit")
    
    # Camera options (can override bundle)
    cam_group = parser.add_argument_group("Camera", "Camera position (overrides bundle)")
    cam_group.add_argument("--cam-phi", type=float, default=None,
                           help="Camera azimuth angle in degrees")
    cam_group.add_argument("--cam-theta", type=float, default=None,
                           help="Camera polar angle in degrees")
    cam_group.add_argument("--cam-radius", type=float, default=None,
                           help="Camera distance from target")
    cam_group.add_argument("--cam-fov", type=float, default=None,
                           help="Camera field of view in degrees")
    
    # Rendering options
    render_group = parser.add_argument_group("Rendering", "Render settings (overrides bundle)")
    render_group.add_argument("--exposure", type=float, default=None,
                              help="Exposure multiplier")
    render_group.add_argument("--z-scale", type=float, default=None,
                              help="Vertical exaggeration")
    render_group.add_argument("--colormap", type=str, default=None,
                              help="Colormap name")
    render_group.add_argument("--colormap-strength", type=float, default=None,
                              help="Colormap blend strength")
    render_group.add_argument("--normal-strength", type=float, default=None,
                              help="Normal map strength")
    render_group.add_argument("--ibl-intensity", type=float, default=None,
                              help="IBL intensity")
    
    # Sun options
    sun_group = parser.add_argument_group("Sun", "Sun lighting (overrides bundle)")
    sun_group.add_argument("--sun-azimuth", type=float, default=None,
                           help="Sun azimuth in degrees")
    sun_group.add_argument("--sun-elevation", type=float, default=None,
                           help="Sun elevation in degrees")
    
    args = parser.parse_args()
    
    # Load bundle if specified
    initial_bookmarks = []
    if args.load_bundle:
        bundle_path = Path(args.load_bundle)
        if not is_bundle(bundle_path):
            print(f"Error: Not a valid bundle: {bundle_path}")
            return 1
        print(f"Loading bundle: {bundle_path}")
        bundle = load_bundle(bundle_path)
        
        # Apply bundle settings (CLI args override)
        if bundle.dem_path and args.dem is None:
            args.dem = bundle.dem_path
        if bundle.hdr_path and args.hdr is None:
            args.hdr = bundle.hdr_path
        
        _apply_bundle_to_args(bundle, args)
        initial_bookmarks = list(bundle.manifest.camera_bookmarks)
        print(f"  Loaded {len(initial_bookmarks)} camera bookmarks")
    
    # Require DEM
    if args.dem is None:
        print("Error: --dem required (or load a bundle with terrain)")
        return 1
    
    dem_path = args.dem.resolve()
    if not dem_path.exists():
        print(f"Error: DEM not found: {dem_path}")
        return 1
    
    # Apply defaults for unset values
    args.cam_phi = args.cam_phi if args.cam_phi is not None else 30.0
    args.cam_theta = args.cam_theta if args.cam_theta is not None else 45.0
    args.cam_radius = args.cam_radius if args.cam_radius is not None else 3800.0
    args.cam_fov = args.cam_fov if args.cam_fov is not None else 30.0
    args.exposure = args.exposure if args.exposure is not None else 1.0
    args.z_scale = args.z_scale if args.z_scale is not None else 2.0
    args.sun_azimuth = args.sun_azimuth if args.sun_azimuth is not None else 135.0
    args.sun_elevation = args.sun_elevation if args.sun_elevation is not None else 35.0
    args.initial_bookmarks = initial_bookmarks
    
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
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": args.cam_phi,
        "theta": args.cam_theta,
        "radius": args.cam_radius,
        "fov": args.cam_fov,
        "zscale": args.z_scale * 0.05,
        "sun_azimuth": args.sun_azimuth,
        "sun_elevation": args.sun_elevation,
    })
    
    # Apply render overrides via PBR config
    pbr_cmd = {
        "cmd": "set_terrain_pbr",
        "enabled": True,
        "exposure": args.exposure,
    }
    if args.ibl_intensity is not None:
        pbr_cmd["ibl_intensity"] = args.ibl_intensity
    if args.normal_strength is not None:
        pbr_cmd["normal_strength"] = args.normal_strength
    if args.hdr:
        pbr_cmd["hdr_path"] = str(args.hdr.resolve())
    
    resp = send_ipc(sock, pbr_cmd)
    if resp.get("ok"):
        overrides = [f"exposure={args.exposure}"]
        if args.ibl_intensity is not None:
            overrides.append(f"ibl_intensity={args.ibl_intensity}")
        if args.normal_strength is not None:
            overrides.append(f"normal_strength={args.normal_strength}")
        if args.hdr:
            overrides.append(f"hdr={args.hdr.name}")
        print(f"PBR enabled: {', '.join(overrides)}")
    else:
        print(f"Warning: PBR config failed: {resp.get('error')}")

    # Snapshot mode
    if args.snapshot:
        time.sleep(0.5)
        resp = send_ipc(sock, {
            "cmd": "snapshot",
            "path": str(args.snapshot.resolve()),
            "width": args.width,
            "height": args.height,
        })
        
        # Save bundle if requested
        if args.save_bundle:
            camera_state = {
                "phi": args.cam_phi,
                "theta": args.cam_theta,
                "radius": args.cam_radius,
                "fov": args.cam_fov,
            }
            preset = _collect_scene_state(args, camera_state, initial_bookmarks)
            save_bundle(
                args.save_bundle,
                name=args.save_bundle.stem,
                dem_path=dem_path,
                preset=preset,
                camera_bookmarks=initial_bookmarks if initial_bookmarks else None,
                hdr_path=args.hdr.resolve() if args.hdr else None,
            )
            print(f"Saved bundle: {args.save_bundle.with_suffix('.forge3d')}")
        
        send_ipc(sock, {"cmd": "close"})
        sock.close()
        process.wait()
        
        if args.snapshot.exists():
            print(f"Saved: {args.snapshot}")
            return 0
        return 1
    
    # Interactive mode
    run_bundle_interactive_loop(sock, process, args, title="SCENE BUNDLE VIEWER")
    
    # Save bundle on exit if requested
    if args.save_bundle:
        camera_state = {
            "phi": args.cam_phi,
            "theta": args.cam_theta,
            "radius": args.cam_radius,
            "fov": args.cam_fov,
        }
        preset = _collect_scene_state(args, camera_state, initial_bookmarks)
        save_bundle(
            args.save_bundle,
            name=args.save_bundle.stem,
            dem_path=dem_path,
            preset=preset,
            camera_bookmarks=args.initial_bookmarks if args.initial_bookmarks else None,
            hdr_path=args.hdr.resolve() if args.hdr else None,
        )
        print(f"Saved bundle: {args.save_bundle.with_suffix('.forge3d')}")
    
    sock.close()
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
