#!/usr/bin/env python3
"""Interactive terrain viewer - load DEM, orbit camera, take snapshots.

This script launches the interactive viewer and loads a terrain DEM for
real-time viewing. You can orbit the camera around the terrain, adjust
lighting, and take snapshots when you're happy with the view.

**Rendering Modes:**

- **Legacy mode** (default): Simple Lambertian shading with height-based colors
- **PBR mode** (`--pbr`): Enhanced Blinn-Phong lighting, ACES tonemapping,
  height+slope materials, configurable exposure

**Terrain Effects (PBR mode):**

- **Heightfield AO** (`--height-ao`): Ambient occlusion from terrain geometry
- **Sun Visibility** (`--sun-vis`): Terrain self-shadowing along sun direction

Usage:
    # Basic usage - load terrain with legacy rendering
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif
    
    # PBR mode - enhanced lighting and materials
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr
    
    # PBR with custom exposure (brighter)
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr --exposure 1.5
    
    # PBR with heightfield AO (darkens valleys/crevices)
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr --height-ao --height-ao-strength 1.0
    
    # PBR with sun visibility (terrain self-shadowing)
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr --sun-vis --sun-vis-mode soft --sun-elevation 15
    
    # PBR with all terrain effects
    python examples/terrain_viewer_interactive.py --dem assets/Gore_Range_Albers_1m.tif \\
        --pbr --height-ao --sun-vis --sun-vis-mode hard --exposure 1.2
    
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
    parser.add_argument("--preset", choices=["default", "alpine", "cinematic", "high_quality"],
                        help="Rendering preset (overrides individual settings)")
    
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
    
    # Heightfield Ray AO options
    ao_group = parser.add_argument_group("Heightfield AO", "Terrain ambient occlusion from heightfield ray-tracing")
    ao_group.add_argument("--height-ao", action="store_true",
                          help="Enable heightfield ray-traced ambient occlusion")
    ao_group.add_argument("--height-ao-directions", type=int, default=6,
                          help="Number of ray directions around horizon [4-16] (default: 6)")
    ao_group.add_argument("--height-ao-steps", type=int, default=16,
                          help="Ray march steps per direction [8-64] (default: 16)")
    ao_group.add_argument("--height-ao-distance", type=float, default=200.0,
                          help="Maximum ray distance in world units (default: 200.0)")
    ao_group.add_argument("--height-ao-strength", type=float, default=1.0,
                          help="AO darkening intensity [0.0-2.0] (default: 1.0)")
    ao_group.add_argument("--height-ao-resolution", type=float, default=0.5,
                          help="AO texture resolution scale [0.1-1.0] (default: 0.5)")
    
    # Sun Visibility options
    sv_group = parser.add_argument_group("Sun Visibility", "Terrain self-shadowing from heightfield ray-tracing")
    sv_group.add_argument("--sun-vis", action="store_true",
                          help="Enable heightfield ray-traced sun visibility")
    sv_group.add_argument("--sun-vis-mode", choices=["hard", "soft"], default="soft",
                          help="Shadow mode: hard (binary) or soft (jittered) (default: soft)")
    sv_group.add_argument("--sun-vis-samples", type=int, default=4,
                          help="Number of jittered samples for soft shadows [1-16] (default: 4)")
    sv_group.add_argument("--sun-vis-steps", type=int, default=24,
                          help="Ray march steps toward sun [8-64] (default: 24)")
    sv_group.add_argument("--sun-vis-distance", type=float, default=400.0,
                          help="Maximum ray distance in world units (default: 400.0)")
    sv_group.add_argument("--sun-vis-softness", type=float, default=1.0,
                          help="Soft shadow penumbra size [0.0-4.0] (default: 1.0)")
    sv_group.add_argument("--sun-vis-bias", type=float, default=0.01,
                          help="Self-shadowing bias to reduce artifacts (default: 0.01)")
    sv_group.add_argument("--sun-vis-resolution", type=float, default=0.5,
                          help="Visibility texture resolution scale [0.1-1.0] (default: 0.5)")
    
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
    
    # M5: Vector Overlay options
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
        
        # Heightfield Ray AO settings
        if args.height_ao:
            pbr_cmd["height_ao"] = {
                "enabled": True,
                "directions": args.height_ao_directions,
                "steps": args.height_ao_steps,
                "max_distance": args.height_ao_distance,
                "strength": args.height_ao_strength,
                "resolution_scale": args.height_ao_resolution,
            }
        
        # Sun Visibility settings
        if args.sun_vis:
            pbr_cmd["sun_visibility"] = {
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
            pbr_cmd["materials"] = {
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
            pbr_cmd["vector_overlay"] = {
                "depth_test": args.overlay_depth,
                "depth_bias": args.overlay_depth_bias,
                "halo_enabled": args.overlay_halo,
                "halo_width": args.overlay_halo_width,
                "halo_color": halo_color,
            }
        
        # M6: Tonemap settings
        pbr_cmd["tonemap"] = {
            "operator": args.tonemap,
            "white_point": args.tonemap_white_point,
            "white_balance_enabled": args.white_balance,
            "temperature": args.temperature,
            "tint": args.tint,
        }
        
        resp = send_ipc(sock, pbr_cmd)
        if not resp.get("ok"):
            print(f"Warning: PBR config failed: {resp.get('error')}")
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
            print(f"PBR mode enabled: {', '.join(features)}")
    
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
