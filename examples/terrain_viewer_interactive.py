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
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

# Import shared utilities from forge3d package
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from forge3d.viewer_ipc import find_viewer_binary, send_ipc
from forge3d.interactive import run_interactive_loop


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
    
    # Option B: Vector Overlay Geometry options
    vo_group = parser.add_argument_group("Vector Overlay (Option B)", "GPU-rendered vector overlays draped on terrain")
    vo_group.add_argument("--vector-overlay", type=str, default=None, metavar="PATH",
                          help="Path to vector file (.gpkg, .geojson, .shp) for overlay")
    vo_group.add_argument("--vo-drape", action="store_true", default=True,
                          help="Drape vector geometry onto terrain surface (default: on)")
    vo_group.add_argument("--no-vo-drape", action="store_false", dest="vo_drape",
                          help="Don't drape - use original Z coordinates")
    vo_group.add_argument("--vo-drape-offset", type=float, default=2.0,
                          help="Height above terrain when draped (default: 2.0)")
    vo_group.add_argument("--vo-line-width", type=float, default=3.0,
                          help="Line width in pixels (default: 3.0)")
    vo_group.add_argument("--vo-point-size", type=float, default=5.0,
                          help="Point size in pixels (default: 5.0)")
    vo_group.add_argument("--vo-depth-bias", type=float, default=0.1,
                          help="Depth bias for z-fighting prevention (default: 0.1)")
    vo_group.add_argument("--vo-opacity", type=float, default=1.0,
                          help="Overlay opacity [0.0-1.0] (default: 1.0)")
    vo_group.add_argument("--vo-color", type=float, nargs=4, default=[0.9, 0.3, 0.1, 1.0],
                          metavar=("R", "G", "B", "A"), help="Line/point color RGBA (default: red-orange)")
    vo_group.add_argument("--vo-enabled", action="store_true", default=True,
                          help="Enable vector overlay rendering (default: on)")
    vo_group.add_argument("--no-vo-enabled", action="store_false", dest="vo_enabled",
                          help="Disable vector overlay rendering")
    
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
        
        # M3: Depth of Field settings
        if args.dof:
            pbr_cmd["dof"] = {
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
            pbr_cmd["motion_blur"] = {
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
            pbr_cmd["lens_effects"] = {
                "enabled": True,
                "distortion": args.lens_distortion,
                "chromatic_aberration": args.lens_ca,
                "vignette_strength": args.lens_vignette,
                "vignette_radius": args.lens_vignette_radius,
                "vignette_softness": args.lens_vignette_softness,
            }
        
        # M5: Denoise settings
        if args.denoise:
            pbr_cmd["denoise"] = {
                "enabled": True,
                "method": args.denoise_method,
                "iterations": args.denoise_iterations,
                "sigma_color": args.denoise_sigma_color,
            }
        
        # M6: Volumetrics settings
        if args.volumetrics:
            pbr_cmd["volumetrics"] = {
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
            pbr_cmd["sky"] = {
                "enabled": True,
                "turbidity": args.sky_turbidity,
                "ground_albedo": args.sky_ground_albedo,
                "sun_intensity": args.sky_sun_intensity,
                "aerial_perspective": args.sky_aerial,
                "sky_exposure": args.sky_exposure,
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
    
    # Interactive mode - use shared interactive loop
    print("\nWindow controls:")
    print("  Mouse drag     - Orbit camera")
    print("  Scroll wheel   - Zoom in/out")
    print("  W/S or ↑/↓     - Tilt camera up/down")
    print("  A/D or ←/→     - Rotate camera left/right")
    print("  Q/E            - Zoom out/in")
    
    run_interactive_loop(sock, process, title="INTERACTIVE TERRAIN VIEWER")
    
    sock.close()
    process.terminate()
    process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
