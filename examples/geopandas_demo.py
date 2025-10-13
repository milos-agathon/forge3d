# examples/geopandas_demo.py
# Updated: Load a GeoTIFF DEM (Gore_Range_Albers_1m.tif) and render with a custom palette.
# - Reads elevation from assets/Gore_Range_Albers_1m.tif
# - Uses an interpolated 128-color palette built from provided hex stops
# - Saves an RGBA preview PNG or opens interactive 3D viewer
#
# CAMERA CONTROL & SNAPSHOT FEATURES:
# 1. Single-terminal interactive mode (recommended):
#    python examples/geopandas_demo.py --viewer-3d --water
#    
#    In the same terminal, use the built-in prompt to control the viewer:
#      forge3d> help
#      forge3d> camera distance=2000 theta=45 phi=30
#      forge3d> snapshot my_view.png 1920 1080
#      forge3d> get_camera
#
#    You can still use the Python API from a second terminal if preferred:
#      python
#      >>> import forge3d as f3d
#      >>> f3d.set_camera(distance=2000, theta=45, phi=30)
#      >>> f3d.snapshot('my_view.png', width=1920, height=1080)
#      >>> print(f3d.get_camera())
#
# 2. Automatic snapshot demo (single terminal - captures 10 views):
#    python examples/geopandas_demo.py --viewer-3d --water --demo-snapshots
#
# Available Python API:
#  - f3d.set_camera(distance=None, theta=None, phi=None): Control camera position
#  - f3d.snapshot(path, width=None, height=None): Save PNG at custom resolution
#  - f3d.get_camera(): Get current camera state for raytracing

from __future__ import annotations

import argparse
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

 

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover
    raise ImportError("forge3d Python API is required. Ensure package is installed or built.") from exc

# Import lean helpers from the package to avoid duplicating logic in the example
from forge3d.render import (
    load_dem as _pkg_load_dem,
    resolve_palette_argument,
    RaytraceMeshCache,
)
from forge3d.helpers.interactive_cli import (
    _command_reader,
    interactive_control_loop,
    demo_snapshots,
)


CUSTOM_HEX_COLORS: Sequence[str] = (
    "#e7d8a2", "#c5a06e", "#995f57", "#4a3c37"
)

def main() -> int:
    parser = argparse.ArgumentParser(description="M3 DEM → Terrain render demo (Gore Range 1m)")
    parser.add_argument("--src", type=Path, default=Path("assets/Gore_Range_Albers_1m.tif"), help="Input GeoTIFF DEM")
    parser.add_argument("--out", type=Path, default=Path("reports/Gore_Range_Albers_1m.png"), help="Output PNG path")
    parser.add_argument("--output-size", type=int, nargs=2, default=(800, 600), metavar=("W", "H"), help="Output size (pixels)")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # Color/palette parameters
    parser.add_argument("--colormap", type=str, default="custom", help="Colormap name or preset")
    parser.add_argument("--invert-palette", action="store_true", help="Invert palette direction")
    parser.add_argument("--palette-interpolate", action="store_true", help="Interpolate palette colors (smooth gradients)")
    parser.add_argument("--no-palette-interpolate", dest="palette_interpolate", action="store_false", help="Use discrete palette colors (no interpolation)")
    parser.set_defaults(palette_interpolate=False)
    parser.add_argument("--palette-size", type=int, default=256, help="Number of colors when interpolating palette")
    parser.add_argument("--contrast-pct", type=float, default=1.0, help="Percentile clip for normalization")
    parser.add_argument("--gamma", type=float, default=1.1, help="Gamma correction")
    parser.add_argument("--equalize", action="store_true", default=True, help="Histogram equalization")
    parser.add_argument("--no-equalize", dest="equalize", action="store_false", help="Disable histogram equalization")
    parser.add_argument("--exaggeration", type=float, default=0.0, help="Vertical exaggeration (<=0 auto)")

    # Shadow/lighting parameters
    shadow_group = parser.add_mutually_exclusive_group()
    shadow_group.add_argument("--shadows", dest="shadow_enabled", action="store_true", help="Enable shadows")
    shadow_group.add_argument("--no-shadows", dest="shadow_enabled", action="store_false", help="Disable shadows")
    parser.set_defaults(shadow_enabled=True)
    parser.add_argument("--shadow-intensity", type=float, default=1.0, help="Shadow strength [0..1]")
    parser.add_argument(
        "--lighting-type",
        type=str,
        default="lambertian",
        choices=["lambertian", "flat", "phong", "blinn-phong"],
        help="Lighting model",
    )
    parser.add_argument("--lighting-intensity", type=float, default=1.0, help="Light intensity multiplier")
    parser.add_argument("--lighting-azimuth", type=float, default=315.0, help="Light azimuth (degrees, 0=N)")
    parser.add_argument("--lighting-elevation", type=float, default=45.0, help="Light elevation (degrees)")
    
    # Camera parameters
    parser.add_argument("--camera-distance", type=float, default=1.0, help="Camera distance")
    parser.add_argument("--camera-phi", type=float, default=0.0, help="Camera azimuthal angle (degrees)")
    parser.add_argument("--camera-theta", type=float, default=90.0, help="Camera polar angle (degrees, 90=overhead)")
    
    # Water parameters
    water_group = parser.add_mutually_exclusive_group()
    water_group.add_argument("--water", dest="water_enabled", action="store_true", help="Enable water detection")
    water_group.add_argument("--no-water", dest="water_enabled", action="store_false", help="Disable water detection")
    parser.set_defaults(water_enabled=True)
    parser.add_argument("--water-level", type=float, default=None, help="Water elevation (None = use percentile)")
    parser.add_argument("--water-level-percentile", type=float, default=30.0, help="Water level percentile")
    parser.add_argument("--water-method", type=str, default="flat", help="Water detection method")
    parser.add_argument("--water-smooth", type=int, default=1, help="Water smoothing iterations")
    parser.add_argument("--water-color", type=float, nargs=3, default=None, metavar=("R", "G", "B"), help="Fixed water RGB")
    parser.add_argument("--water-shallow", type=float, nargs=3, default=None, metavar=("R", "G", "B"), help="Shallow water RGB")
    parser.add_argument("--water-deep", type=float, nargs=3, default=None, metavar=("R", "G", "B"), help="Deep water RGB")
    parser.add_argument("--water-depth-gamma", type=float, default=1.0, help="Water depth gamma")
    parser.add_argument("--water-depth-max", type=float, default=None, help="Max depth for color mapping")
    parser.add_argument("--water-keep-components", type=int, default=2, help="Keep N largest water regions")
    parser.add_argument("--water-min-area-pct", type=float, default=0.01, help="Min water area %%")
    parser.add_argument("--water-morph-iter", type=int, default=1, help="Morphology cleanup iterations")
    parser.add_argument("--water-max-slope-deg", type=float, default=6.0, help="Max slope for water")
    parser.add_argument("--water-min-depth", type=float, default=0.1, help="Min water depth")
    parser.add_argument("--water-debug", action="store_true", help="Water detection debug info")
    
    # Viewer parameters
    parser.add_argument("--viewer", action="store_true", help="Open interactive 2D image viewer instead of saving to file")
    parser.add_argument("--viewer-3d", action="store_true", help="Open interactive 3D terrain viewer with camera controls")
    parser.add_argument("--viewer-subsample", type=int, default=4, help="Subsample factor for 3D mesh (1=full, 2=half, 4=quarter)")
    parser.add_argument("--viewer-vscale", type=float, default=1.0, help="Vertical exaggeration for 3D viewer")
    parser.add_argument("--demo-snapshots", action="store_true", help="Demo: Automatically capture snapshots from multiple camera angles")
    parser.add_argument("--demo-snapshot-dir", type=Path, default=Path("snapshots"), help="Directory for demo snapshots")
    parser.add_argument(
        "--cli-mode",
        type=str,
        default="process",
        choices=["process", "thread"],
        help="Interactive prompt mode: 'process' (default, robust if viewer holds GIL) or 'thread' (shares same stdin)"
    )
    
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    verbose = not bool(args.quiet)

    # Load DEM
    try:
        hm, spacing = _pkg_load_dem(args.src)
    except Exception as exc:
        print(f"Failed to load DEM '{args.src}': {exc}")
        return 0

    # Determine shadow toggle
    shadow_enabled = bool(args.shadow_enabled)

    # Determine water toggle and prepare water parameters
    water_enabled = bool(args.water_enabled)

    # Set default water colors if enabled and not explicitly provided
    if water_enabled:
        water_color = tuple(args.water_color) if args.water_color else None
        water_shallow = tuple(args.water_shallow) if args.water_shallow else (0.4, 0.7, 0.9)  # Light blue
        water_deep = tuple(args.water_deep) if args.water_deep else (0.1, 0.3, 0.6)  # Deep blue
    else:
        water_color = None
        water_shallow = None
        water_deep = None

    # If 3D viewer is requested, prepare CLI and raytrace helper early so the
    # user sees the prompt immediately (before heavy rendering occurs).
    cli_started = False
    stop_event = threading.Event()
    control_thread: Optional[threading.Thread] = None
    # Reader/control resources (filled per --cli-mode)
    command_queue = None  # type: ignore[assignment]
    response_queue = None  # type: ignore[assignment]
    command_proc: Optional[mp.Process] = None
    command_reader_thread: Optional[threading.Thread] = None
    command_stop = None  # type: ignore[assignment]

    raytrace_helper: Optional[RaytraceMeshCache] = None
    if args.viewer_3d:
        try:
            raytrace_helper = RaytraceMeshCache(
                hm,
                spacing,
                subsample=max(1, int(args.viewer_subsample)),
                vertical_scale=float(args.viewer_vscale),
            )
        except Exception as exc:
            if verbose:
                print(f"[WARN] Raytrace mesh unavailable: {exc}")
            raytrace_helper = None

        if not args.demo_snapshots and verbose:
            # Start CLI before heavy rendering (concise notice)
            cli_mode_desc = "thread" if args.cli_mode == "thread" else "process"
            print(f"Interactive CLI ({cli_mode_desc}) started. Type 'help' for commands; 'quit' to exit.", flush=True)

            if args.cli_mode == "thread":
                import queue as threading_queue
                command_queue = threading_queue.Queue()
                response_queue = threading_queue.Queue()
                command_stop = threading.Event()

                # Reader thread to handle stdin and echo responses
                command_reader_thread = threading.Thread(
                    target=_command_reader,
                    args=(command_queue, response_queue, command_stop),
                    daemon=True,
                )
                command_reader_thread.start()

                # Control loop thread to process commands and call f3d.* APIs
                control_thread = threading.Thread(
                    target=interactive_control_loop,
                    kwargs=dict(
                        dem_path=args.src,
                        dem_data=hm,
                        spacing=spacing,
                        raytrace_helper=raytrace_helper,
                        default_size=tuple(args.output_size),
                        stop_event=stop_event,
                        command_queue=command_queue,
                        response_queue=response_queue,
                        command_stop=command_stop,
                    ),
                    daemon=True,
                )
                control_thread.start()
            else:
                # Use a dedicated process to read stdin and a thread to run the control loop
                command_queue = mp.Queue()
                response_queue = mp.Queue()
                command_stop = mp.Event()

                command_proc = mp.Process(
                    target=_command_reader,
                    args=(command_queue, response_queue, command_stop),
                    daemon=True,
                )
                command_proc.start()

                control_thread = threading.Thread(
                    target=interactive_control_loop,
                    kwargs=dict(
                        dem_path=args.src,
                        dem_data=hm,
                        spacing=spacing,
                        raytrace_helper=raytrace_helper,
                        default_size=tuple(args.output_size),
                        stop_event=stop_event,
                        command_queue=command_queue,
                        response_queue=response_queue,
                        command_stop=command_stop,
                    ),
                    daemon=True,
                )
                control_thread.start()

            cli_started = True

    # Resolve palette with interpolation settings (may be used for 2D or 3D)
    palette = resolve_palette_argument(
        args.colormap,
        interpolate=args.palette_interpolate,
        size=args.palette_size,
        base_colors=CUSTOM_HEX_COLORS,
    )

    # Call render_raster with all parameters
    rgba = f3d.render_raster(
        hm,
        size=tuple(args.output_size),
        spacing=spacing,
        renderer="hillshade",
        palette=palette,
        invert_palette=args.invert_palette,
        contrast_pct=args.contrast_pct,
        gamma=args.gamma,
        equalize=args.equalize,
        exaggeration=args.exaggeration,
        shadow_enabled=shadow_enabled,
        shadow_intensity=args.shadow_intensity,
        lighting_type=args.lighting_type,
        lighting_intensity=args.lighting_intensity,
        lighting_azimuth=args.lighting_azimuth,
        lighting_elevation=args.lighting_elevation,
        camera_distance=args.camera_distance,
        camera_phi=args.camera_phi,
        camera_theta=args.camera_theta,
        water_level=args.water_level if water_enabled else None,
        water_level_percentile=args.water_level_percentile if water_enabled else None,
        water_method=args.water_method if water_enabled else None,
        water_smooth=args.water_smooth if water_enabled else 0,
        water_color=water_color if water_enabled else None,
        water_shallow=water_shallow if water_enabled else None,
        water_deep=water_deep if water_enabled else None,
        water_depth_gamma=args.water_depth_gamma if water_enabled else 1.0,
        water_depth_max=args.water_depth_max if water_enabled else None,
        water_keep_components=args.water_keep_components if water_enabled else 0,
        water_min_area_pct=args.water_min_area_pct if water_enabled else 0.0,
        water_morph_iter=args.water_morph_iter if water_enabled else 0,
        water_max_slope_deg=args.water_max_slope_deg if water_enabled else 90.0,
        water_min_depth=args.water_min_depth if water_enabled else 0.0,
        water_debug=args.water_debug if water_enabled else False,
    )

    # Display in viewer or save to file
    if args.viewer_3d:
        # If CLI hasn't started yet, show brief notice
        if not cli_started and verbose:
            print("Initializing viewer and computing terrain texture (first run may take longer)...", flush=True)

        if args.demo_snapshots and verbose:
            print(f"[DEMO] Capturing snapshots to {args.demo_snapshot_dir}...", flush=True)

            def run_demo_after_delay():
                time.sleep(3)  # Give viewer time to initialize
                try:
                    demo_snapshots(
                        args.demo_snapshot_dir,
                        width=args.output_size[0],
                        height=args.output_size[1]
                    )
                except Exception as e:
                    print(f"[DEMO] Error: {e}")

            control_thread = threading.Thread(target=run_demo_after_delay, daemon=True)
            control_thread.start()
        # When not in demo-snapshots mode, the CLI has already started above.

        try:
            f3d.open_terrain_viewer_3d(
                hm,
                texture_rgba=rgba,
                spacing=spacing,
                vertical_scale=args.viewer_vscale,
                subsample=args.viewer_subsample,
                width=args.output_size[0],
                height=args.output_size[1],
                title=f"3D Terrain - {args.src.name}",
            )
        except Exception as exc:
            print(f"Failed to open 3D viewer: {exc}")
            return 1
        finally:
            stop_event.set()
            if control_thread is not None:
                control_thread.join(timeout=1.0)
            if command_stop is not None:
                try:
                    command_stop.set()
                except Exception:
                    pass
            if command_queue is not None:
                try:
                    command_queue.put(None, timeout=0.1)
                except Exception:
                    pass
            if response_queue is not None:
                try:
                    response_queue.put(None)
                except Exception:
                    pass
            if command_proc is not None:
                command_proc.join(timeout=1.0)
            if command_reader_thread is not None:
                command_reader_thread.join(timeout=1.0)
    elif args.viewer:
        if verbose:
            print("Opening 2D image viewer (ESC to close)...")
        try:
            f3d.open_viewer_image(
                rgba,
                width=args.output_size[0],
                height=args.output_size[1],
                title=f"Terrain Render - {args.src.name}",
                vsync=True,
            )
        except Exception as exc:
            print(f"Failed to open viewer: {exc}")
            return 1
    else:
        # Save output
        try:
            f3d.numpy_to_png(str(args.out), rgba)
            if verbose:
                print(f"Wrote {args.out}")
        except Exception:
            try:
                from PIL import Image
                Image.fromarray(rgba, mode='RGBA').save(str(args.out))
                if verbose:
                    print(f"Wrote {args.out}")
            except Exception as exc:
                print(f"Render/save failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
