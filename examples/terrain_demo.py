"""Thin CLI wrapper for the terrain demo.

This module exists to keep the public demo entry point small and readable while
delegating the heavy lifting to the :mod:`forge3d.terrain_demo` helpers. All
behaviour, including CLI flags and tests that import ``_build_renderer_config``,
is preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import forge3d.terrain_pbr_pom as _impl  # type: ignore[import]
from forge3d.terrain_pbr_pom import render_sunrise_to_noon_sequence


# Re-export selected symbols for tests and external callers
DEFAULT_DEM = _impl.DEFAULT_DEM
DEFAULT_HDR = _impl.DEFAULT_HDR
DEFAULT_OUTPUT = _impl.DEFAULT_OUTPUT
DEFAULT_SIZE = _impl.DEFAULT_SIZE
DEFAULT_DOMAIN = _impl.DEFAULT_DOMAIN


def _build_renderer_config(args: argparse.Namespace):
    """Forward to the implementation module.

    Tests import this symbol from ``terrain_demo``, so we keep the name stable
    and delegate to the real implementation.
    """

    return _impl._build_renderer_config(args)


def _apply_json_preset(args: argparse.Namespace, preset_path: Path, cli_explicit: set[str] | None = None) -> None:
    """Apply JSON preset file parameters to args namespace.
    
    CLI arguments take precedence over preset values (for overrides like --unsharp-strength 0.0).
    
    Args:
        args: Parsed arguments namespace
        preset_path: Path to JSON preset file
        cli_explicit: Set of arg names that were explicitly set on CLI (these won't be overwritten)
    """
    if cli_explicit is None:
        cli_explicit = set()
    
    with open(preset_path) as f:
        preset = json.load(f)
    
    cli_params = preset.get("cli_params", {})
    
    # Map preset keys to args attribute names (and CLI flag names for detection)
    param_map = {
        "dem": ("dem", "--dem"), "hdr": ("hdr", "--hdr"), 
        "size": ("size", "--size"), "msaa": ("msaa", "--msaa"),
        "z_scale": ("z_scale", "--z-scale"), "cam_radius": ("cam_radius", "--cam-radius"),
        "cam_phi": ("cam_phi", "--cam-phi"), "cam_theta": ("cam_theta", "--cam-theta"),
        "exposure": ("exposure", "--exposure"), "ibl_intensity": ("ibl_intensity", "--ibl-intensity"),
        "sun_intensity": ("sun_intensity", "--sun-intensity"),
        "gi": ("gi", "--gi"),
        "sun_azimuth": ("sun_azimuth", "--sun-azimuth"), "sun_elevation": ("sun_elevation", "--sun-elevation"),
        "shadows": ("shadows", "--shadows"), "cascades": ("cascades", "--cascades"),
        "colormap": ("colormap", "--colormap"), "colormap_strength": ("colormap_strength", "--colormap-strength"),
        "albedo_mode": ("albedo_mode", "--albedo-mode"), "normal_strength": ("normal_strength", "--normal-strength"),
        "lambert_contrast": ("lambert_contrast", "--lambert-contrast"), "unsharp_strength": ("unsharp_strength", "--unsharp-strength"),
        "detail_strength": ("detail_strength", "--detail-strength"), "detail_sigma_px": ("detail_sigma_px", "--detail-sigma-px"),
        # P6.1: Color space correctness toggles
        "colormap_srgb": ("colormap_srgb", "--colormap-srgb"),
        "output_srgb_eotf": ("output_srgb_eotf", "--output-srgb-eotf"),
    }
    
    # Get the project root for resolving relative paths
    project_root = Path(__file__).parent.parent
    
    for preset_key, (arg_name, cli_flag) in param_map.items():
        if preset_key not in cli_params:
            continue
        
        # Skip if this arg was explicitly set on CLI
        if arg_name in cli_explicit:
            continue
            
        val = cli_params[preset_key]
        
        # Handle path types
        if preset_key in ("dem", "hdr"):
            val = project_root / val
        elif preset_key == "size":
            val = tuple(val)
        elif preset_key == "gi":
            # gi is a list like ["ibl"], convert to comma-separated string
            if isinstance(val, list):
                val = ",".join(val)
        
        # Set the value
        setattr(args, arg_name, val)
    
    # Handle detail_normals section
    detail_normals = preset.get("detail_normals", {})
    if detail_normals.get("enabled", False):
        detail_path = detail_normals.get("detail_normal_path")
        if detail_path:
            args.detail_normals = project_root / detail_path
        if "detail_strength" in detail_normals:
            args.detail_strength = detail_normals["detail_strength"]


def _parse_args() -> argparse.Namespace:
    """Create the argparse namespace used by ``main``.

    This script is intended to be a readable, tweakable example, so the full
    set of CLI flags is defined here rather than hidden inside the library.
    """

    parser = argparse.ArgumentParser(
        description="Render a forge3d PBR terrain from a GeoTIFF DEM.",
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a GeoTIFF DEM.")
    parser.add_argument("--hdr", type=Path, default=DEFAULT_HDR, help="Path to an environment HDR image.")
    parser.add_argument("--ibl-res", type=int, default=256, help="IBL cubemap base resolution.")
    parser.add_argument("--ibl-cache", type=Path, help="Directory used to persist precomputed IBL results.")
    parser.add_argument("--ibl-intensity", type=float, default=1.0, help="IBL intensity multiplier.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=DEFAULT_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Output resolution in pixels.",
    )
    parser.add_argument("--render-scale", type=float, default=1.0, help="Internal render scale multiplier.")
    parser.add_argument("--msaa", type=int, default=4, help="MSAA sample count.")
    parser.add_argument("--z-scale", dest="z_scale", type=float, default=2.0, help="Vertical exaggeration factor.")
    parser.add_argument("--cam-radius", type=float, default=1000.0, help="Camera radius from target point.")
    parser.add_argument("--cam-phi", type=float, default=135.0, help="Camera azimuth angle in degrees.")
    parser.add_argument("--cam-theta", type=float, default=45.0, help="Camera elevation angle in degrees.")
    parser.add_argument("--exposure", type=float, default=1.0, help="ACES exposure multiplier.")
    parser.add_argument("--sun-azimuth", type=float, help="Sun azimuth angle in degrees.")
    parser.add_argument("--sun-elevation", type=float, help="Sun elevation angle in degrees.")
    parser.add_argument("--sun-intensity", type=float, help="Sun light intensity multiplier.")
    parser.add_argument(
        "--sun-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Sun light color as linear RGB triplet.",
    )
    parser.add_argument(
        "--colormap-domain",
        type=float,
        nargs=2,
        metavar=("MIN_ELEV", "MAX_ELEV"),
        help="Override the colormap elevation domain.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="terrain",
        help=(
            "Colormap name ('terrain', 'viridis', 'magma') or comma-separated hex "
            "colors (e.g. '#ff0000,#00ff00,#0000ff')."
        ),
    )
    parser.add_argument(
        "--colormap-interpolate",
        action="store_true",
        help="Interpolate custom hex colormap stops into a smooth gradient.",
    )
    parser.add_argument("--colormap-size", type=int, default=256, help="Number of colors for custom colormaps.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination image path.")
    parser.add_argument("--window", action="store_true", help="Open a display window instead of headless rendering.")
    parser.add_argument("--viewer", action="store_true", help="Launch the interactive viewer after rendering.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    parser.add_argument(
        "--albedo-mode",
        type=str,
        choices=["material", "colormap", "mix"],
        default="mix",
        help=(
            "Albedo source: 'material' (PBR triplanar textures only), "
            "'colormap' (elevation colormap only, bypasses PBR shading), "
            "or 'mix' (blend between material and colormap using --colormap-strength)."
        ),
    )
    parser.add_argument(
        "--colormap-strength",
        type=float,
        default=0.5,
        help=(
            "Colormap blend strength [0.0-1.0] when using 'mix' albedo mode. "
            "0.0 = full material (no colormap), 1.0 = full colormap."
        ),
    )
    parser.add_argument(
        "--height-curve-mode",
        type=str,
        choices=["linear", "pow", "smoothstep", "lut"],
        default="linear",
        help="Height curve mode for vertical remapping.",
    )
    parser.add_argument(
        "--height-curve-strength",
        type=float,
        default=0.0,
        help="Blend between raw height (0) and curved height (1).",
    )
    parser.add_argument(
        "--height-curve-power",
        type=float,
        default=1.0,
        help="Exponent used when --height-curve-mode=pow.",
    )
    parser.add_argument("--height-curve-lut", type=Path, help="Path to a 256-value LUT used for lut mode.")
    parser.add_argument(
        "--light",
        dest="light",
        action="append",
        default=[],
        metavar="SPEC",
        help="Lighting override in key=value form (repeatable).",
    )
    parser.add_argument("--brdf", type=str, help="Shading BRDF model name.")
    parser.add_argument("--preset", type=str, help="High-level preset (studio_pbr, outdoor_sun, toon_viz).")
    parser.add_argument("--shadows", type=str, help="Shadow technique (none, hard, pcf, pcss, vsm, evsm, msm, csm). 'none' disables shadows for P0 baseline.")
    parser.add_argument("--shadow-map-res", dest="shadow_map_res", type=int, help="Shadow map resolution.")
    parser.add_argument("--cascades", type=int, help="Cascade count for cascaded shadow maps.")
    parser.add_argument("--pcss-blocker-radius", dest="pcss_blocker_radius", type=float, help="PCSS blocker radius.")
    parser.add_argument("--pcss-filter-radius", dest="pcss_filter_radius", type=float, help="PCSS filter radius.")
    parser.add_argument("--shadow-light-size", dest="shadow_light_size", type=float, help="Effective light size.")
    parser.add_argument(
        "--shadow-moment-bias",
        dest="shadow_moment_bias",
        type=float,
        help="Moment bias for VSM/EVSM/MSM.",
    )
    parser.add_argument("--gi", type=str, help="Comma-separated list of GI modes (e.g., ibl,ssao,ssgi).")
    parser.add_argument("--sky", type=str, help="Sky model override (hosek-wilkie, preetham, hdri).")
    parser.add_argument("--volumetric", type=str, help="Volumetric fog settings string.")
    parser.add_argument(
        "--debug-lights",
        action="store_true",
        help="Print light buffer debug info after setting lights (P1-09).",
    )
    parser.add_argument(
        "--pom-disabled",
        action="store_true",
        help="Disable parallax occlusion mapping (POM) in the terrain PBR+POM shader.",
    )
    parser.add_argument(
        "--normal-strength",
        type=float,
        default=1.0,
        help="Terrain normal strength (0.25-4.0). Values > 1.0 amplify normals for steeper shading gradients.",
    )
    parser.add_argument(
        "--lambert-contrast",
        type=float,
        default=0.0,
        help="P5-L: Lambert contrast curve strength (0.0-1.0). Higher values increase micro-contrast in slopes.",
    )
    parser.add_argument(
        "--unsharp-strength",
        type=float,
        default=0.0,
        help="P5-US: Luminance unsharp mask strength (0.0-0.5). Enhances local gradients in post-processing.",
    )
    parser.add_argument(
        "--debug-mode",
        type=int,
        default=0,
        help=(
            "Shader debug mode: 0=normal, 4=water mask binary (CYAN/MAGENTA), "
            "5=raw mask value (grayscale), 6=IBL only, "
            "7=diffuse only, 8=specular only, 9=Fresnel, 10=N.V, 11=roughness, 12=energy, "
            "13=linear combined, 14=linear diffuse, 15=linear specular, 16=recomp error, 17=SpecAA sparkle, "
            "18=POM offset magnitude (grayscale), 19=SpecAA sigma², 20=SpecAA sparkle sigma², "
            "100=water binary (blue/gray), 101=shore-distance falsecolor, 102=IBL spec isolated."
        ),
    )
    # P2: Atmospheric fog parameters
    parser.add_argument(
        "--fog-density",
        type=float,
        default=0.0,
        help="Fog density coefficient. 0.0 = disabled (default), higher = denser fog.",
    )
    parser.add_argument(
        "--fog-height-falloff",
        type=float,
        default=0.0,
        help="Fog height falloff rate. Controls how fog thins at higher altitudes.",
    )
    parser.add_argument(
        "--fog-inscatter",
        type=str,
        default="1.0,1.0,1.0",
        help="Fog inscatter color as comma-separated RGB (0-1 range). Default: '1.0,1.0,1.0' (white).",
    )
    # P6: Detail normal map parameters (Gradient Match)
    parser.add_argument(
        "--detail-normals",
        type=Path,
        default=None,
        help="Path to DEM-derived detail normal map texture for P6 Gradient Match.",
    )
    parser.add_argument(
        "--detail-strength",
        type=float,
        default=0.0,
        help="P6: Detail normal blending strength (0.0-1.0). 0.0 = disabled (default).",
    )
    parser.add_argument(
        "--detail-sigma-px",
        type=float,
        default=3.0,
        help="P6: Gaussian sigma used to generate detail normals (for metadata).",
    )
    # P6: Render mode for validator
    parser.add_argument(
        "--render",
        type=str,
        choices=["p5", "p6"],
        default=None,
        help="Render mode preset: 'p5' (P5.0 baseline) or 'p6' (P6 Gradient Match with detail normals).",
    )
    # P6.1: Color space correctness toggles
    parser.add_argument(
        "--colormap-srgb",
        action="store_true",
        help="P6.1: Use sRGB-correct colormap sampling (Rgba8UnormSrgb texture format).",
    )
    parser.add_argument(
        "--output-srgb-eotf",
        action="store_true",
        help="P6.1: Use exact linear_to_srgb() for output encoding instead of pow-gamma.",
    )

    return parser.parse_args()


def main() -> int:
    """Entry point used by ``python examples/terrain_demo.py``.

    This validates CLI arguments, then forwards rendering to
    :mod:`forge3d.terrain_pbr_pom`.
    """

    args = _parse_args()

    # Handle JSON preset file (--preset path/to/preset.json)
    if args.preset and (args.preset.endswith('.json') or '/' in args.preset or '\\' in args.preset):
        preset_path = Path(args.preset)
        project_root = Path(__file__).parent.parent
        if not preset_path.is_absolute():
            preset_path = project_root / preset_path
        if not preset_path.exists():
            raise SystemExit(f"Error: Preset file not found: {preset_path}")
        print(f"[PRESET] Loading JSON preset: {preset_path}")
        
        # Detect which args were explicitly set on CLI (so they override preset)
        cli_explicit = set()
        cli_flag_to_arg = {
            "--exposure": "exposure", "--ibl-intensity": "ibl_intensity",
            "--gi": "gi",
            "--sun-azimuth": "sun_azimuth", "--sun-elevation": "sun_elevation",
            "--sun-intensity": "sun_intensity", "--z-scale": "z_scale",
            "--cam-radius": "cam_radius", "--cam-phi": "cam_phi", "--cam-theta": "cam_theta",
            "--unsharp-strength": "unsharp_strength", "--detail-strength": "detail_strength",
            "--colormap-strength": "colormap_strength", "--normal-strength": "normal_strength",
            "--lambert-contrast": "lambert_contrast", "--msaa": "msaa",
            "--shadows": "shadows", "--cascades": "cascades",
            "--albedo-mode": "albedo_mode", "--colormap": "colormap",
            # P6.1: Color space correctness toggles
            "--colormap-srgb": "colormap_srgb", "--output-srgb-eotf": "output_srgb_eotf",
        }
        import sys
        for flag, arg_name in cli_flag_to_arg.items():
            if flag in sys.argv:
                cli_explicit.add(arg_name)
        
        _apply_json_preset(args, preset_path, cli_explicit)
        args.preset = None  # Clear so it doesn't interfere with high-level preset logic
        
        # Log resolved GI modes after preset application
        print(f"[GI] modes={args.gi}")

    # Basic validation for a few key numeric flags (delegating the rest).
    if not 0.0 <= args.colormap_strength <= 1.0:
        raise SystemExit(
            f"Error: --colormap-strength must be in range [0.0, 1.0], got {args.colormap_strength}"
        )
    if not 0.0 <= args.height_curve_strength <= 1.0:
        raise SystemExit(
            f"Error: --height-curve-strength must be in range [0.0, 1.0], got {args.height_curve_strength}"
        )
    if args.height_curve_power <= 0.0:
        raise SystemExit(
            f"Error: --height-curve-power must be greater than zero, got {args.height_curve_power}"
        )

    # Set debug mode via environment variable (read by terrain shader)
    import os
    debug_mode = getattr(args, "debug_mode", 0)
    if debug_mode != 0:
        os.environ["VF_COLOR_DEBUG_MODE"] = str(debug_mode)
        print(f"[DEBUG] Setting VF_COLOR_DEBUG_MODE={debug_mode}")

    return _impl.run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
