"""Thin CLI wrapper for the terrain demo.

This module exists to keep the public demo entry point small and readable while
delegating the heavy lifting to the :mod:`forge3d.terrain_demo` helpers. All
behaviour, including CLI flags and tests that import ``_build_renderer_config``,
is preserved.
"""

from __future__ import annotations

import argparse
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
        help="Albedo source: material, colormap, or mix (default).",
    )
    parser.add_argument(
        "--colormap-strength",
        type=float,
        default=0.5,
        help="Colormap blend strength [0.0-1.0] when using mix albedo mode.",
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
    parser.add_argument("--shadows", type=str, help="Shadow technique (hard, pcf, pcss, vsm, evsm, msm, csm).")
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
        "--water-detect",
        action="store_true",
        help="Detect water from DEM and tint water bodies in the main output.",
    )
    parser.add_argument(
        "--water-level",
        type=float,
        default=0.35,
        help="Normalized water height in [0,1] relative to inferred DEM domain.",
    )
    parser.add_argument(
        "--water-slope",
        type=float,
        default=0.02,
        help="Slope threshold in normalized units for water detection.",
    )
    parser.add_argument(
        "--water-mask-output",
        type=Path,
        help="Optional debug PNG showing the detected water mask.",
    )
    parser.add_argument(
        "--water-material",
        type=str,
        choices=["none", "overlay", "pbr"],
        default="overlay",
        help=(
            "Water rendering mode when --water-detect is enabled: "
            "'none' (no highlight), 'overlay' (2D tint, default), or 'pbr' "
            "(GPU PBR water using the terrain shader)."
        ),
    )
    parser.add_argument(
        "--pom-disabled",
        action="store_true",
        help="Disable parallax occlusion mapping (POM) in the terrain PBR+POM shader.",
    )

    return parser.parse_args()


def main() -> int:
    """Entry point used by ``python examples/terrain_demo.py``.

    This validates CLI arguments, then forwards rendering to
    :mod:`forge3d.terrain_pbr_pom`.
    """

    args = _parse_args()

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

    return _impl.run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
