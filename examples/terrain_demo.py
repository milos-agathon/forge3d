# examples/terrain_demo.py
# Render a high quality PBR terrain from a local DEM using forge3d.
# Exists to demonstrate Session → DEM → PBR render pipeline with custom LUT.
# RELEVANT FILES:examples/_import_shim.py,python/forge3d/__init__.py,python/forge3d/io.py,task.md

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import os

# Only use import shim if explicitly requested (to avoid breaking rasterio imports)
if os.environ.get("USE_IMPORT_SHIM", "").lower() in ("1", "true", "yes"):
    from _import_shim import ensure_repo_import
    ensure_repo_import()

try:
    import forge3d as f3d
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "forge3d import failed. Build the PyO3 extension with `maturin develop --release`."
    ) from exc


DEFAULT_DEM = Path("assets/Gore_Range_Albers_1m.tif")
DEFAULT_HDR = Path("assets/snow_field_4k.hdr")
DEFAULT_OUTPUT = Path("examples/out/terrain_demo.png")
DEFAULT_SIZE = (2560, 1440)
DEFAULT_DOMAIN = (200.0, 2200.0)
DEFAULT_COLORMAP_STOPS: Sequence[tuple[float, str]] = (
    (200.0, "#00aa00"),   # Low elevation: Vibrant green (valleys)
    (800.0, "#80ff00"),   # Mid-low: Bright lime (foothills)
    (1200.0, "#ffff00"),  # Mid: Pure yellow (slopes)
    (1600.0, "#ff8000"),  # Mid-high: Vivid orange (rocky terrain)
    (2000.0, "#ff0000"),  # High: Pure red (peaks)
    (2200.0, "#800000"),  # Highest: Dark red (summits)
)


def _require_attributes(attr_names: Iterable[str]) -> None:
    missing = [name for name in attr_names if not hasattr(f3d, name)]
    if missing:
        raise SystemExit(
            "Required forge3d attributes are missing in this build: "
            + ", ".join(missing)
        )


def _load_dem(path: Path):
    io_mod = getattr(f3d, "io", None)
    if io_mod is None:
        try:
            from forge3d import io as io_mod  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise SystemExit("forge3d.io module is unavailable.") from exc

    load_dem_fn = getattr(io_mod, "load_dem", None)
    if load_dem_fn is None:
        raise SystemExit(
            "forge3d.io.load_dem is not available. Update the forge3d build to a version "
            "that exposes the DEM loader."
        )

    # Load DEM without fill_nodata to avoid import issues with _import_shim
    # The _import_shim adds python/ to sys.path which breaks rasterio.fill import
    dem = load_dem_fn(
        str(path),
        fill_nodata_values=False,
    )

    return dem


def _infer_domain(dem, fallback: tuple[float, float]) -> tuple[float, float]:
    candidates: Sequence[str] = (
        "domain",
        "elevation_range",
        "height_range",
        "bounds",
        "range",
    )
    for name in candidates:
        value = getattr(dem, name, None)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                lo = float(value[0])
                hi = float(value[1])
            except Exception:
                continue
            if hi > lo:
                return (lo, hi)

    min_keys = ("min_elevation", "min_height", "minimum")
    max_keys = ("max_elevation", "max_height", "maximum")
    for min_name in min_keys:
        for max_name in max_keys:
            lo = getattr(dem, min_name, None)
            hi = getattr(dem, max_name, None)
            if lo is None or hi is None:
                continue
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                continue
            if hi_f > lo_f:
                return (lo_f, hi_f)

    stats = getattr(dem, "stats", None)
    if callable(stats):
        try:
            result = stats()
        except Exception:
            result = None
        if isinstance(result, dict):
            lo = result.get("min")
            hi = result.get("max")
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except Exception:
                lo_f = None
                hi_f = None
            if lo_f is not None and hi_f is not None and hi_f > lo_f:
                return (lo_f, hi_f)

    return fallback


def _build_colormap(domain: tuple[float, float], colormap_name: str = "viridis"):
    """
    Build colormap for the DEM domain.

    Args:
        domain: (min_elevation, max_elevation) tuple
        colormap_name: Name of colormap to use ("viridis", "magma", "terrain")
                      OR comma-separated hex colors (e.g., "#ff0000,#00ff00,#0000ff")
    """
    # Check if colormap_name is a custom hex color palette (contains commas)
    if colormap_name and "," in colormap_name:
        # Parse comma-separated hex colors
        hex_colors = [c.strip() for c in colormap_name.split(",")]

        # Validate hex color format
        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        invalid_colors = [c for c in hex_colors if not hex_pattern.match(c)]

        if invalid_colors:
            print(f"Warning: Invalid hex color format: {invalid_colors}")
            print("Expected format: #RRGGBB (e.g., #ff0000)")
            print("Falling back to viridis colormap")
            colormap_name = "viridis"
        else:
            # Create evenly-spaced stops across the domain
            domain_min, domain_max = domain
            domain_range = domain_max - domain_min
            n_colors = len(hex_colors)

            stops = []
            for i, hex_color in enumerate(hex_colors):
                t = i / (n_colors - 1) if n_colors > 1 else 0.0
                elevation = domain_min + t * domain_range
                stops.append((elevation, hex_color))

            print(f"Custom colormap created with {n_colors} colors")
            return f3d.Colormap1D.from_stops(stops=stops, domain=domain)

    if colormap_name and colormap_name != "terrain":
        # Use built-in colormaps (viridis, magma, etc.)
        # These have better color vibrancy than the earth-tone defaults
        try:
            import numpy as np

            cmap = f3d.get_colormap(f"forge3d:{colormap_name}")

            # Convert colormap RGBA array to stops
            # Sample ~16 evenly spaced colors from the 256-color palette
            rgba_array = cmap.rgba  # Shape: (256, 4), dtype: float32, range: [0, 1]
            n_samples = 16
            indices = np.linspace(0, len(rgba_array) - 1, n_samples, dtype=int)

            # Create elevation stops evenly spaced across the domain
            domain_min, domain_max = domain
            domain_range = domain_max - domain_min

            stops = []
            for i, idx in enumerate(indices):
                # Calculate elevation for this stop
                t = i / (n_samples - 1)  # Normalized position [0, 1]
                elevation = domain_min + t * domain_range

                # Get RGBA color and convert to hex
                rgba = rgba_array[idx]
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                hex_color = f"#{r:02x}{g:02x}{b:02x}"

                stops.append((elevation, hex_color))

            return f3d.Colormap1D.from_stops(stops=stops, domain=domain)

        except Exception as e:
            print(f"Warning: Failed to load colormap '{colormap_name}': {e}")
            print("Falling back to terrain colormap (earth tones)")
            colormap_name = "terrain"

    # Fall back to terrain colormap (earth tones)

    # Original domain from the default stops
    original_min = DEFAULT_COLORMAP_STOPS[0][0]
    original_max = DEFAULT_COLORMAP_STOPS[-1][0]
    original_range = original_max - original_min

    # New domain from actual DEM
    new_min = domain[0]
    new_max = domain[1]
    new_range = new_max - new_min

    # Remap all stops proportionally to new domain
    stops = []
    for value, color in DEFAULT_COLORMAP_STOPS:
        # Calculate position in original range [0.0, 1.0]
        t = (value - original_min) / original_range
        # Map to new range
        mapped_value = new_min + t * new_range
        stops.append((mapped_value, color))

    return f3d.Colormap1D.from_stops(
        stops=stops,
        domain=domain,
    )


def _build_params(
    size: tuple[int, int],
    render_scale: float,
    msaa: int,
    z_scale: float,
    exposure: float,
    domain: tuple[float, float],
    colormap,
    albedo_mode: str = "mix",
    colormap_strength: float = 0.5,
):
    config = f3d.TerrainRenderParamsConfig(
        size_px=size,
        render_scale=render_scale,
        msaa_samples=msaa,
        z_scale=z_scale,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=f3d.LightSettings(
            light_type="Directional",
            azimuth_deg=135.0,
            elevation_deg=35.0,
            intensity=3.0,
            color=[1.0, 1.0, 1.0],
        ),
        ibl=f3d.IblSettings(
            enabled=True,
            intensity=1.0,
            rotation_deg=0.0,
        ),
        shadows=f3d.ShadowSettings(
            enabled=True,
            technique="PCSS",
            resolution=4096,
            cascades=3,
            max_distance=4000.0,
            softness=1.5,
            intensity=0.8,
            slope_scale_bias=0.5,
            depth_bias=0.002,
            normal_bias=0.5,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        ),
        triplanar=f3d.TriplanarSettings(
            scale=6.0,
            blend_sharpness=4.0,
            normal_strength=1.0,
        ),
        pom=f3d.PomSettings(
            enabled=True,
            mode="Occlusion",
            scale=0.04,
            min_steps=12,
            max_steps=40,
            refine_steps=4,
            shadow=True,
            occlusion=True,
        ),
        lod=f3d.LodSettings(level=0, bias=0.0, lod0_bias=-0.5),
        sampling=f3d.SamplingSettings(
            mag_filter="Linear",
            min_filter="Linear",
            mip_filter="Linear",
            anisotropy=8,
            address_u="Repeat",
            address_v="Repeat",
            address_w="Repeat",
        ),
        clamp=f3d.ClampSettings(
            height_range=(domain[0], domain[1]),
            slope_range=(0.04, 1.0),
            ambient_range=(0.0, 1.0),
            shadow_range=(0.0, 1.0),
            occlusion_range=(0.0, 1.0),
        ),
        overlays=[
            f3d.OverlayLayer.from_colormap1d(
                colormap,
                strength=1.0,
                offset=0.0,
                blend_mode="Alpha",
                domain=domain,
            )
        ],
        exposure=exposure,
        gamma=2.2,
        albedo_mode=albedo_mode,
        colormap_strength=colormap_strength,
    )

    # Wrap the config in the native TerrainRenderParams
    return f3d.TerrainRenderParams(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a forge3d PBR terrain from a GeoTIFF DEM."
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        help="Path to a GeoTIFF DEM.",
    )
    parser.add_argument(
        "--hdr",
        type=Path,
        default=DEFAULT_HDR,
        help="Path to an environment HDR image.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=DEFAULT_SIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Output resolution in pixels.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help="Internal render scale multiplier.",
    )
    parser.add_argument(
        "--msaa",
        type=int,
        default=4,
        help="MSAA sample count.",
    )
    parser.add_argument(
        "--z-scale",
        dest="z_scale",
        type=float,
        default=1.5,
        help="Vertical exaggeration factor.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="ACES exposure multiplier.",
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
        default="viridis",
        help="Colormap: 'viridis' (default, blue-green-yellow), 'magma' (purple-red-yellow), 'terrain' (green-yellow-orange-red), "
             "or custom hex colors as comma-separated values (e.g., '#ff0000,#00ff00,#0000ff' for red-green-blue gradient).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination image path (PNG/EXR/TIFF).",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Open a display window instead of headless rendering.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive viewer after rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--albedo-mode",
        type=str,
        choices=["material", "colormap", "mix"],
        default="mix",
        help="Albedo source: 'material' (triplanar only), 'colormap' (LUT only), or 'mix' (blend both). Default: mix",
    )
    parser.add_argument(
        "--colormap-strength",
        type=float,
        default=0.5,
        help="Colormap blend strength [0.0-1.0]. Only used with --albedo-mode=mix. Default: 0.5",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Validate colormap_strength range
    if not 0.0 <= args.colormap_strength <= 1.0:
        raise SystemExit(
            f"Error: --colormap-strength must be in range [0.0, 1.0], got {args.colormap_strength}"
        )

    _require_attributes(
        (
            "Session",
            "TerrainRenderer",
            "MaterialSet",
            "IBL",
            "Colormap1D",
            "TerrainRenderParams",
            "LightSettings",
            "IblSettings",
            "ShadowSettings",
            "TriplanarSettings",
            "PomSettings",
            "LodSettings",
            "SamplingSettings",
            "ClampSettings",
            "OverlayLayer",
        )
    )

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {args.output}. Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sess = f3d.Session(window=bool(args.window))

    dem = _load_dem(args.dem)
    # FIX: Pass the numpy array (dem.data) instead of texture_view dict
    # The renderer expects a numpy array to upload to GPU, not a pre-uploaded texture
    heightmap_array = dem.data

    domain = (
        tuple(args.colormap_domain)  # type: ignore[arg-type]
        if args.colormap_domain is not None
        else _infer_domain(dem, DEFAULT_DOMAIN)
    )
    domain = (float(domain[0]), float(domain[1]))

    colormap = _build_colormap(domain, colormap_name=args.colormap)

    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,
        blend_sharpness=4.0,
    )

    ibl = f3d.IBL.from_hdr(
        str(args.hdr),
        intensity=1.0,
        rotate_deg=0.0,
    )

    params = _build_params(
        size=(int(args.size[0]), int(args.size[1])),
        render_scale=float(args.render_scale),
        msaa=int(args.msaa),
        z_scale=float(args.z_scale),
        exposure=float(args.exposure),
        domain=domain,
        colormap=colormap,
        albedo_mode=args.albedo_mode,
        colormap_strength=float(args.colormap_strength),
    )

    renderer = f3d.TerrainRenderer(sess)

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        heightmap=heightmap_array,
        target=None,
    )

    frame.save(str(args.output))
    print(f"Wrote {args.output}")

    if args.viewer:
        viewer_cls = getattr(f3d, "Viewer", None)
        if viewer_cls is None:
            print("forge3d.Viewer is not available in this build. Skipping interactive viewer.")
        else:
            with viewer_cls(sess, renderer, heightmap_array, materials, ibl, params) as view:
                view.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
