# examples/m3_geopandas_demo.py
# Updated: Load a GeoTIFF DEM (Gore_Range_Albers_1m.tif) and render with a custom palette.
# - Reads elevation from assets/Gore_Range_Albers_1m.tif
# - Uses an interpolated 128-color palette built from provided hex stops
# - Saves an RGBA preview PNG

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Union

import numpy as np

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover
    raise ImportError("forge3d Python API is required. Ensure package is installed or built.") from exc


CUSTOM_HEX_COLORS: Sequence[str] = (
"#e7d8a2", "#c5a06e", "#995f57", "#4a3c37"
)


def load_dem(src_path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Load a DEM from GeoTIFF and return (heightmap32, pixel_spacing_xy)."""

    try:
        import rasterio
    except Exception as exc:  # pragma: no cover
        raise ImportError("rasterio is required. Install with: pip install rasterio") from exc

    with rasterio.open(str(src_path)) as ds:
        band1 = ds.read(1, masked=True)
        data = np.array(band1.filled(np.nan), dtype=np.float32)
        if np.isnan(data).any():
            finite = data[np.isfinite(data)]
            fill_val = float(np.min(finite)) if finite.size else 0.0
            data = np.nan_to_num(data, nan=fill_val)
        try:
            sx = float(ds.transform.a)
            sy = float(-ds.transform.e)
        except Exception:
            sx, sy = 1.0, 1.0
        sx = abs(sx) or 1.0
        sy = abs(sy) or 1.0
        return data, (sx, sy)


def _resolve_palette_argument(colormap: str, interpolate: bool = False, size: int = 256) -> Union[str, Sequence[str]]:
    """Resolve palette argument from colormap name.

    Args:
        colormap: Name of colormap ("custom" or built-in name)
        interpolate: If True, interpolate custom palette to create smooth gradients
        size: Number of colors when interpolating

    Returns:
        Either colormap name (str) or list of hex colors
    """
    if colormap.lower() != "custom":
        return colormap

    # Use custom palette
    base_colors = CUSTOM_HEX_COLORS

    if not interpolate or len(base_colors) < 2:
        # Return discrete colors as-is
        return base_colors

    # Interpolate colors to create smooth gradients
    try:
        # Convert hex to RGB
        rgb_colors = []
        for hex_color in base_colors:
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            rgb_colors.append((r, g, b))

        # Interpolate in RGB space
        num_steps = int(size)
        indices = np.linspace(0, len(rgb_colors) - 1, num_steps)

        interpolated = []
        for idx in indices:
            i0 = int(np.floor(idx))
            i1 = min(i0 + 1, len(rgb_colors) - 1)
            t = idx - i0

            # Linear interpolation
            r = rgb_colors[i0][0] * (1 - t) + rgb_colors[i1][0] * t
            g = rgb_colors[i0][1] * (1 - t) + rgb_colors[i1][1] * t
            b = rgb_colors[i0][2] * (1 - t) + rgb_colors[i1][2] * t

            # Convert back to hex
            hex_val = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            interpolated.append(hex_val)

        return interpolated
    except Exception:
        # Fallback to discrete colors
        return base_colors


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 DEM → Terrain render demo (Gore Range 1m)")
    parser.add_argument("--src", type=Path, default=Path("assets/Gore_Range_Albers_1m.tif"), help="Input GeoTIFF DEM")
    parser.add_argument("--out", type=Path, default=Path("reports/Gore_Range_Albers_1m.png"), help="Output PNG path")
    parser.add_argument("--output-size", type=int, nargs=2, default=(800, 600), metavar=("W", "H"), help="Output size (pixels)")

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
    
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load DEM
    try:
        hm, spacing = load_dem(args.src)
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

    # Resolve palette with interpolation settings
    palette = _resolve_palette_argument(
        args.colormap,
        interpolate=args.palette_interpolate,
        size=args.palette_size
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

    # Save output
    try:
        f3d.numpy_to_png(str(args.out), rgba)
        print(f"Wrote {args.out}")
    except Exception:
        try:
            from PIL import Image
            Image.fromarray(rgba, mode='RGBA').save(str(args.out))
            print(f"Wrote {args.out}")
        except Exception as exc:
            print(f"Render/save failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
