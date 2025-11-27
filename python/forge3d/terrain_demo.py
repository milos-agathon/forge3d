from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import argparse
import math
import os

import numpy as np

import forge3d as f3d
from . import io as _io
from .config import load_renderer_config
from .terrain_params import (
    ShadowSettings as TerrainShadowSettings,
    load_height_curve_lut,
    make_terrain_params_config,
)
from .colormaps.core import (
    interpolate_hex_colors as _cm_interpolate_hex_colors,
    elevation_stops_from_hex_colors as _cm_elevation_stops,
)
from .render import (
    detect_dem_water_mask as _detect_dem_water_mask,
    _load_dem as _render_load_dem,
    _postprocess_water_mask as _postprocess_water_mask,
)
from .lighting import sun_direction_from_angles as _sun_direction_from_angles


DEFAULT_DEM = Path("assets/Gore_Range_Albers_1m.tif")
DEFAULT_HDR = Path("assets/snow_field_4k.hdr")
DEFAULT_OUTPUT = Path("examples/out/terrain_demo.png")
DEFAULT_SIZE = (1920, 1080)
DEFAULT_DOMAIN = (200.0, 2200.0)
DEFAULT_COLORMAP_STOPS: Sequence[tuple[float, str]] = (
    (200.0, "#00aa00"),   # Low elevation: Vibrant green (valleys)
    (800.0, "#80ff00"),   # Mid-low: Bright lime (foothills)
    (1200.0, "#ffff00"),  # Mid: Pure yellow (slopes)
    (1600.0, "#ff8000"),  # Mid-high: Vivid orange (rocky terrain)
    (2000.0, "#ff0000"),  # High: Pure red (peaks)
    (2200.0, "#800000"),  # Highest: Dark red (summits)
)

QUANTILE_DEFAULT_LO = 0.0
QUANTILE_DEFAULT_HI = 1.0


def _require_attributes(attr_names: Iterable[str]) -> None:
    missing = [name for name in attr_names if not hasattr(f3d, name)]
    if missing:
        raise SystemExit(
            "Required forge3d attributes are missing in this build: " + ", ".join(missing)
        )


def _load_dem(path: Path):
    dem = _io.load_dem(str(path), fill_nodata_values=True)
    data = getattr(dem, "data", None)
    if data is None:
        raise SystemExit("DEM object does not expose a .data array")
    # Match legacy terrain_demo orientation
    dem.data = np.flipud(np.asarray(data, dtype=np.float32)).copy()
    return dem


def _build_renderer_config(args: Any):
    overrides: dict[str, Any] = {}
    if getattr(args, "light", None):
        overrides["lights"] = [_parse_light_spec(spec) for spec in args.light]
    overrides["exposure"] = float(args.exposure)
    if getattr(args, "brdf", None):
        overrides["brdf"] = args.brdf
    if getattr(args, "shadows", None):
        overrides["shadows"] = args.shadows
    if getattr(args, "shadow_map_res", None) is not None:
        overrides["shadow_map_res"] = int(args.shadow_map_res)
    if getattr(args, "cascades", None) is not None:
        overrides["cascades"] = int(args.cascades)
    if getattr(args, "pcss_blocker_radius", None) is not None:
        overrides["pcss_blocker_radius"] = float(args.pcss_blocker_radius)
    if getattr(args, "pcss_filter_radius", None) is not None:
        overrides["pcss_filter_radius"] = float(args.pcss_filter_radius)
    if getattr(args, "shadow_light_size", None) is not None:
        overrides["light_size"] = float(args.shadow_light_size)
    if getattr(args, "shadow_moment_bias", None) is not None:
        overrides["moment_bias"] = float(args.shadow_moment_bias)
    if getattr(args, "gi", None):
        modes = [item.strip() for item in str(args.gi).split(",") if item.strip()]
        overrides["gi"] = modes
    if getattr(args, "sky", None):
        overrides["sky"] = args.sky
    if getattr(args, "hdr", None):
        overrides["hdr"] = args.hdr
    if getattr(args, "volumetric", None):
        overrides["volumetric"] = _parse_volumetric_spec(str(args.volumetric))
    if getattr(args, "preset", None):
        try:
            from . import presets as _presets

            preset_map = _presets.get(str(args.preset))
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Unknown or unavailable preset '{args.preset}': {exc}")
        return load_renderer_config(preset_map, overrides)
    return load_renderer_config(None, overrides)


def _split_key_value_string(spec: str) -> dict[str, str]:
    tokens = spec.split(",")
    result: dict[str, str] = {}
    current_key: str | None = None
    for raw in tokens:
        segment = raw.strip()
        if not segment:
            continue
        if "=" in segment:
            key, value = segment.split("=", 1)
            current_key = key.strip().lower()
            result[current_key] = value.strip()
        elif current_key is not None:
            result[current_key] = f"{result[current_key]},{segment}"
        else:
            raise ValueError(f"Invalid segment '{segment}' in specification '{spec}'")
    return result


def _parse_float_list(value: str, length: int, label: str) -> tuple[float, ...]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise ValueError(f"{label} requires exactly {length} comma-separated floats")
    return tuple(parts)


def _parse_light_spec(spec: str) -> dict[str, Any]:
    mapping = _split_key_value_string(spec)
    out: dict[str, Any] = {}
    for key, val in mapping.items():
        if key in {"type", "light"}:
            out["type"] = val
        elif key in {"dir", "direction"}:
            out["direction"] = _parse_float_list(val, 3, "direction")
        elif key in {"pos", "position"}:
            out["position"] = _parse_float_list(val, 3, "position")
        elif key in {"intensity", "power"}:
            out["intensity"] = float(val)
        elif key in {"color", "rgb"}:
            out["color"] = _parse_float_list(val, 3, "color")
        elif key in {"cone", "cone_angle", "angle"}:
            out["cone_angle"] = float(val)
        elif key in {"area", "extent", "area_extent"}:
            out["area_extent"] = _parse_float_list(val, 2, "area extent")
        elif key in {"hdr", "hdr_path"}:
            out["hdr_path"] = val
    return out


def _parse_volumetric_spec(spec: str) -> dict[str, Any]:
    mapping = _split_key_value_string(spec)
    out: dict[str, Any] = {}
    if "density" in mapping:
        out["density"] = float(mapping["density"])
    if "phase" in mapping:
        out["phase"] = mapping["phase"]
    if "g" in mapping:
        out["g"] = float(mapping["g"])
    if "anisotropy" in mapping:
        out["anisotropy"] = float(mapping["anisotropy"])
    return out


def _build_colormap(
    domain: tuple[float, float],
    colormap_name: str = "viridis",
    *,
    heightmap=None,
    q_lo: float = QUANTILE_DEFAULT_LO,
    q_hi: float = QUANTILE_DEFAULT_HI,
    interpolate_custom: bool = False,
    custom_size: int = 256,
):
    """Build a Colormap1D for the DEM elevation domain."""

    def quantile_stops(colors: list[str]):
        if heightmap is None:
            return None
        h = np.asarray(heightmap, dtype=np.float32).reshape(-1)
        h = h[np.isfinite(h)]
        if h.size == 0:
            return None
        qs = np.linspace(q_lo, q_hi, len(colors))
        elevs = np.quantile(h, qs)
        return list(zip([float(e) for e in elevs], colors))

    # Custom hex palette
    if colormap_name and "," in colormap_name:
        hex_colors = [c.strip() for c in colormap_name.split(",")]

        import re

        hex_pattern = re.compile(r"^[0-9A-Fa-f]{6}$")
        invalid_colors = [c for c in hex_colors if not hex_pattern.search(c.lstrip("#"))]
        if invalid_colors:
            print(f"Warning: Invalid hex color format: {invalid_colors}")
            print("Expected format: #RRGGBB (e.g., #ff0000)")
            print("Falling back to viridis colormap")
            colormap_name = "viridis"
        else:
            if interpolate_custom and len(hex_colors) >= 2:
                hex_colors = _cm_interpolate_hex_colors(hex_colors, size=custom_size)

            stops = _cm_elevation_stops(
                domain,
                hex_colors,
                heightmap=heightmap,
                q_lo=q_lo,
                q_hi=q_hi,
            )
            print(f"Custom colormap created with {len(hex_colors)} colors")
            return f3d.Colormap1D.from_stops(stops=stops, domain=domain)

    # Named palette (viridis, magma, ...)
    if colormap_name and colormap_name != "terrain":
        try:
            cmap = f3d.get_colormap(f"forge3d:{colormap_name}")
            rgba_array = cmap.rgba  # (N,4)
            n_samples = 1024
            indices = np.linspace(0, len(rgba_array) - 1, n_samples, dtype=int)
            colors_hex: list[str] = []
            for idx in indices:
                rgba = rgba_array[idx]
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")

            stops = _cm_elevation_stops(
                domain,
                colors_hex,
                heightmap=heightmap,
                q_lo=q_lo,
                q_hi=q_hi,
            )
            return f3d.Colormap1D.from_stops(stops=stops, domain=domain)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Failed to load colormap '{colormap_name}': {exc}")
            print("Falling back to terrain colormap (earth tones)")
            colormap_name = "terrain"

    # Fallback: terrain colormap using DEFAULT_COLORMAP_STOPS
    original_min = DEFAULT_COLORMAP_STOPS[0][0]
    original_max = DEFAULT_COLORMAP_STOPS[-1][0]
    original_range = original_max - original_min

    new_min, new_max = float(domain[0]), float(domain[1])
    new_range = new_max - new_min

    stops = []
    for value, color in DEFAULT_COLORMAP_STOPS:
        t = (value - original_min) / original_range
        mapped_value = new_min + t * new_range
        stops.append((mapped_value, color))

    return f3d.Colormap1D.from_stops(stops=stops, domain=domain)


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
    ibl_enabled: bool = True,
    light_azimuth_deg: float = 135.0,
    light_elevation_deg: float = 35.0,
    sun_intensity: float = 3.0,
    sun_color: Sequence[float] | None = None,
    ibl_intensity: float = 1.0,
    cam_radius: float = 1200.0,
    cam_phi_deg: float = 135.0,
    cam_theta_deg: float = 45.0,
    height_curve_mode: str = "linear",
    height_curve_strength: float = 0.0,
    height_curve_power: float = 1.0,
    height_curve_lut=None,
):
    overlays = [
        f3d.OverlayLayer.from_colormap1d(
            colormap,
            strength=1.0,
            offset=0.0,
            blend_mode="Alpha",
            domain=domain,
        )
    ]

    config = make_terrain_params_config(
        size_px=size,
        render_scale=render_scale,
        msaa_samples=msaa,
        z_scale=z_scale,
        exposure=exposure,
        domain=domain,
        albedo_mode=albedo_mode,
        colormap_strength=colormap_strength,
        ibl_enabled=ibl_enabled,
        light_azimuth_deg=light_azimuth_deg,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=sun_intensity,
        sun_color=sun_color,
        ibl_intensity=ibl_intensity,
        cam_radius=cam_radius,
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        height_curve_mode=height_curve_mode,
        height_curve_strength=height_curve_strength,
        height_curve_power=height_curve_power,
        height_curve_lut=height_curve_lut,
        shadows=TerrainShadowSettings(
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
        overlays=overlays,
    )

    return f3d.TerrainRenderParams(config)


def _save_image(img, path: Path) -> None:
    try:
        from PIL import Image  # type: ignore[import]

        Image.fromarray(img, mode="RGBA").save(str(path))
    except Exception:
        try:
            f3d.numpy_to_png(str(path), img)
        except Exception:
            np.save(str(path).replace(".png", ".npy"), img)
            print("  Warning: Saved as .npy (no PNG writer available)")


def _save_water_mask(mask, path: Path) -> None:
    h, w = mask.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    img[mask, 0:3] = (0, 0, 255)
    _save_image(img, path)


def _resize_mask_to_frame(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to match an output (H, W) using simple nearest-neighbor.

    This keeps dependencies minimal and is sufficient for visually highlighting
    water regions in the final terrain render.
    """

    mask_np = np.asarray(mask, dtype=bool)
    mh, mw = mask_np.shape
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if mh == out_h and mw == out_w:
        return mask_np

    if mh <= 0 or mw <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid mask or output shape for resize")

    row_idx = (np.arange(out_h, dtype=np.float32) * (mh / float(out_h))).astype(np.int64)
    col_idx = (np.arange(out_w, dtype=np.float32) * (mw / float(out_w))).astype(np.int64)
    row_idx = np.clip(row_idx, 0, mh - 1)
    col_idx = np.clip(col_idx, 0, mw - 1)
    return mask_np[row_idx[:, None], col_idx[None, :]]


def _remove_border_connected(mask: np.ndarray) -> np.ndarray:
    """Clear all mask components that touch a band near the image border.

    We use 4-connectivity and treat a small band around the outer rim as the
    "border" so that large ocean-like regions that stop a few pixels short of
    the exact edge are still removed. Complexity is O(H*W).
    """

    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m

    h, w = m.shape
    visited = np.zeros_like(m, dtype=bool)
    out = m.copy()

    from collections import deque

    q: "deque[tuple[int, int]]" = deque()

    def enqueue(i: int, j: int) -> None:
        if 0 <= i < h and 0 <= j < w and not visited[i, j] and out[i, j]:
            visited[i, j] = True
            q.append((i, j))

    # Treat a small band near the outer rim as the "border" for seeding.
    band = max(1, min(h, w) // 128)  # ~31px for 4000x4000, 6px for 800x600

    # Top / bottom bands
    for j in range(w):
        for di in range(band):
            if out[di, j]:
                enqueue(di, j)
            if out[h - 1 - di, j]:
                enqueue(h - 1 - di, j)

    # Left / right bands
    for i in range(h):
        for dj in range(band):
            if out[i, dj]:
                enqueue(i, dj)
            if out[i, w - 1 - dj]:
                enqueue(i, w - 1 - dj)

    # Flood-fill and clear all border-connected pixels.
    while q:
        i, j = q.popleft()
        out[i, j] = False
        if i > 0:
            enqueue(i - 1, j)
        if i + 1 < h:
            enqueue(i + 1, j)
        if j > 0:
            enqueue(i, j - 1)
        if j + 1 < w:
            enqueue(i, j + 1)

    return out


def _apply_water_tint(
    rgba: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (30, 80, 200),
    alpha: float = 0.7,
) -> np.ndarray:
    """Overlay a blue-ish tint on water pixels in the RGBA image.

    The tint is applied in-place on a copy of ``rgba`` using simple linear
    interpolation between the original color and the water color.
    """

    img = np.asarray(rgba, dtype=np.uint8).copy()
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("Expected RGBA image with shape (H, W, 4)")

    h, w, _ = img.shape
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.shape != (h, w):
        raise ValueError("Water mask shape must match RGBA image shape")

    if not np.any(mask_np):
        return img

    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        a = 0.7

    water_rgb = img[mask_np, :3].astype(np.float32)
    tint_rgb = np.array(color, dtype=np.float32)[None, :]
    blended = (1.0 - a) * water_rgb + a * tint_rgb
    img[mask_np, :3] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    # Ensure NA water regions, which may have been rendered fully transparent
    # (alpha=0), become visible after tinting by forcing opaque alpha.
    img[mask_np, 3] = 255
    return img


def render_sunrise_to_noon_sequence(
    *,
    dem_path: Path,
    hdr_path: Path,
    output_dir: Path,
    size: tuple[int, int] = (320, 180),
    steps: int = 4,
) -> list[Path]:
    """Render a short sunrise-to-noon sequence using RendererConfig."""

    output_dir.mkdir(parents=True, exist_ok=True)

    dem = _load_dem(dem_path)
    heightmap_array = getattr(dem, "data", None)
    if heightmap_array is None:
        raise SystemExit("DEM object does not expose a .data array")

    domain_meta = _io.infer_dem_domain(dem, fallback=DEFAULT_DOMAIN)
    domain = _io.robust_dem_domain(
        heightmap_array,
        q_lo=QUANTILE_DEFAULT_LO,
        q_hi=QUANTILE_DEFAULT_HI,
        fallback=(float(domain_meta[0]), float(domain_meta[1])),
    )
    domain = (float(domain[0]), float(domain[1]))
    colormap = _build_colormap(domain, colormap_name="terrain", heightmap=heightmap_array)

    base_overrides: dict[str, Any] = {
        "sky": "hosek-wilkie",
        "hdr": str(hdr_path),
        "gi": ["ibl"],
        "volumetric": {
            "density": 0.03,
            "phase": "hg",
            "g": 0.7,
            "mode": "raymarch",
            "max_steps": 48,
        },
    }

    use_native = bool(f3d.has_gpu()) and all(
        hasattr(f3d, name)
        for name in ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    )

    sess = None
    renderer_native = None
    materials = None
    ibl = None

    if use_native:
        sess = f3d.Session(window=False)
        renderer_native = f3d.TerrainRenderer(sess)
        materials = f3d.MaterialSet.terrain_default(
            triplanar_scale=6.0,
            normal_strength=1.0,
            blend_sharpness=4.0,
        )
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        ibl.set_base_resolution(256)

    if steps <= 1:
        sun_elevations: list[float] = [30.0]
    else:
        sun_elevations = np.linspace(5.0, 60.0, int(steps), dtype=float).tolist()

    azimuth_deg = 135.0
    width, height = int(size[0]), int(size[1])
    outputs: list[Path] = []

    for idx, elev in enumerate(sun_elevations):
        sun_dir = _sun_direction_from_angles(azimuth_deg, float(elev))

        overrides = dict(base_overrides)
        overrides["light"] = [
            {
                "type": "directional",
                "direction": list(sun_dir),
                "intensity": 5.0,
                "color": [1.0, 0.97, 0.94],
            }
        ]

        cfg = load_renderer_config(None, overrides)

        if use_native and sess is not None and renderer_native is not None and materials is not None and ibl is not None:
            ibl_enabled = "ibl" in cfg.gi.modes
            params = _build_params(
                size=(width, height),
                render_scale=1.0,
                msaa=1,
                z_scale=1.0,
                exposure=float(cfg.lighting.exposure),
                domain=domain,
                colormap=colormap,
                albedo_mode="mix",
                colormap_strength=0.5,
                ibl_enabled=ibl_enabled,
                light_azimuth_deg=azimuth_deg,
                light_elevation_deg=elev,
            )

            frame = renderer_native.render_terrain_pbr_pom(
                material_set=materials,
                env_maps=ibl if ibl_enabled else None,
                params=params,
                heightmap=heightmap_array,
                target=None,
            )
            rgba = frame.to_numpy()
        else:
            renderer_fallback = f3d.Renderer(width, height, config=cfg)
            rgba = renderer_fallback.render_triangle_rgba()

        out_path = output_dir / f"terrain_sunrise_{idx:02d}.png"
        _save_image(rgba, out_path)
        outputs.append(out_path)

    return outputs


def run(args: Any) -> int:
    """Execute the terrain demo render using an argparse-style args object."""

    height_curve_lut = None
    if getattr(args, "height_curve_mode", None) == "lut":
        if getattr(args, "height_curve_lut", None) is None:
            raise SystemExit(
                "Error: --height-curve-lut is required when --height-curve-mode=lut"
            )
        height_curve_lut = load_height_curve_lut(args.height_curve_lut)

    water_mask = None

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

    renderer_config = _build_renderer_config(args)

    if args.output.exists() and not getattr(args, "overwrite", False):
        raise SystemExit(
            f"Output file already exists: {args.output}. Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sess = f3d.Session(window=bool(getattr(args, "window", False)))

    dem = _load_dem(args.dem)
    heightmap_array = dem.data

    if getattr(args, "colormap_domain", None) is not None:
        domain_meta = tuple(args.colormap_domain)  # type: ignore[arg-type]
    else:
        domain_meta = _io.infer_dem_domain(dem, fallback=DEFAULT_DOMAIN)

    domain = _io.robust_dem_domain(
        heightmap_array,
        q_lo=QUANTILE_DEFAULT_LO,
        q_hi=QUANTILE_DEFAULT_HI,
        fallback=(float(domain_meta[0]), float(domain_meta[1])),
    )
    domain = (float(domain[0]), float(domain[1]))

    colormap = _build_colormap(
        domain,
        colormap_name=args.colormap,
        heightmap=heightmap_array,
        interpolate_custom=bool(getattr(args, "colormap_interpolate", False)),
        custom_size=int(getattr(args, "colormap_size", 256)),
    )

    if getattr(args, "water_detect", False):
        water_mask = None

        # 1) Prefer the DEM's own mask (NA pixels) as water, matching the
        # geopandas_demo / render_raster behaviour.
        try:
            import rasterio  # type: ignore[import]

            with rasterio.open(str(args.dem)) as ds:  # type: ignore[attr-defined]
                band1 = ds.read(1, masked=True)
                band_mask = getattr(band1, "mask", None)
                if band_mask is not None:
                    m = np.asarray(band_mask, dtype=bool)
                    if m.size:
                        # Match the flipped orientation used for heightmap_array.
                        water_mask = np.flipud(m)
        except Exception:
            water_mask = None

        # 2) Fallback: heuristic DEM water mask, using the same DEM loading
        #    semantics as forge3d.render._load_dem.
        if water_mask is None or not np.any(water_mask):
            try:
                hm_water, spacing_water = _render_load_dem(Path(args.dem))
                hm_water = np.flipud(np.asarray(hm_water, dtype=np.float32))
                water_domain = _io.robust_dem_domain(
                    hm_water,
                    q_lo=QUANTILE_DEFAULT_LO,
                    q_hi=QUANTILE_DEFAULT_HI,
                    fallback=(float(domain_meta[0]), float(domain_meta[1])),
                )
                water_domain = (float(water_domain[0]), float(water_domain[1]))
                water_mask = _detect_dem_water_mask(
                    hm_water,
                    water_domain,
                    level_normalized=float(args.water_level),
                    slope_threshold=float(args.water_slope),
                    spacing=spacing_water,
                    base_min_area_pct=0.0,
                    keep_components=16,
                )
            except Exception:
                water_mask = _detect_dem_water_mask(
                    heightmap_array,
                    domain,
                    level_normalized=float(args.water_level),
                    slope_threshold=float(args.water_slope),
                    spacing=getattr(dem, "resolution", (1.0, 1.0)),
                    base_min_area_pct=0.0,
                    keep_components=16,
                )

    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,
        blend_sharpness=4.0,
    )

    hdr_path_value: Path | str = args.hdr
    if renderer_config.atmosphere.hdr_path is not None:
        hdr_path_value = Path(renderer_config.atmosphere.hdr_path)
    else:
        env_hdr = next(
            (
                light.hdr_path
                for light in renderer_config.lighting.lights
                if light.type == "environment" and light.hdr_path is not None
            ),
            None,
        )
        if env_hdr is not None:
            hdr_path_value = Path(env_hdr)

    ibl = f3d.IBL.from_hdr(
        str(hdr_path_value),
        intensity=float(args.ibl_intensity),
        rotate_deg=0.0,
    )
    if args.ibl_res <= 0:
        raise SystemExit("Error: --ibl-res must be greater than zero.")
    ibl.set_base_resolution(int(args.ibl_res))
    if getattr(args, "ibl_cache", None) is not None:
        ibl.set_cache_dir(str(args.ibl_cache))

    ibl_enabled = True

    sun_azimuth_deg = float(args.sun_azimuth) if getattr(args, "sun_azimuth", None) is not None else 135.0
    sun_elevation_deg = float(args.sun_elevation) if getattr(args, "sun_elevation", None) is not None else 35.0
    sun_intensity = float(args.sun_intensity) if getattr(args, "sun_intensity", None) is not None else 3.0
    sun_color = tuple(args.sun_color) if getattr(args, "sun_color", None) is not None else None

    params = _build_params(
        size=(int(args.size[0]), int(args.size[1])),
        render_scale=float(args.render_scale),
        msaa=int(args.msaa),
        z_scale=float(args.z_scale),
        exposure=float(renderer_config.lighting.exposure),
        domain=domain,
        colormap=colormap,
        albedo_mode=args.albedo_mode,
        colormap_strength=float(args.colormap_strength),
        ibl_enabled=ibl_enabled,
        light_azimuth_deg=sun_azimuth_deg,
        light_elevation_deg=sun_elevation_deg,
        sun_intensity=sun_intensity,
        sun_color=sun_color,
        ibl_intensity=float(args.ibl_intensity),
        cam_radius=float(args.cam_radius),
        cam_phi_deg=float(args.cam_phi),
        cam_theta_deg=float(args.cam_theta),
        height_curve_mode=str(args.height_curve_mode),
        height_curve_strength=float(args.height_curve_strength),
        height_curve_power=float(args.height_curve_power),
        height_curve_lut=height_curve_lut,
    )

    renderer = f3d.TerrainRenderer(sess)

    if renderer_config.lighting.lights:
        light_dicts = []
        for light in renderer_config.lighting.lights:
            d: dict[str, Any] = {"type": light.type}
            if getattr(light, "azimuth", None) is not None:
                d["azimuth"] = light.azimuth
            if getattr(light, "elevation", None) is not None:
                d["elevation"] = light.elevation
            if getattr(light, "position", None) is not None:
                d["position"] = light.position
            if getattr(light, "direction", None) is not None:
                d["direction"] = light.direction
            if getattr(light, "intensity", None) is not None:
                d["intensity"] = light.intensity
            if getattr(light, "color", None) is not None:
                try:
                    d["color"] = list(light.color)
                except TypeError:
                    d["color"] = light.color
            if getattr(light, "range", None) is not None:
                d["range"] = light.range
            if getattr(light, "cone_angle", None) is not None:
                d["cone_angle"] = light.cone_angle
            if getattr(light, "inner_angle", None) is not None:
                d["inner_angle"] = light.inner_angle
            if getattr(light, "outer_angle", None) is not None:
                d["outer_angle"] = light.outer_angle
            if getattr(light, "area_extent", None) is not None:
                d["area_extent"] = light.area_extent
            if getattr(light, "radius", None) is not None:
                d["radius"] = light.radius
            light_dicts.append(d)

        try:
            renderer.set_lights(light_dicts)
            print(f"[P1-09] Uploaded {len(light_dicts)} light(s) to GPU")
            if getattr(args, "debug_lights", False):
                print("\n" + "=" * 60)
                print("Light Buffer Debug Info:")
                print("=" * 60)
                print(renderer.light_debug_info())
                print("=" * 60 + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Failed to upload lights: {exc}")

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl if ibl_enabled else None,
        params=params,
        heightmap=heightmap_array,
        target=None,
    )

    if water_mask is not None:
        rgba = frame.to_numpy()

        # Start from the DEM-based water mask, resized to frame resolution.
        mask_resized = _resize_mask_to_frame(water_mask, rgba.shape[:2])

        # Also treat very dark pixels in the rendered image as lake candidates.
        # We compute a grayscale brightness image and pick the ~0.5%% darkest
        # interior pixels (excluding a small border) so that deep inland
        # basins like the kidney-shaped lake are captured.
        img_u8 = np.asarray(rgba, dtype=np.uint8)
        gray = (
            img_u8[..., 0].astype(np.float32)
            + img_u8[..., 1].astype(np.float32)
            + img_u8[..., 2].astype(np.float32)
        ) / 3.0

        lake_mask = np.zeros_like(gray, dtype=bool)
        if gray.size:
            inner = np.ones_like(gray, dtype=bool)
            h_i, w_i = gray.shape
            if h_i > 6 and w_i > 6:
                inner[:3, :] = False
                inner[-3:, :] = False
                inner[:, :3] = False
                inner[:, -3:] = False
            vals = gray[inner]
            if vals.size:
                try:
                    thresh = float(np.quantile(vals, 0.005))
                except Exception:
                    thresh = float(np.min(vals))
                lake_mask = gray <= thresh

        combined_mask = mask_resized | lake_mask

        # Remove all connected components that touch the image border so outer
        # DEM rims are never treated as water.
        combined_mask = _remove_border_connected(combined_mask)

        # Optionally drop tiny speckles; this is a no-op when SciPy is not
        # available because _postprocess_water_mask will simply return the mask.
        h_m, w_m = combined_mask.shape
        min_area_px = int(max(1, h_m * w_m * 0.0005))
        combined_mask = _postprocess_water_mask(
            combined_mask,
            keep_components=16,
            min_area_px=min_area_px,
        )

        tinted = _apply_water_tint(rgba, combined_mask)

        # Optional debug: if caller supplied --water-mask-output, emit a mask PNG
        # so it is easy to verify which pixels are classified as water.
        water_mask_out = getattr(args, "water_mask_output", None)
        if water_mask_out is not None:
            _save_water_mask(combined_mask, Path(water_mask_out))

        _save_image(tinted, args.output)
    else:
        frame.save(str(args.output))
    print(f"Wrote {args.output}")

    if getattr(args, "viewer", False):
        viewer_cls = getattr(f3d, "Viewer", None)
        if viewer_cls is None:
            print("forge3d.Viewer is not available in this build. Skipping interactive viewer.")
        else:
            if getattr(args, "sky", None):
                os.environ["FORGE3D_SKY_MODEL"] = str(args.sky)
            with viewer_cls(sess, renderer, heightmap_array, materials, ibl if ibl_enabled else None, params) as view:
                view.run()

    return 0
