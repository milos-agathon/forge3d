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
    FogSettings as TerrainFogSettings,
    ReflectionSettings as TerrainReflectionSettings,
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


def _binary_mask_to_shore_distance(mask: np.ndarray) -> np.ndarray:
    """Convert binary water mask to normalized distance-to-shore field.
    
    Uses Euclidean distance transform to compute distance from each water
    pixel to the nearest shore (mask boundary). Result is normalized so
    0.0 = shore and 1.0 = maximum distance (lake center).
    
    Args:
        mask: Binary boolean mask where True = water
        
    Returns:
        Float32 array where 0.0 = not water or at shore, values 0-1 = distance from shore
    """
    from scipy.ndimage import distance_transform_edt
    
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float32)
    
    # Distance transform: each True pixel gets its distance to nearest False pixel
    dist = distance_transform_edt(mask)
    
    # Normalize to 0-1 range
    max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist
    
    # Result: 0.0 outside water, 0.0 at shore (edge), up to 1.0 at lake center
    return dist.astype(np.float32)


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
    pom_enabled: bool | None = None,
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
    shadow_config=None,  # Optional ShadowParams from CLI
    fog_config=None,  # P2: Optional FogSettings from CLI
    reflection_config=None,  # P4: Optional ReflectionSettings from CLI
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
            enabled=shadow_config.enabled if shadow_config else True,
            technique=shadow_config.technique.upper() if shadow_config else "PCSS",
            resolution=shadow_config.map_size if shadow_config else 4096,
            cascades=shadow_config.cascades if shadow_config else 3,
            max_distance=4000.0,  # TODO: add to ShadowParams
            softness=shadow_config.light_size if shadow_config else 1.5,
            intensity=0.8,
            # Shadow bias values - these must be small (0.0001-0.01) for proper depth comparison
            # Larger values cause all fragments to appear lit (no shadows)
            slope_scale_bias=0.001,  # Slope-scaled bias for grazing angles
            depth_bias=shadow_config.moment_bias if shadow_config else 0.0005,  # Base depth bias
            normal_bias=0.0002,  # Peter-panning offset (bias along normal)
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        ),
        overlays=overlays,
        fog=fog_config,  # P2: Pass fog config (None = disabled)
        reflection=reflection_config,  # P4: Pass reflection config (None = disabled)
    )

    if pom_enabled is not None:
        try:
            config.pom.enabled = bool(pom_enabled)
        except AttributeError:
            pass

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


def _save_water_mask(mask, path: Path, mode: str = "overlay") -> None:
    """Save water mask to disk.

    Args:
        mask: Boolean mask array
        path: Output file path
        mode: 'binary' for pure 0/255 grayscale (good for analysis),
              'overlay' for colored RGBA (good for visual inspection)
    """
    h, w = mask.shape
    if mode == "binary":
        # Pure grayscale: 0 = not water, 255 = water
        img = np.zeros((h, w), dtype=np.uint8)
        img[mask] = 255
        _save_image(np.stack([img, img, img, np.full_like(img, 255)], axis=-1), path)
    else:
        # Colored overlay: black background, blue water
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


def _remove_border_connected(
    mask: np.ndarray,
    max_aspect_ratio: float = 8.0,
    max_border_contact: float = 0.25,
) -> np.ndarray:
    """Remove mask components that touch the border AND look like edge artifacts.

    A component is considered an "edge artifact" if it touches the border AND:
    - Has very elongated aspect ratio (thin strips along borders), OR
    - Has high border contact fraction (blocky wedges hugging the edge)

    Real lakes clipped by tile boundary are usually compact with low-to-moderate
    border contact and get preserved.

    Args:
        mask: Boolean mask of water candidates
        max_aspect_ratio: Components with aspect ratio > this that touch border
            are removed. Default 8.0 means a 80x10 strip would be removed.
        max_border_contact: Components with > this fraction of their pixels
            on the border are removed. Default 0.25 catches blocky wedges.

    Complexity is O(H*W) for labeling + O(num_components) for analysis.
    """
    try:
        from scipy import ndimage
    except ImportError:
        # Without scipy, we can't do proper component analysis - skip filtering
        return mask

    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m

    h, w = m.shape
    band = max(1, min(h, w) // 128)  # Border band width

    # Label connected components
    labeled, num_features = ndimage.label(m)
    if num_features == 0:
        return m

    out = m.copy()

    # For each component, check if it touches border AND looks like artifact
    for comp_id in range(1, num_features + 1):
        comp_mask = labeled == comp_id
        rows, cols = np.where(comp_mask)
        if len(rows) == 0:
            continue

        # Check if component touches any border band
        in_top_band = rows < band
        in_bottom_band = rows >= h - band
        in_left_band = cols < band
        in_right_band = cols >= w - band

        touches_top = np.any(in_top_band)
        touches_bottom = np.any(in_bottom_band)
        touches_left = np.any(in_left_band)
        touches_right = np.any(in_right_band)
        touches_border = touches_top or touches_bottom or touches_left or touches_right

        if not touches_border:
            continue  # Keep components that don't touch border

        # Calculate aspect ratio of bounding box
        bbox_h = rows.max() - rows.min() + 1
        bbox_w = cols.max() - cols.min() + 1
        aspect_ratio = max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))

        # Calculate border contact fraction: what % of component pixels are in border band
        area = len(rows)
        border_pixels = np.sum(in_top_band | in_bottom_band | in_left_band | in_right_band)
        border_contact = border_pixels / area if area > 0 else 0

        # Remove if elongated (thin strip) OR high border contact (blocky wedge)
        is_thin_strip = aspect_ratio > max_aspect_ratio
        is_border_hugger = border_contact > max_border_contact

        if is_thin_strip or is_border_hugger:
            out[comp_mask] = False

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
                shadow_config=cfg.shadows,  # Pass shadow config from renderer config
            )

            # Native binding currently expects a non-None IBL handle; turning IBL on/off
            # is controlled via params (ibl_enabled/intensity), not by omitting env_maps.
            frame = renderer_native.render_terrain_pbr_pom(
                material_set=materials,
                env_maps=ibl,
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

        # Optional DEM-only closed-depression gating for automatic lake detection.
        # When provided (via CLI/args), this is forwarded to detect_dem_water_mask
        # as depression_min_depth in elevation units.
        water_depression_min_depth = getattr(args, "water_depression_min_depth", None)
        if water_depression_min_depth is not None:
            water_min_area_pct = 0.001
        else:
            water_min_area_pct = 0.01

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
                    base_min_area_pct=water_min_area_pct,  # Require at least 0.01% of image
                    keep_components=3,  # Keep only top 3 largest water bodies
                    depression_min_depth=water_depression_min_depth,
                )
            except Exception:
                water_mask = _detect_dem_water_mask(
                    heightmap_array,
                    domain,
                    level_normalized=float(args.water_level),
                    slope_threshold=float(args.water_slope),
                    spacing=getattr(dem, "resolution", (1.0, 1.0)),
                    base_min_area_pct=water_min_area_pct,
                    keep_components=3,
                    depression_min_depth=water_depression_min_depth,
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

    # Respect GI configuration: enable IBL only when 'ibl' is present in gi.modes.
    # This mirrors render_sunrise_to_noon_sequence(), where cfg.gi.modes controls
    # whether the native TerrainRenderer receives env_maps or None.
    ibl_enabled = "ibl" in renderer_config.gi.modes

    sun_azimuth_deg = float(args.sun_azimuth) if getattr(args, "sun_azimuth", None) is not None else 135.0
    sun_elevation_deg = float(args.sun_elevation) if getattr(args, "sun_elevation", None) is not None else 35.0
    sun_intensity = float(args.sun_intensity) if getattr(args, "sun_intensity", None) is not None else 3.0
    sun_color = tuple(args.sun_color) if getattr(args, "sun_color", None) is not None else None

    # P2: Parse fog parameters from CLI
    fog_density = float(getattr(args, "fog_density", 0.0) or 0.0)
    fog_height_falloff = float(getattr(args, "fog_height_falloff", 0.0) or 0.0)
    fog_inscatter_str = getattr(args, "fog_inscatter", "1.0,1.0,1.0") or "1.0,1.0,1.0"
    try:
        fog_inscatter = tuple(float(x.strip()) for x in fog_inscatter_str.split(","))[:3]
        if len(fog_inscatter) < 3:
            fog_inscatter = (1.0, 1.0, 1.0)
    except (ValueError, AttributeError):
        fog_inscatter = (1.0, 1.0, 1.0)
    
    # Only create FogSettings if density > 0 (otherwise leave as None for no-op)
    fog_config = None
    if fog_density > 0.0:
        fog_config = TerrainFogSettings(
            density=fog_density,
            height_falloff=fog_height_falloff,
            inscatter=fog_inscatter,
        )

    # P4: Parse reflection parameters from CLI
    water_reflections = bool(getattr(args, "water_reflections", False))
    reflection_intensity = float(getattr(args, "reflection_intensity", 0.8) or 0.8)
    reflection_fresnel_power = float(getattr(args, "reflection_fresnel_power", 5.0) or 5.0)
    reflection_wave_strength = float(getattr(args, "reflection_wave_strength", 0.02) or 0.02)
    reflection_shore_atten = float(getattr(args, "reflection_shore_atten", 0.3) or 0.3)
    reflection_plane_height = float(getattr(args, "reflection_plane_height", 0.0) or 0.0)
    
    # Only create ReflectionSettings if enabled (otherwise leave as None for no-op)
    reflection_config = None
    if water_reflections:
        reflection_config = TerrainReflectionSettings(
            enabled=True,
            intensity=reflection_intensity,
            fresnel_power=reflection_fresnel_power,
            wave_strength=reflection_wave_strength,
            shore_atten_width=reflection_shore_atten,
            water_plane_height=reflection_plane_height,
        )

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
        pom_enabled=not bool(getattr(args, "pom_disabled", False)),
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
        shadow_config=renderer_config.shadows,  # Pass CLI shadow config
        fog_config=fog_config,  # P2: Pass fog config
        reflection_config=reflection_config,  # P4: Pass reflection config
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

    water_material = str(getattr(args, "water_material", "overlay")).lower()
    water_mask_kw = None

    # For PBR mode, we must clean the mask *before* passing to the GPU shader.
    # The overlay path does its own cleaning later (combined with render-time
    # dark pixel detection), but PBR mode needs a clean mask upfront.
    if water_material == "pbr" and water_mask is not None:
        # 1) Remove any component touching the image border (kills edge artifacts)
        cleaned_mask = _remove_border_connected(water_mask)
        # 2) Morphological opening to remove isolated pixels/speckles
        try:
            from scipy import ndimage
            structure = np.ones((3, 3), dtype=bool)
            cleaned_mask = ndimage.binary_opening(cleaned_mask, structure=structure, iterations=1)
        except ImportError:
            pass
        # 3) Postprocess: keep only largest components, stricter min area (0.2%)
        h_m, w_m = cleaned_mask.shape
        if water_depression_min_depth is not None:
            min_area_px = int(max(1, h_m * w_m * 0.0005))
        else:
            min_area_px = int(max(1, h_m * w_m * 0.002))  # 0.2% minimum
        cleaned_mask = _postprocess_water_mask(
            cleaned_mask,
            keep_components=3,
            min_area_px=min_area_px,
        )
        # 4) Convert to shore-distance field (0.0 = shore, 1.0 = lake center)
        # This gives meaningful depth variation for water shading
        water_mask_float = _binary_mask_to_shore_distance(cleaned_mask)
        water_mask_kw = water_mask_float

    # Save debug water mask (the CLEANED version that actually goes to GPU/overlay)
    water_mask_out = getattr(args, "water_mask_output", None)
    mask_out_mode = getattr(args, "water_mask_output_mode", "overlay")
    if water_mask_out is not None and water_mask_kw is not None and water_material == "pbr":
        # For visualization, convert float distance back to binary for overlay mode
        if mask_out_mode == "overlay":
            _save_water_mask(water_mask_kw > 0.0, Path(water_mask_out), mode=mask_out_mode)
        else:
            _save_water_mask(water_mask_kw > 0.0, Path(water_mask_out), mode=mask_out_mode)

    # Debug: report mask statistics before rendering
    if water_mask_kw is not None:
        # Note: os is already imported at module level - do not re-import locally
        _n_water = int(np.sum(water_mask_kw > 0.0))
        _pct = 100.0 * _n_water / water_mask_kw.size
        _has_gradient = np.any((water_mask_kw > 0.01) & (water_mask_kw < 0.99))
        print(f"[WATER DEBUG] Passing mask to GPU: shape={water_mask_kw.shape}, dtype={water_mask_kw.dtype}, water_pixels={_n_water} ({_pct:.2f}%), distance_encoded={_has_gradient}")
        # Extended debug: print water pixel locations (row, col) in numpy array
        water_yx = np.argwhere(water_mask_kw > 0.0)
        if len(water_yx) > 0:
            y_min, y_max = int(water_yx[:, 0].min()), int(water_yx[:, 0].max())
            x_min, x_max = int(water_yx[:, 1].min()), int(water_yx[:, 1].max())
            h, w = water_mask_kw.shape
            print(f"[WATER DEBUG] CPU mask water bounds: row[{y_min}..{y_max}], col[{x_min}..{x_max}] of array {h}x{w}")
            # Position interpretation
            is_upper = y_min < h // 2
            is_right = x_max > w // 2
            print(f"[WATER DEBUG] CPU position: upper-half={is_upper}, right-half={is_right}")
        if os.environ.get("VF_COLOR_DEBUG_MODE") == "4":
            print(f"[WATER DEBUG] VF_COLOR_DEBUG_MODE=4 (water mask visualization enabled)")

    try:
        frame = renderer.render_terrain_pbr_pom(
            material_set=materials,
            env_maps=ibl,
            params=params,
            heightmap=heightmap_array,
            target=None,
            water_mask=water_mask_kw,
        )
    except TypeError as exc:
        msg = str(exc)
        if "water_mask" in msg and "unexpected keyword" in msg:
            frame = renderer.render_terrain_pbr_pom(
                material_set=materials,
                env_maps=ibl,
                params=params,
                heightmap=heightmap_array,
                target=None,
            )
        else:
            raise

    if water_mask is not None and water_material == "overlay":
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

        # FORGE3D_WATER_DEM_ONLY=1 disables the lake_mask union for tests
        # that need consistent DEM-only water detection
        use_dem_only = os.environ.get("FORGE3D_WATER_DEM_ONLY", "0") == "1"
        if use_dem_only:
            combined_mask = mask_resized  # DEM-only water
        else:
            combined_mask = mask_resized | lake_mask  # Union with brightness-based detection

        # Remove all connected components that touch the image border so outer
        # DEM rims are never treated as water.
        combined_mask = _remove_border_connected(combined_mask)

        # Strict filtering to remove false positives (speckles)
        # 1) Morphological opening to remove isolated pixels
        try:
            from scipy import ndimage
            structure = np.ones((3, 3), dtype=bool)
            combined_mask = ndimage.binary_opening(combined_mask, structure=structure, iterations=1)
        except ImportError:
            pass
        
        # 2) Keep only largest components with stricter min area (0.2% of image)
        h_m, w_m = combined_mask.shape
        if water_depression_min_depth is not None:
            min_area_px = int(max(1, h_m * w_m * 0.0005))
        else:
            min_area_px = int(max(1, h_m * w_m * 0.002))  # 0.2% minimum
        combined_mask = _postprocess_water_mask(
            combined_mask,
            keep_components=3,  # Keep only 3 largest water bodies
            min_area_px=min_area_px,
        )

        tinted = _apply_water_tint(rgba, combined_mask)

        # Optional debug: if caller supplied --water-mask-output, emit a mask PNG
        # so it is easy to verify which pixels are classified as water.
        water_mask_out = getattr(args, "water_mask_output", None)
        mask_out_mode = getattr(args, "water_mask_output_mode", "overlay")
        if water_mask_out is not None:
            _save_water_mask(combined_mask, Path(water_mask_out), mode=mask_out_mode)

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
            with viewer_cls(sess, renderer, heightmap_array, materials, ibl, params) as view:
                view.run()

    return 0
