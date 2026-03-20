#!/usr/bin/env python3
"""Render bundled DEM scenes that demonstrate Epic TV1 terrain atmosphere path parity.

This example uses only DEM assets already checked into the repo and renders the
three regression-style looks used to validate TV1:

- clear midday atmosphere
- hazy aerial perspective
- low-sun warm atmosphere

It writes one PNG per variant plus a contact sheet in ``examples/out/``.

Examples:
    python examples/terrain_atmosphere_path_demo.py
    python examples/terrain_atmosphere_path_demo.py --dem fuji --size 1600 900
    python examples/terrain_atmosphere_path_demo.py --variant hazy
"""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except Exception:
    HAS_PIL = False

import forge3d as f3d
from forge3d import io as f3dio
from forge3d.terrain_params import PomSettings, ShadowSettings, SkySettings, make_terrain_params_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples" / "out" / "terrain_atmosphere_path"
AVAILABLE_DEMS = {
    "rainier": PROJECT_ROOT / "assets" / "tif" / "dem_rainier.tif",
    "fuji": PROJECT_ROOT / "assets" / "tif" / "Mount_Fuji_30m.tif",
    "gore": PROJECT_ROOT / "assets" / "tif" / "Gore_Range_Albers_1m.tif",
    "luxembourg": PROJECT_ROOT / "assets" / "tif" / "luxembourg_dem.tif",
    "switzerland": PROJECT_ROOT / "assets" / "tif" / "switzerland_dem.tif",
}


@dataclass(frozen=True)
class AtmospherePreset:
    name: str
    title: str
    light_elevation_deg: float
    light_azimuth_deg: float
    direct_sun_intensity: float
    sky: SkySettings


PRESETS = (
    AtmospherePreset(
        name="clear",
        title="Clear alpine air",
        light_elevation_deg=35.0,
        light_azimuth_deg=135.0,
        direct_sun_intensity=1.75,
        sky=SkySettings(
            enabled=True,
            turbidity=2.1,
            ground_albedo=0.18,
            sun_intensity=1.0,
            sun_size=0.95,
            aerial_density=0.70,
            sky_exposure=0.90,
        ),
    ),
    AtmospherePreset(
        name="hazy",
        title="Warm haze / aerial perspective",
        light_elevation_deg=35.0,
        light_azimuth_deg=135.0,
        direct_sun_intensity=1.05,
        sky=SkySettings(
            enabled=True,
            turbidity=4.6,
            ground_albedo=0.22,
            sun_intensity=1.0,
            sun_size=1.15,
            aerial_density=1.35,
            sky_exposure=0.84,
        ),
    ),
    AtmospherePreset(
        name="low_sun",
        title="Low sun / evening relief",
        light_elevation_deg=12.0,
        light_azimuth_deg=135.0,
        direct_sun_intensity=2.2,
        sky=SkySettings(
            enabled=True,
            turbidity=3.2,
            ground_albedo=0.22,
            sun_intensity=1.8,
            sun_size=1.7,
            aerial_density=1.10,
            sky_exposure=0.92,
        ),
    ),
)


def _require_native() -> None:
    required = ("Session", "TerrainRenderer", "TerrainRenderParams", "MaterialSet", "IBL", "OverlayLayer")
    if not f3d.has_gpu() or not all(hasattr(f3d, name) for name in required):
        raise SystemExit(
            "This example requires a GPU-backed forge3d native build. "
            "Rebuild with `maturin develop --release`."
        )


def _create_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    # The terrain renderer currently expects an IBL handle even when IBL
    # lighting is disabled in params, so this script generates a tiny neutral
    # HDR internally instead of depending on a repo HDR asset.
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for _ in range(width * height):
            handle.write(bytes([128, 128, 128, 128]))


def _downsample_heightmap(heightmap: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return heightmap
    longest = max(int(heightmap.shape[0]), int(heightmap.shape[1]))
    if longest <= max_dim:
        return heightmap
    step = int(math.ceil(longest / max_dim))
    return np.ascontiguousarray(heightmap[::step, ::step])


def _load_dem(path: Path, max_dim: int) -> tuple[object, np.ndarray]:
    try:
        dem = f3dio.load_dem(str(path), fill_nodata_values=True)
    except ImportError:
        data: np.ndarray | None = None
        if HAS_PIL:
            with Image.open(path) as image:
                data = np.asarray(image, dtype=np.float32)
        else:
            try:
                import imageio.v3 as iio

                data = np.asarray(iio.imread(path), dtype=np.float32)
            except Exception as exc:
                raise SystemExit(
                    "DEM loading requires either rasterio or a TIFF-capable Pillow/imageio install."
                ) from exc

        if data.ndim == 3:
            data = data[..., 0]
        dem = f3dio.load_dem_from_array(data)

    heightmap = np.asarray(dem.data, dtype=np.float32).copy()
    return dem, _downsample_heightmap(heightmap, max_dim)


def _terrain_span_m(dem: object) -> tuple[float, str]:
    resolution = getattr(dem, "resolution", (1.0, 1.0)) or (1.0, 1.0)
    try:
        dx_m = float(resolution[0] or 1.0)
        dy_m = float(resolution[1] or 1.0)
    except Exception:
        dx_m, dy_m = 1.0, 1.0

    note = "projected/unspecified"
    crs = str(getattr(dem, "crs", "") or "").lower()
    bounds = getattr(dem, "bounds", None)
    if "4326" in crs or "wgs84" in crs:
        lat_deg = 0.0
        if bounds and len(bounds) == 4:
            try:
                lat_deg = 0.5 * (float(bounds[1]) + float(bounds[3]))
            except Exception:
                lat_deg = 0.0
        lat_rad = math.radians(lat_deg)
        meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2.0 * lat_rad) + 1.175 * math.cos(4.0 * lat_rad)
        meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3.0 * lat_rad)
        dx_m *= meters_per_deg_lon
        dy_m *= meters_per_deg_lat
        note = f"geographic(lat={lat_deg:.3f})"

    h, w = np.asarray(getattr(dem, "data")).shape[:2]
    span_x = max(dx_m, 1e-6) * float(w)
    span_y = max(dy_m, 1e-6) * float(h)
    return float(max(span_x, span_y)), note


def _make_colormap(domain: tuple[float, float]) -> f3d.Colormap1D:
    lo, hi = domain
    span = max(hi - lo, 1e-6)
    stops = (
        (lo + 0.00 * span, "#233d23"),
        (lo + 0.10 * span, "#3f5a31"),
        (lo + 0.26 * span, "#5c7242"),
        (lo + 0.44 * span, "#7f8058"),
        (lo + 0.62 * span, "#927357"),
        (lo + 0.78 * span, "#bca891"),
        (lo + 0.92 * span, "#ddd8cc"),
        (lo + 1.00 * span, "#f6f7fb"),
    )
    return f3d.Colormap1D.from_stops(stops=stops, domain=domain)


def _build_shadow_settings(max_distance: float) -> ShadowSettings:
    return ShadowSettings(
        enabled=True,
        technique="PCSS",
        resolution=2048,
        cascades=3,
        max_distance=max_distance,
        softness=1.5,
        intensity=0.8,
        slope_scale_bias=0.001,
        depth_bias=0.0005,
        normal_bias=0.0002,
        min_variance=1e-4,
        light_bleed_reduction=0.5,
        evsm_exponent=40.0,
        fade_start=1.0,
    )


def _resolve_camera_radius(terrain_span: float, radius_scale: float, cam_radius: float | None) -> float:
    if cam_radius is not None:
        return max(float(cam_radius), 1.0)
    return max(float(terrain_span) * float(radius_scale), 1.0)


def _build_native_params(
    *,
    size_px: tuple[int, int],
    terrain_span: float,
    domain: tuple[float, float],
    overlay: object,
    preset: AtmospherePreset,
    cam_phi_deg: float,
    cam_theta_deg: float,
    cam_fov_deg: float,
    cam_radius: float | None,
    radius_scale: float,
    z_scale: float,
    msaa: int,
) -> object:
    clip_far = max(terrain_span * 3.0, 6000.0)
    config = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=msaa,
        z_scale=z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="mix",
        colormap_strength=0.35,
        ibl_enabled=False,
        light_azimuth_deg=preset.light_azimuth_deg,
        light_elevation_deg=preset.light_elevation_deg,
        sun_intensity=preset.direct_sun_intensity,
        cam_radius=_resolve_camera_radius(terrain_span, radius_scale, cam_radius),
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        fov_y_deg=cam_fov_deg,
        camera_mode="mesh",
        clip=(0.1, clip_far),
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        shadows=_build_shadow_settings(clip_far),
        sky=preset.sky,
    )
    return f3d.TerrainRenderParams(config)


def _render_variant(
    renderer: object,
    material_set: object,
    ibl: object,
    heightmap: np.ndarray,
    params: object,
) -> np.ndarray:
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )
    return frame.to_numpy()


def _orient_output_image(image: np.ndarray) -> np.ndarray:
    # The offline mesh renderer currently lands 180 deg out from the
    # interactive viewer's screen-space orientation for this demo setup.
    return np.ascontiguousarray(np.rot90(image, 2))


def _make_contact_sheet(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    if not images:
        raise ValueError("images must not be empty")
    if len(images) == 1:
        return images[0]

    separator = np.full((images[0].shape[0], 12, 4), 255, dtype=np.uint8)
    separator[..., :3] = 8

    parts: list[np.ndarray] = []
    for idx, image in enumerate(images):
        if idx:
            parts.append(separator)
        parts.append(image)
    sheet = np.concatenate(parts, axis=1)

    if not HAS_PIL:
        return sheet

    pil_image = Image.fromarray(sheet)
    draw = ImageDraw.Draw(pil_image, mode="RGBA")
    panel_width = images[0].shape[1]
    stride = panel_width + separator.shape[1]
    for idx, label in enumerate(labels):
        x = idx * stride + 18
        draw.rounded_rectangle((x - 8, 16, x + 270, 56), radius=10, fill=(0, 0, 0, 180))
        draw.text((x, 24), label, fill=(255, 255, 255, 255))
    return np.asarray(pil_image)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", choices=sorted(AVAILABLE_DEMS), default="rainier", help="Bundled DEM asset to render.")
    parser.add_argument(
        "--variant",
        choices=["all"] + [preset.name for preset in PRESETS],
        default="all",
        help="Render one TV1 variant or the full three-panel comparison.",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-dem-size", type=int, default=1536, help="Downsample DEM if its longest edge exceeds this size.")
    parser.add_argument("--cam-phi", type=float, default=300.0, help="Camera azimuth in degrees.")
    parser.add_argument("--cam-theta", type=float, default=40.0, help="Camera polar angle in degrees.")
    parser.add_argument("--cam-fov", type=float, default=30.0, help="Vertical field of view in degrees.")
    parser.add_argument(
        "--cam-radius",
        type=float,
        default=None,
        help="Absolute camera distance in world units. Defaults to terrain_span * radius_scale.",
    )
    parser.add_argument("--radius-scale", type=float, default=1.10, help="Fallback camera distance as a fraction of terrain span.")
    parser.add_argument("--z-scale", type=float, default=0.10, help="Terrain vertical exaggeration.")
    parser.add_argument("--msaa", type=int, choices=[1, 4, 8], default=1, help="MSAA sample count.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for rendered PNGs.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _require_native()

    dem_path = AVAILABLE_DEMS[args.dem]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dem, heightmap = _load_dem(dem_path, int(args.max_dem_size))
    terrain_span, span_note = _terrain_span_m(dem)
    domain = f3dio.robust_dem_domain(
        heightmap,
        fallback=f3dio.infer_dem_domain(dem, fallback=(0.0, 1.0)),
    )
    colormap = _make_colormap(domain)
    overlay = f3d.OverlayLayer.from_colormap1d(colormap, strength=0.85, domain=domain)

    presets = [preset for preset in PRESETS if args.variant in ("all", preset.name)]

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default(
        triplanar_scale=4.5,
        normal_strength=1.2,
        blend_sharpness=4.5,
    )

    with tempfile.TemporaryDirectory(prefix="forge3d-tv1-") as temp_dir:
        hdr_path = Path(temp_dir) / "neutral.hdr"
        _create_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        ibl.set_base_resolution(64)

        images: list[np.ndarray] = []
        labels: list[str] = []

        print(f"[TV1] DEM: {dem_path.relative_to(PROJECT_ROOT)}")
        print(f"[TV1] Heightmap: {heightmap.shape[1]}x{heightmap.shape[0]} samples, span={terrain_span:.1f}m ({span_note})")
        print(f"[TV1] Domain: min={domain[0]:.2f}, max={domain[1]:.2f}")

        for preset in presets:
            native_params = _build_native_params(
                size_px=(int(args.size[0]), int(args.size[1])),
                terrain_span=terrain_span,
                domain=domain,
                overlay=overlay,
                preset=preset,
                cam_phi_deg=float(args.cam_phi),
                cam_theta_deg=float(args.cam_theta),
                cam_fov_deg=float(args.cam_fov),
                cam_radius=None if args.cam_radius is None else float(args.cam_radius),
                radius_scale=float(args.radius_scale),
                z_scale=float(args.z_scale),
                msaa=int(args.msaa),
            )
            image = _render_variant(renderer, material_set, ibl, heightmap, native_params)
            image = _orient_output_image(image)
            output_path = args.output_dir / f"{args.dem}_tv1_{preset.name}.png"
            f3d.numpy_to_png(output_path, image)
            images.append(image)
            labels.append(preset.title)
            print(
                f"[TV1] Wrote {_display_path(output_path)} "
                f"(turbidity={preset.sky.turbidity}, aerial_density={preset.sky.aerial_density}, "
                f"sun_elev={preset.light_elevation_deg})"
            )

    if len(images) > 1:
        contact_sheet = _make_contact_sheet(images, labels)
        sheet_path = args.output_dir / f"{args.dem}_tv1_contact_sheet.png"
        f3d.numpy_to_png(sheet_path, contact_sheet)
        print(f"[TV1] Wrote {_display_path(sheet_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
