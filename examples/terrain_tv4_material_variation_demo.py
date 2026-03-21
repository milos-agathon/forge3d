from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np


def _import_forge3d():
    try:
        import forge3d as f3d

        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


f3d = _import_forge3d()
f3dio = f3d.io
from forge3d.terrain_params import (
    MaterialLayerSettings,
    MaterialNoiseSettings,
    PomSettings,
    make_terrain_params_config,
)


DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv4_material_variation"


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 180, 128]))


def _downsample_heightmap(heightmap: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return np.ascontiguousarray(heightmap)
    longest = max(int(heightmap.shape[0]), int(heightmap.shape[1]))
    if longest <= max_dim:
        return np.ascontiguousarray(heightmap)
    step = int(math.ceil(longest / max_dim))
    return np.ascontiguousarray(heightmap[::step, ::step])


def _load_dem(path: Path, max_dim: int) -> tuple[object, np.ndarray]:
    try:
        dem = f3dio.load_dem(str(path), fill_nodata_values=True)
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit(
                "DEM loading requires rasterio or Pillow. "
                "Install with `pip install rasterio` or `pip install pillow`."
            ) from exc

        with Image.open(path) as image:
            data = np.asarray(image, dtype=np.float32)
        if data.ndim == 3:
            data = data[..., 0]
        dem = f3dio.load_dem_from_array(data)

    heightmap = np.asarray(dem.data, dtype=np.float32).copy()
    return dem, _downsample_heightmap(heightmap, max_dim)


def _terrain_span(dem: object, heightmap: np.ndarray) -> float:
    resolution = getattr(dem, "resolution", None)
    if resolution is not None:
        try:
            dx = float(resolution[0] or 1.0)
            dy = float(resolution[1] or 1.0)
            if dx > 0.0 and dy > 0.0:
                return max(dx * heightmap.shape[1], dy * heightmap.shape[0])
        except Exception:
            pass
    return float(max(heightmap.shape))


def _relief_scale(domain: tuple[float, float], terrain_span: float) -> float:
    relief = max(float(domain[1]) - float(domain[0]), 1e-6)
    return float(np.clip((terrain_span / relief) * 0.18, 0.12, 10.0))


def _make_overlay(domain: tuple[float, float]):
    lo, hi = map(float, domain)
    span = max(hi - lo, 1e-6)
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (lo + 0.00 * span, "#132d17"),
            (lo + 0.18 * span, "#39602c"),
            (lo + 0.42 * span, "#677440"),
            (lo + 0.64 * span, "#8c7a52"),
            (lo + 0.84 * span, "#c2b39e"),
            (lo + 1.00 * span, "#f4f7fb"),
        ],
        domain=(lo, hi),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _build_params(
    *,
    width: int,
    height: int,
    terrain_span: float,
    domain: tuple[float, float],
    overlay,
    z_scale: float,
    materials: MaterialLayerSettings,
):
    clip_far = max(terrain_span * 6.0, 4000.0)
    config = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=4,
        z_scale=z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="mix",
        colormap_strength=0.35,
        ibl_enabled=True,
        light_azimuth_deg=132.0,
        light_elevation_deg=16.0,
        sun_intensity=2.8,
        # TV4 demo: use a closer framing so the material breakup reads clearly.
        cam_radius=max(terrain_span * 0.30, 5.0),
        cam_phi_deg=176.0,
        cam_theta_deg=42.0,
        fov_y_deg=28.0,
        camera_mode="mesh",
        clip=(0.1, clip_far),
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        materials=materials,
    )
    return f3d.TerrainRenderParams(config)


def _build_materials(domain: tuple[float, float], z_scale: float) -> tuple[MaterialLayerSettings, MaterialLayerSettings]:
    scaled_min = float(domain[0]) * z_scale
    scaled_max = float(domain[1]) * z_scale
    scaled_span = max(scaled_max - scaled_min, 1e-6)
    # Favor broad transition bands so TV4 breakup is visible on a full real-DEM scene.
    snow_altitude_min = scaled_min + scaled_span * 0.30
    snow_altitude_blend = scaled_span * 0.55

    baseline = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=snow_altitude_min,
        snow_altitude_blend=snow_altitude_blend,
        snow_slope_max=52.0,
        snow_slope_blend=22.0,
        rock_enabled=True,
        rock_slope_min=14.0,
        rock_slope_blend=40.0,
        wetness_enabled=True,
        wetness_strength=1.0,
        wetness_slope_influence=1.0,
    )
    varied = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=snow_altitude_min,
        snow_altitude_blend=snow_altitude_blend,
        snow_slope_max=52.0,
        snow_slope_blend=22.0,
        rock_enabled=True,
        rock_slope_min=14.0,
        rock_slope_blend=40.0,
        wetness_enabled=True,
        wetness_strength=1.0,
        wetness_slope_influence=1.0,
        variation=MaterialNoiseSettings(
            macro_scale=2.4,
            detail_scale=10.5,
            octaves=6,
            snow_macro_amplitude=0.70,
            snow_detail_amplitude=0.45,
            rock_macro_amplitude=0.60,
            rock_detail_amplitude=0.45,
            wetness_macro_amplitude=0.70,
            wetness_detail_amplitude=0.45,
        ),
    )
    return baseline, varied


def _compose_comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h or right.shape[0] != h:
        raise ValueError("comparison images must share the same height")
    divider = np.full((h, 12, 4), 255, dtype=np.uint8)
    divider[..., :3] = 12
    return np.concatenate([left, divider, right], axis=1)


def render_demo(
    *,
    dem_path: Path = DEFAULT_DEM,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    width: int = 1440,
    height: int = 900,
    max_dem_size: int = 1024,
) -> dict[str, object]:
    dem_path = dem_path.resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dem, heightmap = _load_dem(dem_path, max_dem_size)
    domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
    domain = (float(domain[0]), float(domain[1]))
    terrain_span = _terrain_span(dem, heightmap)
    z_scale = _relief_scale(domain, terrain_span)
    overlay = _make_overlay(domain)
    baseline_materials, varied_materials = _build_materials(domain, z_scale)

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_preview_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)

    baseline_params = _build_params(
        width=width,
        height=height,
        terrain_span=float(terrain_span),
        domain=domain,
        overlay=overlay,
        z_scale=float(z_scale),
        materials=baseline_materials,
    )
    varied_params = _build_params(
        width=width,
        height=height,
        terrain_span=float(terrain_span),
        domain=domain,
        overlay=overlay,
        z_scale=float(z_scale),
        materials=varied_materials,
    )

    baseline_frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=baseline_params,
        heightmap=heightmap,
    )
    varied_frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=varied_params,
        heightmap=heightmap,
    )

    baseline_rgba = baseline_frame.to_numpy()
    varied_rgba = varied_frame.to_numpy()
    comparison = _compose_comparison(baseline_rgba, varied_rgba)
    mean_abs_diff = float(
        np.mean(np.abs(baseline_rgba[..., :3].astype(np.float32) - varied_rgba[..., :3].astype(np.float32)))
    )

    baseline_path = output_dir / "terrain_tv4_baseline.png"
    varied_path = output_dir / "terrain_tv4_variation.png"
    comparison_path = output_dir / "terrain_tv4_comparison.png"
    baseline_frame.save(str(baseline_path))
    varied_frame.save(str(varied_path))
    f3d.numpy_to_png(comparison_path, comparison)

    return {
        "dem_path": str(dem_path),
        "heightmap_shape": (int(heightmap.shape[0]), int(heightmap.shape[1])),
        "domain": domain,
        "terrain_span": float(terrain_span),
        "z_scale": float(z_scale),
        "baseline_path": str(baseline_path),
        "varied_path": str(varied_path),
        "comparison_path": str(comparison_path),
        "mean_abs_diff": mean_abs_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Epic TV4 demo: terrain material variation on a real repo DEM."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1440, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Render height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension before upload (0 disables downsampling)",
    )
    args = parser.parse_args()

    result = render_demo(
        dem_path=args.dem,
        output_dir=args.output_dir,
        width=int(args.width),
        height=int(args.height),
        max_dem_size=int(args.max_dem_size),
    )

    print(f"DEM: {result['dem_path']}")
    print(f"Heightmap: {result['heightmap_shape'][1]}x{result['heightmap_shape'][0]}")
    print(f"Domain: {result['domain'][0]:.3f} .. {result['domain'][1]:.3f}")
    print(f"Terrain span: {result['terrain_span']:.2f}")
    print(f"Z scale: {result['z_scale']:.4f}")
    print(f"Mean RGB difference: {result['mean_abs_diff']:.3f}")
    print(f"Wrote {result['baseline_path']}")
    print(f"Wrote {result['varied_path']}")
    print(f"Wrote {result['comparison_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
