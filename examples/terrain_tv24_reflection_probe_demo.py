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
    PomSettings,
    ProbeSettings,
    ReflectionProbeSettings,
    make_terrain_params_config,
)


DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "dem_rainier.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv24_reflection_probe"


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                b = int((1.0 - x / max(width - 1, 1)) * 220)
                handle.write(bytes([r, g, b, 128]))


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
    fallback = float(max(heightmap.shape))
    resolution = getattr(dem, "resolution", None)
    if resolution is not None:
        try:
            dx = float(resolution[0] or 1.0)
            dy = float(resolution[1] or 1.0)
            if dx > 0.0 and dy > 0.0:
                span = max(dx * heightmap.shape[1], dy * heightmap.shape[0])
                if np.isfinite(span) and span >= 1.0:
                    return float(span)
        except Exception:
            pass
    return fallback


def _relief_scale(domain: tuple[float, float], terrain_span: float) -> float:
    relief = max(float(domain[1]) - float(domain[0]), 1e-6)
    return float(np.clip((terrain_span / relief) * 0.18, 0.12, 10.0))


def _make_overlay(domain: tuple[float, float]):
    lo, hi = map(float, domain)
    span = max(hi - lo, 1e-6)
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (lo + 0.00 * span, "#102d16"),
            (lo + 0.16 * span, "#315a28"),
            (lo + 0.40 * span, "#69723d"),
            (lo + 0.62 * span, "#8d7754"),
            (lo + 0.82 * span, "#c8b7a5"),
            (lo + 1.00 * span, "#f5f7fb"),
        ],
        domain=(lo, hi),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _build_water_mask(heightmap: np.ndarray) -> np.ndarray:
    h, w = heightmap.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    lake_a = np.exp(-(((xx + 0.22) / 0.20) ** 2 + ((yy - 0.08) / 0.10) ** 2) * 2.2)
    lake_b = np.exp(-(((xx - 0.18) / 0.16) ** 2 + ((yy + 0.18) / 0.09) ** 2) * 2.0)
    basin_mask = (heightmap <= np.quantile(heightmap, 0.42)).astype(np.float32)
    mask = np.clip(np.maximum(lake_a, lake_b) * basin_mask * 1.35, 0.0, 1.0).astype(np.float32)
    if float(mask.max()) < 0.1:
        fallback = np.exp(-((xx * xx + yy * yy) / 0.08)).astype(np.float32)
        mask = np.clip(fallback * (heightmap <= np.quantile(heightmap, 0.50)), 0.0, 1.0)
    return np.ascontiguousarray(mask.astype(np.float32))


def _fit_reflection_probe_settings(
    water_mask: np.ndarray,
    terrain_span: float,
    *,
    grid_dims: tuple[int, int] = (4, 4),
    resolution: int = 32,
) -> ReflectionProbeSettings:
    ys, xs = np.where(water_mask > 0.15)
    if xs.size == 0:
        return ReflectionProbeSettings(
            enabled=True,
            grid_dims=grid_dims,
            resolution=resolution,
            ray_count=16,
        )

    h, w = water_mask.shape
    u0 = float(xs.min()) / max(w - 1, 1)
    u1 = float(xs.max()) / max(w - 1, 1)
    v0 = float(ys.min()) / max(h - 1, 1)
    v1 = float(ys.max()) / max(h - 1, 1)
    x0 = (u0 - 0.5) * terrain_span
    x1 = (u1 - 0.5) * terrain_span
    y0 = (v0 - 0.5) * terrain_span
    y1 = (v1 - 0.5) * terrain_span
    span_x = max(x1 - x0, terrain_span * 0.06)
    span_y = max(y1 - y0, terrain_span * 0.06)
    margin_x = max(span_x * 0.20, terrain_span * 0.02)
    margin_y = max(span_y * 0.20, terrain_span * 0.02)
    origin = (x0 - margin_x, y0 - margin_y)
    spacing = (
        (span_x + margin_x * 2.0) / max(grid_dims[0] - 1, 1),
        (span_y + margin_y * 2.0) / max(grid_dims[1] - 1, 1),
    )
    return ReflectionProbeSettings(
        enabled=True,
        grid_dims=grid_dims,
        origin=origin,
        spacing=spacing,
        height_offset=max(terrain_span * 0.005, 5.0),
        resolution=resolution,
        ray_count=16,
        fallback_blend_distance=min(spacing) * 0.85,
    )


def _build_params(
    *,
    width: int,
    height: int,
    terrain_span: float,
    domain: tuple[float, float],
    overlay,
    z_scale: float,
    debug_mode: int,
    reflection_probes: ReflectionProbeSettings | None,
):
    clip_far = max(terrain_span * 4.5, 6000.0)
    config = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=4,
        z_scale=z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="mix",
        colormap_strength=0.34,
        ibl_enabled=True,
        ibl_intensity=2.8,
        light_azimuth_deg=142.0,
        light_elevation_deg=24.0,
        sun_intensity=1.2,
        cam_radius=max(terrain_span * 1.75, 5.0),
        cam_phi_deg=150.0,
        cam_theta_deg=57.0,
        fov_y_deg=48.0,
        camera_mode="mesh",
        clip=(0.1, clip_far),
        debug_mode=debug_mode,
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        probes=ProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=32),
        reflection_probes=reflection_probes,
    )
    return f3d.TerrainRenderParams(config)


def _compose_comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 12, 4), 255, dtype=np.uint8)
    divider[..., :3] = 16
    return np.concatenate([left, divider, right], axis=1)


def render_demo(
    *,
    dem_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    max_dem_size: int = 768,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dem, heightmap = _load_dem(dem_path, max_dem_size)
    domain = tuple(map(float, getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))))
    terrain_span = _terrain_span(dem, heightmap)
    z_scale = _relief_scale(domain, terrain_span)
    overlay = _make_overlay(domain)
    water_mask = _build_water_mask(heightmap)
    reflection_probe_settings = _fit_reflection_probe_settings(water_mask, terrain_span)

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

    diffuse_only = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            overlay=overlay,
            z_scale=z_scale,
            debug_mode=0,
            reflection_probes=ReflectionProbeSettings(enabled=False),
        ),
        heightmap=heightmap,
        water_mask=water_mask,
    )
    reflection_on = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            overlay=overlay,
            z_scale=z_scale,
            debug_mode=0,
            reflection_probes=reflection_probe_settings,
        ),
        heightmap=heightmap,
        water_mask=water_mask,
    )
    reflection_debug = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            overlay=overlay,
            z_scale=z_scale,
            debug_mode=52,
            reflection_probes=reflection_probe_settings,
        ),
        heightmap=heightmap,
        water_mask=water_mask,
    )
    reflection_weight = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            overlay=overlay,
            z_scale=z_scale,
            debug_mode=53,
            reflection_probes=reflection_probe_settings,
        ),
        heightmap=heightmap,
        water_mask=water_mask,
    )
    probe_memory = renderer.get_probe_memory_report()
    reflection_probe_memory = renderer.get_reflection_probe_memory_report()
    water_debug = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            overlay=overlay,
            z_scale=z_scale,
            debug_mode=4,
            reflection_probes=ReflectionProbeSettings(enabled=False),
        ),
        heightmap=heightmap,
        water_mask=water_mask,
    )

    diffuse_path = output_dir / "terrain_tv24_diffuse_only.png"
    reflection_path = output_dir / "terrain_tv24_reflection_on.png"
    reflection_debug_path = output_dir / "terrain_tv24_reflection_debug.png"
    reflection_weight_path = output_dir / "terrain_tv24_reflection_weight.png"
    comparison_path = output_dir / "terrain_tv24_comparison.png"

    diffuse_path.parent.mkdir(parents=True, exist_ok=True)
    diffuse_only.save(str(diffuse_path))
    reflection_on.save(str(reflection_path))
    reflection_debug.save(str(reflection_debug_path))
    reflection_weight.save(str(reflection_weight_path))

    left = diffuse_only.to_numpy()
    right = reflection_on.to_numpy()
    comparison = _compose_comparison(left, right)
    f3d.numpy_to_png(comparison_path, comparison)

    mean_abs_diff = float(np.mean(np.abs(left[..., :3].astype(np.float32) - right[..., :3].astype(np.float32))))
    water_pixels = int(np.count_nonzero(water_mask > 0.15))
    rendered_water = np.all(
        water_debug.to_numpy()[..., :3] == np.array([0, 255, 255], dtype=np.uint8),
        axis=-1,
    )
    rendered_water_pixels = int(np.count_nonzero(rendered_water))
    water_mean_abs_diff = (
        float(np.mean(np.abs(left[..., :3].astype(np.float32)[rendered_water] - right[..., :3].astype(np.float32)[rendered_water])))
        if rendered_water_pixels > 0
        else 0.0
    )

    return {
        "diffuse_path": str(diffuse_path),
        "reflection_path": str(reflection_path),
        "reflection_debug_path": str(reflection_debug_path),
        "reflection_weight_path": str(reflection_weight_path),
        "comparison_path": str(comparison_path),
        "mean_abs_diff": mean_abs_diff,
        "probe_memory": probe_memory,
        "reflection_probe_memory": reflection_probe_memory,
        "terrain_span": terrain_span,
        "domain": domain,
        "z_scale": z_scale,
        "heightmap_shape": tuple(int(v) for v in heightmap.shape),
        "reflection_probe_grid_dims": reflection_probe_settings.grid_dims,
        "reflection_probe_origin": reflection_probe_settings.origin,
        "reflection_probe_spacing": reflection_probe_settings.spacing,
        "water_pixels": water_pixels,
        "rendered_water_pixels": rendered_water_pixels,
        "water_mean_abs_diff": water_mean_abs_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TV24 local reflection probe demo")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="DEM asset to render")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--max-dem-size", type=int, default=768, help="Downsample DEM so its longest side is <= this size")
    args = parser.parse_args()

    result = render_demo(
        dem_path=args.dem.resolve(),
        output_dir=args.output_dir.resolve(),
        width=int(args.width),
        height=int(args.height),
        max_dem_size=int(args.max_dem_size),
    )

    print(f"DEM: {args.dem}")
    print(f"Heightmap: {result['heightmap_shape'][1]}x{result['heightmap_shape'][0]}")
    print(f"Terrain span: {result['terrain_span']:.2f}")
    print(f"Domain: {result['domain'][0]:.2f} .. {result['domain'][1]:.2f}")
    print(f"Z scale: {result['z_scale']:.3f}")
    print(f"Water pixels: {result['water_pixels']}")
    print(f"Diffuse probe memory: {result['probe_memory']}")
    print(f"Reflection probe memory: {result['reflection_probe_memory']}")
    print(f"Mean abs diff: {result['mean_abs_diff']:.4f}")
    print(f"Wrote {result['diffuse_path']}")
    print(f"Wrote {result['reflection_path']}")
    print(f"Wrote {result['reflection_debug_path']}")
    print(f"Wrote {result['reflection_weight_path']}")
    print(f"Wrote {result['comparison_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
