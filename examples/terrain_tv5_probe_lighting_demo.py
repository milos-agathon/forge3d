from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv5_probe_lighting"


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


def _build_synthetic_heightmap(size: int = 256) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    bowl = 0.58 * (xx * xx + yy * yy)
    ridge = 0.22 * np.exp(-((xx - 0.52) ** 2 * 20.0 + (yy + 0.12) ** 2 * 13.0))
    spur = 0.16 * np.exp(-((xx + 0.40) ** 2 * 28.0 + (yy - 0.30) ** 2 * 20.0))
    slope = 0.10 * xx
    heightmap = bowl + ridge + spur + slope
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _load_heightmap(path: Path | None) -> tuple[np.ndarray, tuple[float, float], float]:
    if path is None:
        heightmap = _build_synthetic_heightmap()
        return heightmap, (0.0, 1.0), 4.0

    dem = f3dio.load_dem(str(path), fill_nodata_values=True)
    heightmap = np.asarray(dem.data, dtype=np.float32).copy()
    domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
    resolution = getattr(dem, "resolution", None)
    terrain_span = float(max(heightmap.shape))
    if resolution is not None:
        try:
            terrain_span = float(
                max(float(resolution[0]) * heightmap.shape[1], float(resolution[1]) * heightmap.shape[0])
            )
        except Exception:
            pass
    return heightmap, (float(domain[0]), float(domain[1])), terrain_span


def _make_overlay(domain: tuple[float, float]):
    lo, hi = domain
    span = max(hi - lo, 1e-6)
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (lo + 0.00 * span, "#153218"),
            (lo + 0.28 * span, "#3d6730"),
            (lo + 0.62 * span, "#8b7a53"),
            (lo + 1.00 * span, "#f3f7fb"),
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
    z_scale: float,
    overlay,
    probes: ProbeSettings | None,
    reflection_probes: ReflectionProbeSettings | None,
    debug_mode: int = 0,
):
    config = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,
        z_scale=z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        ibl_intensity=3.0,
        light_azimuth_deg=136.0,
        light_elevation_deg=16.0,
        sun_intensity=0.8,
        cam_radius=max(terrain_span * 1.5, 6.0),
        cam_phi_deg=138.0,
        cam_theta_deg=58.0,
        fov_y_deg=48.0,
        camera_mode="screen",
        debug_mode=debug_mode,
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        probes=probes,
        reflection_probes=reflection_probes,
    )
    return f3d.TerrainRenderParams(config)


def _compose_comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 12, 4), 255, dtype=np.uint8)
    divider[..., :3] = 12
    return np.concatenate([left, divider, right], axis=1)


def render_demo(
    *,
    dem_path: Path | None,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    heightmap, domain, terrain_span = _load_heightmap(dem_path)
    overlay = _make_overlay(domain)
    z_scale = max(terrain_span / max(domain[1] - domain[0], 1e-6) * 0.18, 0.4)

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

    enabled_probes = ProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=48)
    enabled_reflection_probes = ReflectionProbeSettings(
        enabled=True,
        grid_dims=(4, 4),
        ray_count=16,
    )

    probes_off = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            z_scale=z_scale,
            overlay=overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=False),
        ),
        heightmap=heightmap,
    )
    probes_on = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            z_scale=z_scale,
            overlay=overlay,
            probes=enabled_probes,
            reflection_probes=enabled_reflection_probes,
        ),
        heightmap=heightmap,
    )
    probe_irradiance = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            z_scale=z_scale,
            overlay=overlay,
            probes=enabled_probes,
            reflection_probes=enabled_reflection_probes,
            debug_mode=50,
        ),
        heightmap=heightmap,
    )
    probe_weight = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            z_scale=z_scale,
            overlay=overlay,
            probes=enabled_probes,
            reflection_probes=enabled_reflection_probes,
            debug_mode=51,
        ),
        heightmap=heightmap,
    )
    reflection_probe_color = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(
            width=width,
            height=height,
            terrain_span=terrain_span,
            domain=domain,
            z_scale=z_scale,
            overlay=overlay,
            probes=enabled_probes,
            reflection_probes=enabled_reflection_probes,
            debug_mode=52,
        ),
        heightmap=heightmap,
    )

    off_path = output_dir / "terrain_tv5_probes_off.png"
    on_path = output_dir / "terrain_tv5_probes_on.png"
    irradiance_path = output_dir / "terrain_tv5_probe_irradiance.png"
    weight_path = output_dir / "terrain_tv5_probe_weight.png"
    reflection_color_path = output_dir / "terrain_tv5_reflection_probe_color.png"
    comparison_path = output_dir / "terrain_tv5_probe_comparison.png"

    probes_off.save(str(off_path))
    probes_on.save(str(on_path))
    probe_irradiance.save(str(irradiance_path))
    probe_weight.save(str(weight_path))
    reflection_probe_color.save(str(reflection_color_path))
    comparison = _compose_comparison(probes_off.to_numpy(), probes_on.to_numpy())
    f3d.numpy_to_png(comparison_path, comparison)

    return {
        "off_path": str(off_path),
        "on_path": str(on_path),
        "irradiance_path": str(irradiance_path),
        "weight_path": str(weight_path),
        "reflection_color_path": str(reflection_color_path),
        "comparison_path": str(comparison_path),
        "memory_report": renderer.get_probe_memory_report(),
        "terrain_span": terrain_span,
        "domain": domain,
        "z_scale": z_scale,
        "heightmap_shape": tuple(int(v) for v in heightmap.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TV5 local probe lighting demo")
    parser.add_argument("--dem", type=Path, default=None, help="Optional DEM to render instead of the synthetic bowl scene")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=960, help="Render width")
    parser.add_argument("--height", type=int, default=960, help="Render height")
    args = parser.parse_args()

    result = render_demo(
        dem_path=args.dem,
        output_dir=args.output_dir.resolve(),
        width=int(args.width),
        height=int(args.height),
    )

    print(f"Heightmap: {result['heightmap_shape'][1]}x{result['heightmap_shape'][0]}")
    print(f"Terrain span: {result['terrain_span']:.2f}")
    print(f"Domain: {result['domain'][0]:.3f} .. {result['domain'][1]:.3f}")
    print(f"Z scale: {result['z_scale']:.3f}")
    print(f"Probe memory: {result['memory_report']}")
    print(f"Wrote {result['off_path']}")
    print(f"Wrote {result['on_path']}")
    print(f"Wrote {result['irradiance_path']}")
    print(f"Wrote {result['weight_path']}")
    print(f"Wrote {result['reflection_color_path']}")
    print(f"Wrote {result['comparison_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
