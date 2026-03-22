"""TV12: Offline render quality demo.

Renders three outputs from a real DEM (or synthetic fallback):
  1. Single-sample baseline  (aa_samples=1)
  2. Multi-sample offline    (aa_samples=N, default 16)
  3. Multi-sample + atrous denoised

Saves a side-by-side comparison PNG, an HDR EXR from the multi-sample
resolve, and prints convergence statistics.
"""
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
from forge3d.denoise import atrous_denoise
from forge3d.terrain_params import PomSettings, make_terrain_params_config


DEFAULT_DEM = Path(__file__).resolve().parents[1] / "assets" / "tif" / "dem_rainier.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv12_offline_quality"


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

    if not path.exists():
        print(f"DEM not found: {path} -- falling back to synthetic heightmap")
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
    aa_samples: int = 1,
    aa_seed: int | None = None,
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
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aa_samples=aa_samples,
        aa_seed=aa_seed,
    )
    return f3d.TerrainRenderParams(config)


def _compose_three(left: np.ndarray, center: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 8, 4), 255, dtype=np.uint8)
    divider[..., :3] = 12
    return np.concatenate([left, divider, center, divider, right], axis=1)


def _offline_render(
    renderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    params,
    aa_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the offline accumulation pipeline and return (tonemapped_np, hdr_np, stats)."""
    renderer.begin_offline_accumulation(params, heightmap, material_set, ibl)
    try:
        total = 0
        batch_size = min(4, aa_samples)
        metrics = None
        while total < aa_samples:
            count = min(batch_size, aa_samples - total)
            batch_result = renderer.accumulate_batch(count)
            total = batch_result.total_samples

        if aa_samples > 1:
            metrics = renderer.read_accumulation_metrics(0.001)

        hdr_frame, aov_frame = renderer.resolve_offline_hdr()
        hdr_np = hdr_frame.to_numpy_f32()
        frame = renderer.tonemap_offline_hdr(hdr_frame)
        tonemapped_np = frame.to_numpy()
    finally:
        renderer.end_offline_accumulation()

    stats = {
        "samples": total,
    }
    if metrics is not None:
        stats.update({
            "mean_delta": metrics.mean_delta,
            "p95_delta": metrics.p95_delta,
            "max_tile_delta": metrics.max_tile_delta,
            "converged_tile_ratio": metrics.converged_tile_ratio,
        })

    return tonemapped_np, hdr_np, stats


def render_demo(
    *,
    dem_path: Path | None,
    output_dir: Path,
    width: int,
    height: int,
    samples: int,
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

    # --- 1. Single-sample baseline ---
    params_1 = _build_params(
        width=width, height=height,
        terrain_span=terrain_span, domain=domain, z_scale=z_scale,
        overlay=overlay, aa_samples=1, aa_seed=42,
    )
    baseline_np, _, baseline_stats = _offline_render(
        renderer, material_set, ibl, heightmap, params_1, aa_samples=1,
    )

    # --- 2. Multi-sample offline ---
    params_n = _build_params(
        width=width, height=height,
        terrain_span=terrain_span, domain=domain, z_scale=z_scale,
        overlay=overlay, aa_samples=samples, aa_seed=42,
    )
    multi_np, multi_hdr_np, multi_stats = _offline_render(
        renderer, material_set, ibl, heightmap, params_n, aa_samples=samples,
    )

    # --- 3. Multi-sample + atrous denoised ---
    denoised_hdr = atrous_denoise(
        multi_hdr_np[:, :, :3],
        iterations=3,
        sigma_color=0.1,
        sigma_normal=0.1,
    )
    # Re-upload denoised HDR and tonemap
    denoised_hdr_rgba = np.zeros((height, width, 4), dtype=np.float32)
    denoised_hdr_rgba[:, :, :3] = denoised_hdr
    denoised_hdr_rgba[:, :, 3] = 1.0
    denoised_hdr_frame = renderer.upload_hdr_frame(denoised_hdr_rgba, (width, height))
    denoised_frame = renderer.tonemap_offline_hdr(denoised_hdr_frame)
    denoised_np = denoised_frame.to_numpy()

    # --- Save outputs ---
    baseline_path = output_dir / "tv12_baseline_1spp.png"
    multi_path = output_dir / "tv12_multi_{:d}spp.png".format(samples)
    denoised_path = output_dir / "tv12_denoised_{:d}spp.png".format(samples)
    comparison_path = output_dir / "tv12_comparison.png"
    exr_path = output_dir / "tv12_multi_{:d}spp.exr".format(samples)

    f3d.numpy_to_png(baseline_path, baseline_np)
    f3d.numpy_to_png(multi_path, multi_np)
    f3d.numpy_to_png(denoised_path, denoised_np)

    comparison = _compose_three(baseline_np, multi_np, denoised_np)
    f3d.numpy_to_png(comparison_path, comparison)

    # Save HDR EXR from the multi-sample resolve
    params_exr = _build_params(
        width=width, height=height,
        terrain_span=terrain_span, domain=domain, z_scale=z_scale,
        overlay=overlay, aa_samples=samples, aa_seed=42,
    )
    renderer.begin_offline_accumulation(params_exr, heightmap, material_set, ibl)
    try:
        renderer.accumulate_batch(samples)
        hdr_frame, _aov = renderer.resolve_offline_hdr()
        hdr_frame.save(str(exr_path))
    finally:
        renderer.end_offline_accumulation()

    return {
        "baseline_path": str(baseline_path),
        "multi_path": str(multi_path),
        "denoised_path": str(denoised_path),
        "comparison_path": str(comparison_path),
        "exr_path": str(exr_path),
        "baseline_stats": baseline_stats,
        "multi_stats": multi_stats,
        "terrain_span": terrain_span,
        "domain": domain,
        "z_scale": z_scale,
        "heightmap_shape": tuple(int(v) for v in heightmap.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TV12 offline render quality demo")
    parser.add_argument(
        "--dem", type=Path, default=DEFAULT_DEM,
        help="DEM GeoTIFF path (default: assets/tif/dem_rainier.tif)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument("--width", type=int, default=960, help="Render width")
    parser.add_argument("--height", type=int, default=960, help="Render height")
    parser.add_argument("--samples", type=int, default=16, help="Multi-sample count")
    args = parser.parse_args()

    result = render_demo(
        dem_path=args.dem,
        output_dir=args.output_dir.resolve(),
        width=int(args.width),
        height=int(args.height),
        samples=int(args.samples),
    )

    print(f"Heightmap: {result['heightmap_shape'][1]}x{result['heightmap_shape'][0]}")
    print(f"Terrain span: {result['terrain_span']:.2f}")
    print(f"Domain: {result['domain'][0]:.3f} .. {result['domain'][1]:.3f}")
    print(f"Z scale: {result['z_scale']:.3f}")
    print()
    print("--- Baseline (1 spp) ---")
    print(f"  Samples: {result['baseline_stats']['samples']}")
    print()
    print(f"--- Multi-sample ({args.samples} spp) ---")
    ms = result["multi_stats"]
    print(f"  Samples: {ms['samples']}")
    if "mean_delta" in ms:
        print(f"  Mean delta:            {ms['mean_delta']:.6f}")
        print(f"  P95 delta:             {ms['p95_delta']:.6f}")
        print(f"  Max tile delta:        {ms['max_tile_delta']:.6f}")
        print(f"  Converged tile ratio:  {ms['converged_tile_ratio']:.4f}")
    print()
    print(f"Wrote {result['baseline_path']}")
    print(f"Wrote {result['multi_path']}")
    print(f"Wrote {result['denoised_path']}")
    print(f"Wrote {result['comparison_path']}")
    print(f"Wrote {result['exr_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
