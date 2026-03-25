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
    DenoiseSettings,
    OfflineQualitySettings,
    PomSettings,
    make_terrain_params_config,
)

DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "dem_rainier.tif"


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 160, 128]))


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
        data: np.ndarray | None = None
        try:
            from PIL import Image

            with Image.open(path) as image:
                data = np.asarray(image, dtype=np.float32)
        except ImportError as exc:
            raise SystemExit(
                "DEM loading requires rasterio or Pillow. "
                "Install with `pip install rasterio` or `pip install pillow`."
            ) from exc

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
    return float(np.clip((terrain_span / relief) * 0.18, 0.1, 10.0))


def _make_overlay(domain: tuple[float, float]):
    lo, hi = map(float, domain)
    span = max(hi - lo, 1e-6)
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (lo + 0.00 * span, "#17351b"),
            (lo + 0.18 * span, "#40672c"),
            (lo + 0.42 * span, "#6f7c41"),
            (lo + 0.64 * span, "#8f7a4f"),
            (lo + 0.82 * span, "#bba792"),
            (lo + 1.00 * span, "#f4f6fb"),
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
    aa_samples: int,
    aa_seed: int,
    denoise_method: str,
):
    clip_far = max(terrain_span * 4.0, 6000.0)
    denoise = DenoiseSettings(
        enabled=denoise_method != "none",
        method=denoise_method,
        iterations=2,
        sigma_color=0.08,
        sigma_normal=0.1,
        sigma_depth=0.1,
    )
    config = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,
        z_scale=z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="mix",
        colormap_strength=0.4,
        ibl_enabled=True,
        light_azimuth_deg=138.0,
        light_elevation_deg=28.0,
        sun_intensity=2.5,
        cam_radius=max(terrain_span * 1.85, 5.0),
        cam_phi_deg=142.0,
        cam_theta_deg=64.0,
        fov_y_deg=52.0,
        camera_mode="mesh",
        clip=(0.1, clip_far),
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aa_samples=aa_samples,
        aa_seed=aa_seed,
        denoise=denoise,
    )
    return f3d.TerrainRenderParams(config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a terrain scene through the TV12 offline accumulation pipeline."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("terrain_tv12_output"),
        help="Directory for offline outputs",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Output height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension before upload (0 disables downsampling)",
    )
    parser.add_argument("--samples", type=int, default=16, help="Offline sample budget")
    parser.add_argument("--seed", type=int, default=11, help="Deterministic jitter seed")
    parser.add_argument(
        "--denoise",
        choices=("none", "atrous", "bilateral", "oidn"),
        default="oidn",
        help="Offline denoiser",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive stopping instead of a fixed sample count",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Samples per offline batch")
    parser.add_argument(
        "--target-variance",
        type=float,
        default=0.002,
        help="Adaptive convergence threshold",
    )
    parser.add_argument(
        "--convergence-ratio",
        type=float,
        default=0.9,
        help="Adaptive converged tile ratio threshold",
    )
    args = parser.parse_args()

    dem_path = args.dem.resolve()
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dem, heightmap = _load_dem(dem_path, int(args.max_dem_size))
    domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
    terrain_span = _terrain_span(dem, heightmap)
    z_scale = _relief_scale(domain, terrain_span)
    overlay = _make_overlay(domain)

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
        width=int(args.width),
        height=int(args.height),
        terrain_span=float(terrain_span),
        domain=(float(domain[0]), float(domain[1])),
        overlay=overlay,
        z_scale=float(z_scale),
        aa_samples=1,
        aa_seed=int(args.seed),
        denoise_method="none",
    )
    offline_params = _build_params(
        width=int(args.width),
        height=int(args.height),
        terrain_span=float(terrain_span),
        domain=(float(domain[0]), float(domain[1])),
        overlay=overlay,
        z_scale=float(z_scale),
        aa_samples=max(int(args.samples), 1),
        aa_seed=int(args.seed),
        denoise_method=str(args.denoise),
    )

    baseline = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        baseline_params,
        heightmap,
        settings=OfflineQualitySettings(enabled=True, adaptive=False, batch_size=1),
    )
    result = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        offline_params,
        heightmap,
        settings=OfflineQualitySettings(
            enabled=True,
            adaptive=bool(args.adaptive),
            target_variance=float(args.target_variance),
            max_samples=max(int(args.samples), 1),
            min_samples=min(4, max(int(args.samples), 1)),
            batch_size=max(int(args.batch_size), 1),
            tile_size=8,
            convergence_ratio=float(args.convergence_ratio),
        ),
    )

    baseline_path = output_dir / "terrain_tv12_baseline.png"
    beauty_path = output_dir / "terrain_tv12_offline.png"
    hdr_exr_path = output_dir / "terrain_tv12_offline_hdr.exr"

    baseline.frame.save(str(baseline_path))
    result.frame.save(str(beauty_path))
    result.aov_frame.save_all(str(output_dir), "terrain_tv12_offline")

    hdr_saved = False
    try:
        result.hdr_frame.save(str(hdr_exr_path))
        hdr_saved = True
    except RuntimeError as exc:
        print(f"HDR EXR save unavailable: {exc}")

    print(f"DEM: {dem_path}")
    print(f"Heightmap: {heightmap.shape[1]}x{heightmap.shape[0]}")
    print(f"Domain: {float(domain[0]):.3f} .. {float(domain[1]):.3f}")
    print(f"Terrain span: {terrain_span:.2f}")
    print(f"Z scale: {z_scale:.4f}")
    print(f"Samples used: {result.metadata['samples_used']}")
    print(f"Denoiser used: {result.metadata['denoiser_used']}")
    print(f"Adaptive: {result.metadata['adaptive']}")
    print(f"Final converged ratio: {result.metadata['converged_ratio']}")
    print(f"Final p95 delta: {result.metadata['final_p95_delta']}")
    print(f"Wrote {baseline_path}")
    print(f"Wrote {beauty_path}")
    print(f"Wrote {output_dir / 'terrain_tv12_offline_albedo.png'}")
    print(f"Wrote {output_dir / 'terrain_tv12_offline_normal.png'}")
    print(f"Wrote {output_dir / 'terrain_tv12_offline_depth.png'}")
    if hdr_saved:
        print(f"Wrote {hdr_exr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
