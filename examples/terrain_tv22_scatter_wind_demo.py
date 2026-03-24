"""TV22 Scatter Wind Animation Demo.

Renders scatter vegetation on Mount Fuji at multiple time steps to
demonstrate GPU wind deformation.  Each frame uses a different
``time_seconds`` value so the vegetation sways over the animation.

Usage::

    python examples/terrain_tv22_scatter_wind_demo.py
    python examples/terrain_tv22_scatter_wind_demo.py --width 1920 --height 1080
"""
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
ts = f3d.terrain_scatter

from forge3d.terrain_params import PomSettings, make_terrain_params_config


DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out"

# Animation time steps (seconds).
TIME_STEPS = [0.0, 0.5, 1.0, 1.5]


# ---------------------------------------------------------------------------
# DEM helpers (same pattern as terrain_tv3_scatter_demo.py)
# ---------------------------------------------------------------------------

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
        dem = f3d.io.load_dem(str(path), fill_nodata_values=True)
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
        dem = f3d.io.load_dem_from_array(data)

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


# ---------------------------------------------------------------------------
# Scatter placement & wind settings
# ---------------------------------------------------------------------------

def _build_scatter_batches(
    heightmap: np.ndarray,
    z_scale: float,
) -> list[ts.TerrainScatterBatch]:
    source = ts.TerrainScatterSource(heightmap, z_scale=z_scale)
    elevation_span = max(source.max_height - source.min_height, 1e-6)

    # Simple tree mesh: cone with y=0 base, height ~2-3 units.
    tree_hi = f3d.geometry.primitive_mesh("cone", radial_segments=12)
    tree_lo = f3d.geometry.primitive_mesh("box")

    terrain_width = float(source.terrain_width)
    near_dist = terrain_width * 0.45
    far_dist = terrain_width * 1.2
    draw_dist = terrain_width * 2.4

    # Vegetation placement using grid_jitter_transforms with slope/elevation
    # filters so trees only appear on moderate slopes at lower altitudes.
    vegetation_transforms = ts.grid_jitter_transforms(
        source,
        spacing=20.0,
        seed=42,
        jitter=0.7,
        scale_range=(10.0, 18.0),
        yaw_range_deg=(0.0, 360.0),
        edge_margin=12.0,
        filters=ts.TerrainScatterFilters(
            max_slope_deg=35.0,
            min_elevation=source.min_height + 0.05 * elevation_span,
            max_elevation=source.min_height + 0.55 * elevation_span,
        ),
    )

    # Wind settings for the vegetation batch.
    wind = ts.ScatterWindSettings(
        enabled=True,
        amplitude=1.5,
        speed=0.8,
        rigidity=0.3,
        gust_strength=0.5,
        gust_frequency=0.3,
        fade_start=500.0,
        fade_end=800.0,
    )

    return [
        ts.TerrainScatterBatch(
            name="wind_trees",
            color=(0.20, 0.42, 0.18, 1.0),
            max_draw_distance=draw_dist,
            transforms=vegetation_transforms,
            levels=[
                ts.TerrainScatterLevel(mesh=tree_hi, max_distance=near_dist),
                ts.TerrainScatterLevel(mesh=tree_lo, max_distance=far_dist),
            ],
            wind=wind,
        ),
    ]


# ---------------------------------------------------------------------------
# Render params
# ---------------------------------------------------------------------------

def _build_params(
    *,
    width: int,
    height: int,
    terrain_span: float,
    domain: tuple[float, float],
    overlay,
    z_scale: float,
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
        colormap_strength=0.28,
        ibl_enabled=True,
        light_azimuth_deg=138.0,
        light_elevation_deg=28.0,
        sun_intensity=2.4,
        cam_radius=max(terrain_span * 1.9, 5.0),
        cam_phi_deg=146.0,
        cam_theta_deg=58.0,
        fov_y_deg=50.0,
        camera_mode="mesh",
        clip=(0.1, clip_far),
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
    )
    return f3d.TerrainRenderParams(config)


# ---------------------------------------------------------------------------
# Frame rendering loop
# ---------------------------------------------------------------------------

def _render_frames(
    *,
    heightmap: np.ndarray,
    terrain_span: float,
    domain: tuple[float, float],
    z_scale: float,
    output_dir: Path,
    width: int,
    height: int,
    batches: list[ts.TerrainScatterBatch],
    time_steps: list[float],
) -> list[dict]:
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

    params = _build_params(
        width=width,
        height=height,
        terrain_span=terrain_span,
        domain=domain,
        overlay=_make_overlay(domain),
        z_scale=z_scale,
    )

    ts.apply_to_renderer(renderer, batches)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for t in time_steps:
        frame = renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=params,
            heightmap=heightmap,
            time_seconds=t,
        )

        tag = f"{t:.1f}".replace(".", "_")
        out_path = output_dir / f"tv22_wind_t{tag}.png"
        frame.save(str(out_path))

        rgba = frame.to_numpy()
        non_black = int(np.count_nonzero(np.any(rgba[..., :3] > 0, axis=-1)))

        info = {
            "time_seconds": t,
            "path": str(out_path),
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            "non_black_pixels": non_black,
        }
        results.append(info)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TV22 demo: scatter wind animation on Mount Fuji DEM."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1440, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Render height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension (0 disables downsampling)",
    )
    args = parser.parse_args()

    dem_path = args.dem.resolve()
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    dem, heightmap = _load_dem(dem_path, int(args.max_dem_size))
    domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
    domain = (float(domain[0]), float(domain[1]))
    terrain_span = _terrain_span(dem, heightmap)
    z_scale = _relief_scale(domain, terrain_span)
    batches = _build_scatter_batches(heightmap, z_scale)

    print(f"DEM source: {dem_path}")
    print(f"Effective DEM size: {heightmap.shape[1]}x{heightmap.shape[0]}")
    print(f"Terrain span: {terrain_span:.2f}")
    print(f"Z scale: {z_scale:.4f}")
    print(f"Wind settings: amplitude=1.5, speed=0.8, rigidity=0.3, "
          f"gust_strength=0.5, gust_frequency=0.3")
    print(f"Rendering {len(TIME_STEPS)} frames at time_seconds = {TIME_STEPS}")
    print()

    results = _render_frames(
        heightmap=heightmap,
        terrain_span=float(terrain_span),
        domain=domain,
        z_scale=float(z_scale),
        output_dir=args.output_dir.resolve(),
        width=int(args.width),
        height=int(args.height),
        batches=batches,
        time_steps=TIME_STEPS,
    )

    for info in results:
        print(
            f"  t={info['time_seconds']:.1f}s  "
            f"{info['width']}x{info['height']}  "
            f"non-black pixels: {info['non_black_pixels']}  "
            f"-> {info['path']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
