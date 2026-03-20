from __future__ import annotations

import argparse
import math
import os
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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "terrain_tv3_scatter_demo.png"
DEFAULT_VIEWER_OUTPUT = Path(__file__).resolve().parent / "out" / "terrain_tv3_scatter_viewer.png"


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


def _write_heightmap_tiff(path: Path, heightmap: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Writing a temporary viewer DEM requires Pillow (`pip install pillow`).") from exc

    Image.fromarray(np.ascontiguousarray(heightmap, dtype=np.float32)).save(path, format="TIFF")


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


def _build_scatter_batches(heightmap: np.ndarray, z_scale: float) -> list[ts.TerrainScatterBatch]:
    source = ts.TerrainScatterSource(heightmap, z_scale=z_scale)
    elevation_span = max(source.max_height - source.min_height, 1e-6)
    terrain_width = float(source.terrain_width)
    forest_near = terrain_width * 0.35
    forest_far = terrain_width * 1.15
    forest_draw = terrain_width * 2.4
    hero_near = terrain_width * 0.45
    hero_far = terrain_width * 1.8
    hero_draw = terrain_width * 2.6
    normalized = source.normalized_elevation
    low_slope = np.clip(1.0 - (source.slope_degrees / 38.0), 0.0, 1.0)
    mid_band = np.clip(1.0 - np.abs(normalized - 0.28) * 3.0, 0.0, 1.0)
    vegetation_mask = np.ascontiguousarray((low_slope * mid_band).astype(np.float32))

    forest_transforms = ts.mask_density_transforms(
        source,
        vegetation_mask,
        spacing=18.0,
        seed=13,
        jitter=0.85,
        density_scale=0.95,
        scale_range=(12.0, 18.0),
        edge_margin=14.0,
        filters=ts.TerrainScatterFilters(
            max_slope_deg=35.0,
            min_elevation=source.min_height + 0.03 * elevation_span,
            max_elevation=source.min_height + 0.62 * elevation_span,
        ),
    )

    ridge_transforms = ts.grid_jitter_transforms(
        source,
        spacing=42.0,
        seed=21,
        jitter=0.5,
        scale_range=(8.0, 14.0),
        yaw_range_deg=(0.0, 180.0),
        edge_margin=10.0,
        filters=ts.TerrainScatterFilters(
            min_slope_deg=18.0,
            max_slope_deg=58.0,
            min_elevation=source.min_height + 0.30 * elevation_span,
            max_elevation=source.min_height + 0.80 * elevation_span,
        ),
    )

    hero_transforms = ts.seeded_random_transforms(
        source,
        count=96,
        seed=34,
        scale_range=(18.0, 28.0),
        edge_margin=18.0,
        filters=ts.TerrainScatterFilters(
            max_slope_deg=28.0,
            min_elevation=source.min_height + 0.18 * elevation_span,
            max_elevation=source.min_height + 0.52 * elevation_span,
        ),
    )

    tree_hi = f3d.geometry.primitive_mesh("cone", radial_segments=12)
    tree_lo = f3d.geometry.primitive_mesh("box")
    rock_hi = f3d.geometry.primitive_mesh("cylinder", radial_segments=8)
    rock_lo = f3d.geometry.primitive_mesh("box")

    return [
        ts.TerrainScatterBatch(
            name="forest",
            color=(0.20, 0.42, 0.18, 1.0),
            max_draw_distance=forest_draw,
            transforms=forest_transforms,
            levels=[
                ts.TerrainScatterLevel(mesh=tree_hi, max_distance=forest_near),
                ts.TerrainScatterLevel(mesh=tree_lo, max_distance=forest_far),
            ],
        ),
        ts.TerrainScatterBatch(
            name="ridge_outcrops",
            color=(0.62, 0.47, 0.27, 1.0),
            max_draw_distance=hero_draw,
            transforms=ridge_transforms,
            levels=[
                ts.TerrainScatterLevel(mesh=rock_hi, max_distance=forest_near),
                ts.TerrainScatterLevel(mesh=rock_lo, max_distance=hero_far),
            ],
        ),
        ts.TerrainScatterBatch(
            name="hero_trees",
            color=(0.16, 0.34, 0.15, 1.0),
            max_draw_distance=hero_draw,
            transforms=hero_transforms,
            levels=[
                ts.TerrainScatterLevel(mesh=tree_hi, max_distance=hero_near),
                ts.TerrainScatterLevel(mesh=tree_lo, max_distance=hero_far),
            ],
        ),
    ]


def _render_offscreen(
    *,
    heightmap: np.ndarray,
    terrain_span: float,
    domain: tuple[float, float],
    z_scale: float,
    output_path: Path,
    width: int,
    height: int,
    batches: list[ts.TerrainScatterBatch],
) -> tuple[dict, dict]:
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
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(str(output_path))
    return renderer.get_scatter_stats(), renderer.get_scatter_memory_report()


def _render_viewer_snapshot(
    *,
    terrain_path: Path,
    z_scale: float,
    viewer_radius: float,
    width: int,
    height: int,
    batches: list[ts.TerrainScatterBatch],
    snapshot_path: Path,
) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with f3d.open_viewer_async(
        width=width,
        height=height,
        title="Forge3D TV3 Terrain Scatter Demo",
        terrain_path=str(terrain_path),
        timeout=60.0,
    ) as viewer:
        viewer.set_z_scale(z_scale)
        viewer.set_orbit_camera(
            phi_deg=146.0,
            theta_deg=58.0,
            radius=float(viewer_radius),
            fov_deg=50.0,
        )
        viewer.send_ipc(
            {
                "cmd": "set_terrain_sun",
                "azimuth_deg": 138.0,
                "elevation_deg": 28.0,
                "intensity": 2.4,
            }
        )
        ts.apply_to_viewer(viewer, batches)
        viewer.snapshot(str(snapshot_path), width=width, height=height)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Epic TV3 demo: terrain-native scatter placement on a real repo DEM."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Offscreen output PNG")
    parser.add_argument(
        "--viewer-output",
        type=Path,
        default=DEFAULT_VIEWER_OUTPUT,
        help="Optional viewer snapshot output PNG",
    )
    parser.add_argument("--width", type=int, default=1440, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Render height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension before scatter generation and rendering (0 disables downsampling)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Also open the interactive viewer and snapshot the same scatter batches",
    )
    args = parser.parse_args()

    dem_path = args.dem.resolve()
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    dem, heightmap = _load_dem(dem_path, int(args.max_dem_size))
    domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
    domain = (float(domain[0]), float(domain[1]))
    terrain_span = _terrain_span(dem, heightmap)
    scatter_terrain_width = float(max(heightmap.shape[0], heightmap.shape[1]))
    z_scale = _relief_scale(domain, terrain_span)
    viewer_radius = ts.viewer_orbit_radius(scatter_terrain_width)
    batches = _build_scatter_batches(heightmap, z_scale)

    scatter_stats, memory_report = _render_offscreen(
        heightmap=heightmap,
        terrain_span=float(terrain_span),
        domain=domain,
        z_scale=float(z_scale),
        output_path=args.output.resolve(),
        width=int(args.width),
        height=int(args.height),
        batches=batches,
    )

    print(f"DEM source: {dem_path}")
    print(f"Effective DEM size: {heightmap.shape[1]}x{heightmap.shape[0]}")
    print(f"Terrain span: {terrain_span:.2f}")
    print(f"Z scale: {z_scale:.4f}")
    print(f"Scatter batches: {scatter_stats['batch_count']}")
    print(f"Visible instances: {scatter_stats['visible_instances']} / {scatter_stats['total_instances']}")
    print(f"LOD counts: {scatter_stats['lod_instance_counts']}")
    print(f"Scatter memory bytes: {memory_report['total_buffer_bytes']}")
    print(f"Wrote {args.output.resolve()}")

    if args.viewer:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            viewer_dem_path = Path(tmp.name)
        try:
            _write_heightmap_tiff(viewer_dem_path, heightmap)
            _render_viewer_snapshot(
                terrain_path=viewer_dem_path,
                z_scale=float(z_scale),
                viewer_radius=viewer_radius,
                width=int(args.width),
                height=int(args.height),
                batches=batches,
                snapshot_path=args.viewer_output.resolve(),
            )
        finally:
            try:
                os.unlink(viewer_dem_path)
            except OSError:
                pass
        print(f"Wrote {args.viewer_output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
