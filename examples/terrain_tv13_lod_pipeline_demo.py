"""TV13 — Terrain Population LOD Pipeline demo.

Demonstrates the full TV13 LOD pipeline on a real DEM:
  1. QEM mesh simplification (simplify_mesh)
  2. LOD chain generation (generate_lod_chain)
  3. Automatic scatter LOD levels (auto_lod_levels)
  4. Baseline rendering without HLOD
  5. HLOD-enabled rendering with stats comparison
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def _import_forge3d():
    import sys

    try:
        import forge3d as f3d

        # Verify TV13 APIs are available; an older installed forge3d may lack them.
        from forge3d.geometry import simplify_mesh as _check  # noqa: F401
        from forge3d.terrain_scatter import auto_lod_levels as _check2  # noqa: F401

        return f3d
    except (ModuleNotFoundError, ImportError):
        # Purge any stale forge3d entries so the re-import picks up the repo copy.
        for key in [k for k in sys.modules if k == "forge3d" or k.startswith("forge3d.")]:
            del sys.modules[key]

        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


f3d = _import_forge3d()
ts = f3d.terrain_scatter

from forge3d.geometry import primitive_mesh, simplify_mesh, generate_lod_chain
from forge3d.terrain_params import PomSettings, make_terrain_params_config
from forge3d.terrain_scatter import (
    HLODPolicy,
    TerrainScatterBatch,
    TerrainScatterLevel,
    TerrainScatterFilters,
    TerrainScatterSource,
    auto_lod_levels,
)


DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv13_lod_pipeline"


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
# TV13 LOD pipeline demonstrations
# ---------------------------------------------------------------------------

def demo_simplify_mesh():
    """Demonstrate QEM mesh simplification."""
    print("\n=== TV13.1: QEM Mesh Simplification ===")
    sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
    print(f"  Original sphere: {sphere.triangle_count} triangles, {sphere.vertex_count} vertices")

    for ratio in [0.5, 0.25, 0.1]:
        simplified = simplify_mesh(sphere, ratio)
        print(f"  Simplified (ratio={ratio}): {simplified.triangle_count} triangles, "
              f"{simplified.vertex_count} vertices")


def demo_lod_chain():
    """Demonstrate LOD chain generation."""
    print("\n=== TV13.1: LOD Chain Generation ===")
    cone = primitive_mesh("cone", radial_segments=32)
    print(f"  Source cone: {cone.triangle_count} triangles")

    ratios = [1.0, 0.4, 0.15, 0.05]
    chain = generate_lod_chain(cone, ratios)
    for i, mesh in enumerate(chain):
        print(f"  LOD {i} (ratio={ratios[i]}): {mesh.triangle_count} triangles")


def demo_auto_lod_levels():
    """Demonstrate auto_lod_levels for scatter."""
    print("\n=== TV13.2: auto_lod_levels ===")
    cone = primitive_mesh("cone", radial_segments=24)
    levels = auto_lod_levels(cone, lod_count=3, draw_distance=300.0)
    for i, level in enumerate(levels):
        dist_str = f"{level.max_distance:.0f}" if level.max_distance is not None else "inf"
        print(f"  Level {i}: {level.mesh.triangle_count} triangles, max_distance={dist_str}")


def _build_scatter_batches(
    heightmap: np.ndarray, z_scale: float, *, hlod: HLODPolicy | None = None,
) -> list[TerrainScatterBatch]:
    """Build scatter batches with optional HLOD policy."""
    source = TerrainScatterSource(heightmap, z_scale=z_scale)
    elevation_span = max(source.max_height - source.min_height, 1e-6)
    terrain_width = float(source.terrain_width)
    forest_draw = terrain_width * 2.4

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
        filters=TerrainScatterFilters(
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
        filters=TerrainScatterFilters(
            min_slope_deg=18.0,
            max_slope_deg=58.0,
            min_elevation=source.min_height + 0.30 * elevation_span,
            max_elevation=source.min_height + 0.80 * elevation_span,
        ),
    )

    tree_mesh = primitive_mesh("cone", radial_segments=12)
    tree_levels = auto_lod_levels(tree_mesh, lod_count=2, draw_distance=forest_draw)
    rock_mesh = primitive_mesh("cylinder", radial_segments=8)
    rock_levels = auto_lod_levels(rock_mesh, lod_count=2, draw_distance=forest_draw)

    return [
        TerrainScatterBatch(
            name="forest",
            color=(0.20, 0.42, 0.18, 1.0),
            max_draw_distance=forest_draw,
            transforms=forest_transforms,
            levels=tree_levels,
            hlod=hlod,
        ),
        TerrainScatterBatch(
            name="ridge_outcrops",
            color=(0.62, 0.47, 0.27, 1.0),
            max_draw_distance=forest_draw,
            transforms=ridge_transforms,
            levels=rock_levels,
            hlod=hlod,
        ),
    ]


def _render_scene(
    *,
    heightmap: np.ndarray,
    terrain_span: float,
    domain: tuple[float, float],
    z_scale: float,
    output_path: Path,
    width: int,
    height: int,
    batches: list[TerrainScatterBatch],
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


def _print_stats(label: str, stats: dict, memory: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    Batches: {stats['batch_count']}  |  Rendered: {stats['rendered_batches']}")
    print(f"    Total instances: {stats['total_instances']}  |  Visible: {stats['visible_instances']}")
    print(f"    LOD counts: {stats['lod_instance_counts']}")
    print(f"    HLOD cluster draws: {stats.get('hlod_cluster_draws', 0)}")
    print(f"    HLOD covered instances: {stats.get('hlod_covered_instances', 0)}")
    print(f"    Effective draws: {stats.get('effective_draws', 0)}")
    print(f"    Memory — total: {memory['total_buffer_bytes']} bytes")
    print(f"    Memory — HLOD buffers: {memory.get('hlod_buffer_bytes', 0)} bytes")
    print(f"    Memory — HLOD clusters: {memory.get('hlod_cluster_count', 0)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TV13 demo: terrain population LOD pipeline with HLOD on a real DEM."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1440, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Render height in pixels")
    parser.add_argument(
        "--max-dem-size", type=int, default=512,
        help="Clamp the longest DEM dimension (0 disables downsampling)",
    )
    args = parser.parse_args()

    dem_path = args.dem.resolve()
    if not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}")

    print(f"Loading DEM: {dem_path}")
    dem, heightmap = _load_dem(dem_path, int(args.max_dem_size))
    domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
    domain = (float(domain[0]), float(domain[1]))
    terrain_span = _terrain_span(dem, heightmap)
    z_scale = _relief_scale(domain, terrain_span)

    print(f"Effective DEM size: {heightmap.shape[1]}x{heightmap.shape[0]}")
    print(f"Terrain span: {terrain_span:.2f}  |  Z scale: {z_scale:.4f}")
    print(f"Domain: [{domain[0]:.1f}, {domain[1]:.1f}]")

    # --- TV13.1 demos ---
    demo_simplify_mesh()
    demo_lod_chain()
    demo_auto_lod_levels()

    # --- Baseline render (no HLOD) ---
    print("\n=== Rendering: Baseline (no HLOD) ===")
    baseline_path = args.output_dir / "baseline_no_hlod.png"
    baseline_batches = _build_scatter_batches(heightmap, z_scale, hlod=None)
    baseline_stats, baseline_mem = _render_scene(
        heightmap=heightmap,
        terrain_span=terrain_span,
        domain=domain,
        z_scale=z_scale,
        output_path=baseline_path,
        width=args.width,
        height=args.height,
        batches=baseline_batches,
    )
    _print_stats("Baseline", baseline_stats, baseline_mem)
    print(f"  Saved: {baseline_path}")

    # --- HLOD render ---
    print("\n=== Rendering: With HLOD ===")
    hlod_path = args.output_dir / "hlod_enabled.png"
    hlod_policy = HLODPolicy(
        hlod_distance=terrain_span * 0.3,
        cluster_radius=terrain_span * 0.08,
        simplify_ratio=0.1,
    )
    hlod_batches = _build_scatter_batches(heightmap, z_scale, hlod=hlod_policy)
    hlod_stats, hlod_mem = _render_scene(
        heightmap=heightmap,
        terrain_span=terrain_span,
        domain=domain,
        z_scale=z_scale,
        output_path=hlod_path,
        width=args.width,
        height=args.height,
        batches=hlod_batches,
    )
    _print_stats("HLOD", hlod_stats, hlod_mem)
    print(f"  Saved: {hlod_path}")

    # --- Comparison ---
    print("\n=== Stats Comparison ===")
    b_eff = baseline_stats.get("effective_draws", 0)
    h_eff = hlod_stats.get("effective_draws", 0)
    if b_eff > 0:
        reduction = (1.0 - h_eff / b_eff) * 100.0
        print(f"  Effective draws: {b_eff} -> {h_eff}  ({reduction:+.1f}%)")
    else:
        print(f"  Effective draws: {b_eff} -> {h_eff}")
    print(f"  HLOD cluster draws: {hlod_stats.get('hlod_cluster_draws', 0)}")
    print(f"  HLOD covered instances: {hlod_stats.get('hlod_covered_instances', 0)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
