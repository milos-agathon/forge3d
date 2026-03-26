#!/usr/bin/env python3
"""
Terrain Material Virtual Texturing Demo

Demonstrates using the VT material streaming system to render large terrains
with material sources generated from elevation data.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np


def _import_forge3d():
    from _import_shim import ensure_repo_import

    ensure_repo_import()
    import forge3d as f3d

    return f3d


f3d = _import_forge3d()

from forge3d.terrain_params import TerrainVTSettings, VTLayerFamily, make_terrain_params_config

# Use default test DEM if available
DEFAULT_DEM_PATH = Path(__file__).parent.parent / "assets" / "tif" / "dem_rainier.tif"
DEFAULT_HDR_PATH = Path(__file__).parent.parent / "assets" / "hdr" / "env.hdr"
DEFAULT_OUTPUT = Path(__file__).parent / "out" / "tv20_demo.png"
DEMO_VT_MATERIAL_COUNT = 4


def load_dem(dem_path: Path) -> np.ndarray:
    """Load elevation data from GeoTIFF."""
    try:
        import rasterio

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            return dem
    except ImportError:
        # Fallback: generate synthetic DEM
        print(f"rasterio not available, generating synthetic DEM")
        return np.random.rand(512, 512).astype(np.float32) * 1000


def generate_material_albedo(dem: np.ndarray, material_index: int, virtual_size: int) -> np.ndarray:
    """Generate a deterministic RGBA8 VT source for one terrain material layer."""
    normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

    y_index = np.clip(
        np.floor(np.linspace(0, dem.shape[0], virtual_size, endpoint=False)).astype(np.int32),
        0,
        dem.shape[0] - 1,
    )
    x_index = np.clip(
        np.floor(np.linspace(0, dem.shape[1], virtual_size, endpoint=False)).astype(np.int32),
        0,
        dem.shape[1] - 1,
    )
    resized = normalized[np.ix_(y_index, x_index)]

    coords = np.linspace(0.0, 1.0, virtual_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    stripe = 0.5 + 0.5 * np.sin(
        (xx * (material_index + 1.5) * 14.0 + yy * (material_index + 2.0) * 11.0) * np.pi
    )
    checker = (
        (
            np.floor(xx * (10 + material_index * 3))
            + np.floor(yy * (12 + material_index * 2))
        )
        % 2.0
    ).astype(np.float32)
    modulation = np.clip(0.25 + 0.45 * resized + 0.30 * (0.55 * checker + 0.45 * stripe), 0.0, 1.0)
    palette = np.array(
        [
            [0.88, 0.18, 0.12],
            [0.14, 0.72, 0.22],
            [0.16, 0.34, 0.90],
            [0.92, 0.84, 0.18],
        ],
        dtype=np.float32,
    )
    base = palette[material_index % len(palette)]
    rgb = np.clip(base * modulation[..., None] + (1.0 - modulation[..., None]) * 0.08, 0.0, 1.0)
    alpha = np.ones((virtual_size, virtual_size, 1), dtype=np.float32)
    return np.ascontiguousarray((np.concatenate([rgb, alpha], axis=-1) * 255.0).round().astype(np.uint8))


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    """Write a minimal preview HDR file."""
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 180, 128]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Terrain material virtual texturing demo")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH, help="DEM file path")
    parser.add_argument("--hdr", type=Path, default=DEFAULT_HDR_PATH, help="HDR environment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1024, help="Render width")
    parser.add_argument("--height", type=int, default=1024, help="Render height")
    parser.add_argument("--atlas-size", type=int, default=4096, help="VT atlas size")
    parser.add_argument("--virtual-size", type=int, default=4096, help="Virtual texture size")
    parser.add_argument("--budget-mb", type=float, default=256.0, help="Residency budget MB")
    parser.add_argument("--no-vt", action="store_true", help="Disable VT (baseline)")
    args = parser.parse_args()

    print(f"Loading DEM from {args.dem}...")
    dem = load_dem(args.dem)

    print(f"Creating session and renderer...")
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    # Load or create HDR
    hdr_path = args.hdr
    if not hdr_path.exists():
        print(f"HDR not found at {args.hdr}, creating temporary preview HDR...")
        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            hdr_path = Path(tmp.name)
        try:
            _write_preview_hdr(hdr_path)
        except Exception:
            pass

    print(f"Loading IBL from {hdr_path}...")
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)

    # Create render parameters
    print("Setting up render parameters...")
    vt_settings = (
        None
        if args.no_vt
        else TerrainVTSettings(
            enabled=True,
            atlas_size=args.atlas_size,
            residency_budget_mb=args.budget_mb,
            layers=[VTLayerFamily(family="albedo", virtual_size_px=(args.virtual_size, args.virtual_size))],
        )
    )

    config = make_terrain_params_config(
        size_px=(args.width, args.height),
        render_scale=1.0,
        terrain_span=max(dem.shape),
        msaa_samples=4,
        z_scale=1.0,
        exposure=1.0,
        domain=(float(dem.min()), float(dem.max())),
        albedo_mode="material",
        colormap_strength=0.0,
        camera_mode="mesh",
    )

    if vt_settings:
        config.vt = vt_settings
        renderer.clear_material_vt_sources()
        print(f"Registering {DEMO_VT_MATERIAL_COUNT} VT material sources...")
        for material_index in range(DEMO_VT_MATERIAL_COUNT):
            source = generate_material_albedo(dem, material_index, args.virtual_size)
            fallback_rgb = source[..., :3].astype(np.float32).mean(axis=(0, 1)) / 255.0
            renderer.register_material_vt_source(
                material_index,
                "albedo",
                source,
                (args.virtual_size, args.virtual_size),
                [float(fallback_rgb[0]), float(fallback_rgb[1]), float(fallback_rgb[2]), 1.0],
            )

    # Wrap in native TerrainRenderParams
    params = f3d.TerrainRenderParams(config)

    print(f"Rendering ({args.width}×{args.height}, VT={'enabled' if vt_settings else 'disabled'})...")
    frame = renderer.render_terrain_pbr_pom(material_set, ibl, params, dem)

    if vt_settings:
        print(f"\nVT Configuration:")
        print(f"  Enabled: {vt_settings.enabled}")
        print(f"  Atlas size: {vt_settings.atlas_size}px")
        print(f"  Residency budget: {vt_settings.residency_budget_mb}MB")
        print(f"  Max mip levels: {vt_settings.max_mip_levels}")
        print(f"  Layers: {len(vt_settings.layers)}")
        for layer in vt_settings.layers:
            print(f"    - {layer.family}: {layer.virtual_size_px}px (tile {layer.tile_size}px)")
        print("  Runtime family support: albedo only (normal/mask are reserved for a later native path)")
        print("\nVT Runtime Stats:")
        stats = renderer.get_material_vt_stats()
        for key in (
            "resident_pages",
            "total_pages",
            "cache_budget_pages",
            "cache_budget_mb",
            "cache_hits",
            "cache_misses",
            "miss_rate",
            "tiles_streamed",
            "evictions",
            "avg_upload_ms",
            "resident_megabytes",
            "source_count",
            "feedback_requests",
        ):
            print(f"  {key}: {stats.get(key, 0.0)}")

    print(f"Saving output to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(args.output, frame.to_numpy())
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
