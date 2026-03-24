#!/usr/bin/env python3
"""
TV20 Terrain Material Virtual Texturing Demo

Demonstrates using the VT material streaming system to render large terrains
with material sources generated from elevation data.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def _import_forge3d():
    from _import_shim import ensure_repo_import

    ensure_repo_import()
    import forge3d as f3d

    return f3d


f3d = _import_forge3d()

from forge3d.terrain_params import PomSettings, TerrainVTSettings, VTLayerFamily, make_terrain_params_config

# Use default test DEM if available
DEFAULT_DEM_PATH = Path(__file__).parent.parent / "assets" / "tif" / "dem_rainier.tif"
DEFAULT_HDR_PATH = Path(__file__).parent.parent / "assets" / "hdr" / "env.hdr"
DEFAULT_OUTPUT = Path(__file__).parent / "out" / "tv20_demo.png"


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


def generate_material_albedo(dem: np.ndarray, material_index: int, virtual_size: int) -> bytes:
    """Generate per-material albedo from elevation bands."""
    # Simple height-based coloring
    normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

    # Resample to virtual_size
    scale_y = virtual_size / dem.shape[0]
    scale_x = virtual_size / dem.shape[1]
    resized = np.zeros((virtual_size, virtual_size), dtype=np.float32)
    for y in range(virtual_size):
        for x in range(virtual_size):
            src_y = min(int(y / scale_y), dem.shape[0] - 1)
            src_x = min(int(x / scale_x), dem.shape[1] - 1)
            resized[y, x] = normalized[src_y, src_x]

    # Color by material index + elevation
    color_value = 0.3 + (material_index * 0.1) + (resized * 0.3)
    color_value = np.clip(color_value, 0, 1)

    # Convert to RGBA bytes
    rgba = np.zeros((virtual_size, virtual_size, 4), dtype=np.uint8)
    rgba[:, :, 0] = (color_value * 255).astype(np.uint8)  # R
    rgba[:, :, 1] = (color_value * 0.8 * 255).astype(np.uint8)  # G
    rgba[:, :, 2] = (color_value * 0.6 * 255).astype(np.uint8)  # B
    rgba[:, :, 3] = 255  # A

    return rgba.tobytes()


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
    parser = argparse.ArgumentParser(description="TV20 Virtual Texturing Demo")
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

    # VT source registration would happen here if the feature is available in the build.
    # The VT settings are passed to the renderer through the params, and the renderer
    # will use those settings for its internal streaming system.

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
    )

    # Set VT settings on the config
    if vt_settings:
        config.vt = vt_settings

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

    print(f"Saving output to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(args.output, frame.to_numpy())
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
