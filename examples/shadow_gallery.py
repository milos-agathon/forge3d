#!/usr/bin/env python3
"""
Shadow Gallery Example (P7)

Demonstrates different shadow techniques side-by-side for comparison using the
terrain raster renderer. Falls back to placeholder tiles when GPU/native types
are unavailable.

Usage:
    python examples/shadow_gallery.py
    python examples/shadow_gallery.py --output shadows.png --outdir examples/out
    python examples/shadow_gallery.py --map-res 2048 --tile-size 384 --hdr assets/snow_field_4k.hdr
    python examples/shadow_gallery.py --techniques Hard PCF PCSS VSM
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover
    raise SystemExit("forge3d import failed. Build with `maturin develop --release`.") from exc


# Shadow technique configurations
SHADOW_TECHNIQUES = {
    'Hard': {
        'technique': 'Hard',
        'description': 'Hard shadows - single sample, crisp edges',
    },
    'PCF': {
        'technique': 'PCF',
        'description': 'PCF - Percentage Closer Filtering, soft edges',
    },
    'PCSS': {
        'technique': 'PCSS',
        'description': 'PCSS - Percentage Closer Soft Shadows, variable penumbra',
    },
    'VSM': {
        'technique': 'VSM',
        'description': 'VSM - Variance Shadow Maps, 2-moment filtering',
    },
    'EVSM': {
        'technique': 'EVSM',
        'description': 'EVSM - Exponential Variance, reduced light bleeding',
    },
    'MSM': {
        'technique': 'MSM',
        'description': 'MSM - Moment Shadow Maps, 4-moment accuracy',
    },
    'CSM': {
        'technique': 'CSM',
        'description': 'CSM - Cascaded Shadow Maps, large scene coverage',
    },
}


def _synthetic_dem(width: int, height: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    peak = 400.0 * np.exp(-(xx ** 2 + yy ** 2) / 0.18)
    ridges = 120.0 * np.sin(9.0 * np.arctan2(yy, xx)) * np.exp(-(xx ** 2 + yy ** 2) / 0.5)
    np.random.seed(7)
    noise = 35.0 * np.random.randn(height, width).astype(np.float32)
    heightmap = peak + ridges + noise
    heightmap = np.clip(heightmap, 0.0, 1000.0).astype(np.float32)
    return heightmap


def _label_tile(img: np.ndarray, text: str, sub: str | None = None) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img, mode="RGBA")
        draw = ImageDraw.Draw(im)
        w, h = im.size
        band_h = max(18, h // 14)
        draw.rectangle([0, 0, w, band_h], fill=(0, 0, 0, 110))
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((8, 2), text, fill=(255, 255, 255, 255), font=font)
        if sub:
            draw.text((8, band_h + 2), sub, fill=(220, 220, 220, 255), font=font)
        return np.array(im, dtype=np.uint8)
    except Exception:
        out = np.array(img, copy=True)
        out[:6, :] = [0, 0, 0, 255]
        return out


def _save_image(img: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(img, mode="RGBA").save(str(path))
    except Exception:
        try:
            f3d.numpy_to_png(str(path), img)
        except Exception:
            np.save(str(path).replace(".png", ".npy"), img)
            print("  Warning: Saved as .npy (no PNG writer available)")


def _build_params_for_tech(
    tech: str,
    map_res: int,
    size_px: Tuple[int, int],
    domain: Tuple[float, float] = (0.0, 1000.0),
) -> "f3d.TerrainRenderParams":
    # Simple 3-stop colormap for elevation
    cmap = f3d.Colormap1D.from_stops(
        stops=[(domain[0], "#1e3a5f"), ((domain[0] + domain[1]) * 0.5, "#6ca365"), (domain[1], "#f5f1d0")],
        domain=domain,
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.4)

    # Per-technique ShadowSettings
    base_kwargs = dict(
        enabled=True,
        technique=str(tech).upper(),
        resolution=int(map_res),
        cascades=3 if str(tech).upper() == "CSM" else 1,
        max_distance=1500.0,
        softness=1.4 if str(tech).upper() == "PCSS" else 1.0,
        intensity=0.85,
        slope_scale_bias=0.5,
        depth_bias=0.002,
        normal_bias=0.5,
        min_variance=1e-4,
        light_bleed_reduction=0.5,
        evsm_exponent=40.0,
        fade_start=1.0,
    )
    if str(tech).upper() in {"VSM", "EVSM", "MSM"}:
        base_kwargs["min_variance"] = 1e-4
        base_kwargs["light_bleed_reduction"] = 0.5

    params_cfg = f3d.TerrainRenderParamsConfig(
        size_px=(int(size_px[0]), int(size_px[1])),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=6.0,
        cam_phi_deg=135.0,
        cam_theta_deg=42.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 2000.0),
        light=f3d.LightSettings("Directional", 135.0, 40.0, 3.0, [1.0, 0.97, 0.92]),
        ibl=f3d.IblSettings(True, 1.0, 0.0),
        shadows=f3d.ShadowSettings(**base_kwargs),
        triplanar=f3d.TriplanarSettings(6.0, 4.0, 1.0),
        pom=f3d.PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        lod=f3d.LodSettings(0, 0.0, -0.5),
        sampling=f3d.SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=f3d.ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )
    return f3d.TerrainRenderParams(params_cfg)


def render_shadow_comparison(
    techniques: List[str],
    output_path: Path,
    hdr_path: Path,
    outdir: Path,
    map_res: int = 2048,
    tile_size: int = 384,
    cols: int = 3,
    save_tiles: bool = False,
):
    print(f"Rendering shadow gallery: {len(techniques)} techniques")
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare renderer stack (if GPU/native available)
    use_native = bool(f3d.has_gpu()) and all(
        hasattr(f3d, name)
        for name in ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    )

    heightmap = _synthetic_dem(256, 256)
    domain = (0.0, 1000.0)
    if use_native:
        sess = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(sess)
        materials = f3d.MaterialSet.terrain_default()
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        ibl.set_base_resolution(256)
    else:
        print("  Note: GPU/native path unavailable. Using labeled placeholders.")

    # Grid dimensions
    rows = (len(techniques) + cols - 1) // cols
    full = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)

    for idx, tech_name in enumerate(techniques):
        if tech_name not in SHADOW_TECHNIQUES:
            print(f"  Warning: Unknown technique '{tech_name}', skipping")
            continue
        r = idx // cols
        c = idx % cols
        x0, y0 = c * tile_size, r * tile_size
        label = tech_name
        desc = SHADOW_TECHNIQUES[tech_name]["description"]

        if use_native:
            try:
                params = _build_params_for_tech(tech_name, map_res, (tile_size, tile_size), domain)
                frame = renderer.render_terrain_pbr_pom(
                    material_set=materials,
                    env_maps=ibl,
                    params=params,
                    heightmap=heightmap,
                    target=None,
                )
                rgba = frame.to_numpy()
            except Exception as exc:
                print(f"    Warning: render failed for {tech_name} ({exc}); using placeholder")
                rgba = create_shadow_placeholder(tile_size, label, desc)
        else:
            rgba = create_shadow_placeholder(tile_size, label, desc)

        rgba = _label_tile(rgba, label)
        full[y0 : y0 + tile_size, x0 : x0 + tile_size] = rgba

        if save_tiles:
            _save_image(rgba, outdir / f"shadow_{idx:02d}_{label}.png")

    _save_image(full, output_path)
    print(f"Saved gallery to {output_path}")
    print_technique_comparison(techniques)


def create_shadow_placeholder(size: int, name: str, description: str) -> np.ndarray:
    """
    Create a placeholder tile for shadow visualization

    Args:
        size: Tile size in pixels
        name: Shadow technique name
        description: Technique description

    Returns:
        RGBA image array (size, size, 4)
    """
    # Create a simple visualization based on technique
    tile = np.ones((size, size, 4), dtype=np.uint8) * 200  # Light gray background

    # Add "shadow" visualization (darker region)
    shadow_start = size // 3
    shadow_width = size // 2

    # Different patterns for different techniques
    if name == 'Hard':
        # Sharp edge
        tile[shadow_start:, shadow_start:shadow_start+shadow_width] = [50, 50, 50, 255]
    elif name == 'PCF':
        # Soft edge (gradient)
        for i in range(shadow_width):
            blend = i / shadow_width
            color = int(50 + blend * 150)
            tile[shadow_start:, shadow_start+i] = [color, color, color, 255]
    elif name in ['PCSS', 'VSM', 'EVSM', 'MSM']:
        # Variable penumbra (softer gradient)
        for i in range(shadow_width + 20):
            if shadow_start + i < size:
                blend = min(1.0, i / (shadow_width + 20))
                color = int(50 + blend * 150)
                tile[shadow_start:, shadow_start+i] = [color, color, color, 255]
    else:  # CSM
        # Multiple cascade regions
        cascade_size = shadow_width // 4
        for c in range(4):
            offset = cascade_size * c
            color = int(50 + c * 30)
            tile[shadow_start:, shadow_start+offset:shadow_start+offset+cascade_size] = \
                [color, color, color, 255]

    # Add border and label area
    tile[:30, :] = [100, 100, 150, 255]  # Top label area
    tile[:5, :] = [255, 255, 255, 255]  # Top border
    tile[-5:, :] = [255, 255, 255, 255]  # Bottom border
    tile[:, :5] = [255, 255, 255, 255]  # Left border
    tile[:, -5:] = [255, 255, 255, 255]  # Right border

    return tile


def print_technique_comparison(techniques: List[str]):
    """Print a text comparison table of shadow techniques"""
    print("\n" + "="*70)
    print("SHADOW TECHNIQUE COMPARISON")
    print("="*70)
    print(f"{'Technique':<12} | {'Quality':<10} | {'Performance':<12} | {'Features'}")
    print("-"*70)

    comparisons = {
        'Hard': ('Basic', 'Fastest', 'Crisp edges, aliasing'),
        'PCF': ('Good', 'Fast', 'Soft edges, Poisson sampling'),
        'PCSS': ('Excellent', 'Moderate', 'Variable penumbra, realistic'),
        'VSM': ('Good', 'Fast', '2-moment, light leaking'),
        'EVSM': ('Very Good', 'Moderate', '4-moment, reduced leaking'),
        'MSM': ('Excellent', 'Moderate', '4-moment, high accuracy'),
        'CSM': ('Good', 'Moderate', 'Large scenes, cascades'),
    }

    for tech in techniques:
        if tech in comparisons:
            quality, perf, features = comparisons[tech]
            print(f"{tech:<12} | {quality:<10} | {perf:<12} | {features}")

    print("="*70 + "\n")


def save_image(img: np.ndarray, path: str):  # Backward-compatible helper
    _save_image(img, Path(path))


def main():
    parser = argparse.ArgumentParser(
        description='Compare shadow techniques side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render all techniques
  python examples/shadow_gallery.py

  # Compare specific techniques
  python examples/shadow_gallery.py --techniques Hard PCF PCSS

  # High-resolution shadow maps
  python examples/shadow_gallery.py --map-res 4096 --tile-size 800
        """
    )

    parser.add_argument('--output', '-o', default='shadow_gallery.png', help='Output image path (default: shadow_gallery.png)')
    parser.add_argument('--outdir', default='examples/out', help='Directory for per-tile outputs (optional)')
    parser.add_argument('--hdr', default='assets/snow_field_4k.hdr', help='Environment HDR path for IBL lighting')
    parser.add_argument('--techniques', '-t', nargs='+',
                       default=list(SHADOW_TECHNIQUES.keys()),
                       choices=list(SHADOW_TECHNIQUES.keys()),
                       help='Shadow techniques to compare (default: all)')
    parser.add_argument('--map-res', '-r', type=int, default=2048,
                       help='Shadow map resolution (default: 2048)')
    parser.add_argument('--tile-size', '-s', type=int, default=512,
                       help='Size of each tile in pixels (default: 512)')
    parser.add_argument('--cols', '-c', type=int, default=3,
                       help='Number of columns in gallery (default: 3)')

    args = parser.parse_args()

    render_shadow_comparison(
        techniques=args.techniques,
        output_path=Path(args.output),
        hdr_path=Path(args.hdr),
        outdir=Path(args.outdir),
        map_res=int(args.map_res),
        tile_size=int(args.tile_size),
        cols=int(args.cols),
        save_tiles=False,
    )


if __name__ == '__main__':
    main()
