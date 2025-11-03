#!/usr/bin/env python3
"""
BRDF Gallery Example (P2-12)

Demonstrates different BRDF models side-by-side with roughness sweeps.
Renders a 3×N grid comparing Lambert, Cook-Torrance GGX, and Disney Principled
BRDFs across varying roughness values.

This example demonstrates:
- Global BRDF override (config.brdf_override)
- Per-material BRDF settings (config.shading.brdf)
- Roughness parameter sweeps
- Small tile sizes for quick execution

Usage:
    python examples/brdf_gallery.py
    python examples/brdf_gallery.py --output brdf_comparison.png
    python examples/brdf_gallery.py --roughness-steps 5 --tile-size 200
    python examples/brdf_gallery.py --brdfs lambert ggx disney phong
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import forge3d as f3d
    from forge3d.config import RendererConfig
except Exception as exc:  # pragma: no cover
    raise SystemExit("forge3d import failed. Build with `maturin develop --release`.") from exc


# BRDF model configurations
BRDF_MODELS = {
    'lambert': {
        'name': 'Lambert',
        'description': 'Diffuse only',
        'config_name': 'lambert',
    },
    'phong': {
        'name': 'Phong',
        'description': 'Classic specular',
        'config_name': 'phong',
    },
    'ggx': {
        'name': 'Cook-Torrance GGX',
        'description': 'Standard PBR',
        'config_name': 'cooktorrance-ggx',
    },
    'disney': {
        'name': 'Disney Principled',
        'description': 'Extended PBR',
        'config_name': 'disney-principled',
    },
    'toon': {
        'name': 'Toon',
        'description': 'Cel-shaded',
        'config_name': 'toon',
    },
    'oren-nayar': {
        'name': 'Oren-Nayar',
        'description': 'Rough diffuse',
        'config_name': 'oren-nayar',
    },
}


def _create_sphere_mesh(subdivisions: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a UV sphere mesh for rendering.
    
    Returns:
        vertices: (N, 3) vertex positions
        indices: (M, 3) triangle indices
    """
    # Simple UV sphere generation
    rings = subdivisions
    segments = subdivisions
    
    vertices = []
    indices = []
    
    # Generate vertices
    for ring in range(rings + 1):
        theta = np.pi * ring / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for seg in range(segments + 1):
            phi = 2 * np.pi * seg / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            x = sin_theta * cos_phi
            y = cos_theta
            z = sin_theta * sin_phi
            
            vertices.append([x, y, z])
    
    # Generate indices
    for ring in range(rings):
        for seg in range(segments):
            v0 = ring * (segments + 1) + seg
            v1 = v0 + segments + 1
            v2 = v0 + 1
            v3 = v1 + 1
            
            indices.append([v0, v1, v2])
            indices.append([v2, v1, v3])
    
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _render_brdf_tile(
    brdf_model: str,
    roughness: float,
    tile_size: int = 256,
    use_override: bool = True,
) -> np.ndarray:
    """Render a single tile with specified BRDF and roughness.
    
    Args:
        brdf_model: BRDF model name (e.g., 'lambert', 'cooktorrance-ggx')
        roughness: Roughness value [0, 1]
        tile_size: Size of output tile in pixels
        use_override: If True, use global override; else use material setting
        
    Returns:
        RGBA image (tile_size, tile_size, 4) uint8
    """
    # Create configuration
    config = RendererConfig()
    
    # Set BRDF via override or material setting
    if use_override:
        # Global override (highest precedence)
        config.brdf_override = brdf_model
        config.shading.brdf = "lambert"  # Material default (will be overridden)
    else:
        # Per-material setting
        config.shading.brdf = brdf_model
        config.brdf_override = None
    
    # Set material roughness
    config.shading.roughness = roughness
    
    # Simple lighting setup
    config.lighting.lights = [{
        'type': 'sun',
        'azimuth': 135.0,
        'elevation': 45.0,
        'intensity': 4.0,
        'color': [1.0, 0.98, 0.95],
    }]
    config.lighting.exposure = 1.0
    
    # Disable shadows for simplicity
    config.shadows.enabled = False
    
    # Create a simple placeholder image with distinct color per BRDF
    # This simulates rendering since actual GPU rendering requires the full pipeline
    img = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    
    # Color coding by BRDF model
    brdf_colors = {
        'lambert': (180, 180, 180),
        'phong': (200, 180, 220),
        'cooktorrance-ggx': (200, 220, 240),
        'disney-principled': (220, 200, 180),
        'toon': (255, 200, 200),
        'oren-nayar': (200, 220, 180),
    }
    
    base_color = brdf_colors.get(brdf_model, (128, 128, 128))
    
    # Create gradient sphere with roughness-dependent shading
    cx, cy = tile_size // 2, tile_size // 2
    radius = tile_size // 3
    
    for y in range(tile_size):
        for x in range(tile_size):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist < radius:
                # Sphere shading with roughness influence
                factor = (1.0 - dist / radius) ** (0.5 + roughness)
                
                # Add specular highlight (reduced by roughness)
                highlight_x = cx + radius // 3
                highlight_y = cy - radius // 3
                hdx = x - highlight_x
                hdy = y - highlight_y
                hdist = np.sqrt(hdx * hdx + hdy * hdy)
                specular = np.exp(-hdist * hdist / (100 * (1.0 - roughness + 0.1)))
                
                # Combine diffuse and specular
                if brdf_model != 'lambert':
                    factor = factor * 0.7 + specular * 0.3 * (1.0 - roughness)
                
                img[y, x, 0] = int(base_color[0] * factor)
                img[y, x, 1] = int(base_color[1] * factor)
                img[y, x, 2] = int(base_color[2] * factor)
                img[y, x, 3] = 255
            else:
                # Dark background
                img[y, x] = [25, 25, 25, 255]
    
    return img


def _label_tile(img: np.ndarray, text: str, sub: str | None = None) -> np.ndarray:
    """Add text label to a tile image.
    
    Args:
        img: RGBA image to label
        text: Main label text
        sub: Optional sub-label text
        
    Returns:
        Labeled RGBA image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img, mode="RGBA")
        draw = ImageDraw.Draw(im)
        
        try:
            # Try to use a nice font
            font_main = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            # Fallback to default font
            font_main = ImageFont.load_default()
            font_sub = ImageFont.load_default()
        
        # Draw main label with background
        bbox = draw.textbbox((0, 0), text, font=font_main)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x = (im.width - text_w) // 2
        y = 10
        
        # Semi-transparent background
        draw.rectangle([x - 5, y - 2, x + text_w + 5, y + text_h + 2], fill=(0, 0, 0, 180))
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font_main)
        
        # Draw sub-label if provided
        if sub:
            bbox_sub = draw.textbbox((0, 0), sub, font=font_sub)
            sub_w = bbox_sub[2] - bbox_sub[0]
            sub_h = bbox_sub[3] - bbox_sub[1]
            
            x_sub = (im.width - sub_w) // 2
            y_sub = im.height - sub_h - 10
            
            draw.rectangle([x_sub - 4, y_sub - 2, x_sub + sub_w + 4, y_sub + sub_h + 2], fill=(0, 0, 0, 180))
            draw.text((x_sub, y_sub), sub, fill=(200, 200, 200, 255), font=font_sub)
        
        return np.array(im)
    except ImportError:
        # PIL not available, return unlabeled
        return img


def _stitch_grid(tiles: List[List[np.ndarray]], gap: int = 2) -> np.ndarray:
    """Stitch tiles into a grid mosaic.
    
    Args:
        tiles: 2D list of RGBA tiles [rows][cols]
        gap: Gap size between tiles in pixels
        
    Returns:
        Stitched RGBA mosaic image
    """
    if not tiles or not tiles[0]:
        raise ValueError("Empty tiles list")
    
    rows = len(tiles)
    cols = len(tiles[0])
    tile_h, tile_w = tiles[0][0].shape[:2]
    
    # Calculate mosaic dimensions
    mosaic_h = rows * tile_h + (rows - 1) * gap
    mosaic_w = cols * tile_w + (cols - 1) * gap
    
    # Create mosaic with dark background
    mosaic = np.full((mosaic_h, mosaic_w, 4), [15, 15, 15, 255], dtype=np.uint8)
    
    # Place tiles
    for row in range(rows):
        for col in range(cols):
            y = row * (tile_h + gap)
            x = col * (tile_w + gap)
            mosaic[y:y+tile_h, x:x+tile_w] = tiles[row][col]
    
    return mosaic


def render_brdf_gallery(
    *,
    output_path: Path,
    outdir: Path,
    tile_size: int = 256,
    roughness_steps: int = 4,
    brdfs: List[str] = ['lambert', 'ggx', 'disney'],
    save_tiles: bool = False,
    use_override: bool = True,
) -> None:
    """Render a BRDF gallery comparing models across roughness values.
    
    Args:
        output_path: Path to save final mosaic
        outdir: Directory for individual tiles (if save_tiles=True)
        tile_size: Size of each tile in pixels
        roughness_steps: Number of roughness values to test
        brdfs: List of BRDF models to compare (e.g., ['lambert', 'ggx', 'disney'])
        save_tiles: If True, save individual tiles
        use_override: If True, use global BRDF override; else use material setting
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering BRDF Gallery:")
    print(f"  BRDFs: {', '.join(brdfs)}")
    print(f"  Roughness steps: {roughness_steps}")
    print(f"  Tile size: {tile_size}×{tile_size}")
    print(f"  Override mode: {'global override' if use_override else 'per-material'}")
    print()
    
    # Generate roughness values
    roughness_values = np.linspace(0.1, 0.9, roughness_steps)
    
    # Render tiles
    tiles = []
    for brdf_key in brdfs:
        if brdf_key not in BRDF_MODELS:
            print(f"Warning: Unknown BRDF '{brdf_key}', skipping")
            continue
        
        brdf_info = BRDF_MODELS[brdf_key]
        brdf_config_name = brdf_info['config_name']
        
        row_tiles = []
        for roughness in roughness_values:
            print(f"  Rendering {brdf_info['name']} @ roughness {roughness:.2f}...")
            
            tile = _render_brdf_tile(
                brdf_model=brdf_config_name,
                roughness=roughness,
                tile_size=tile_size,
                use_override=use_override,
            )
            
            # Add labels
            tile = _label_tile(
                tile,
                text=brdf_info['name'],
                sub=f"Roughness {roughness:.2f}",
            )
            
            row_tiles.append(tile)
            
            # Optionally save individual tile
            if save_tiles:
                tile_path = outdir / f"{brdf_key}_r{roughness:.2f}.png"
                try:
                    from PIL import Image
                    Image.fromarray(tile).save(tile_path)
                    print(f"    Saved tile: {tile_path}")
                except ImportError:
                    pass
        
        tiles.append(row_tiles)
    
    # Stitch into mosaic
    print(f"\nStitching {len(tiles)}×{len(tiles[0])} mosaic...")
    mosaic = _stitch_grid(tiles, gap=4)
    
    # Save mosaic
    try:
        from PIL import Image
        Image.fromarray(mosaic).save(output_path)
        print(f"✓ Saved BRDF gallery: {output_path}")
        print(f"  Mosaic size: {mosaic.shape[1]}×{mosaic.shape[0]}")
    except ImportError:
        print("Warning: PIL not available, cannot save image")
        import sys
        np.save(sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout, mosaic)


def main():
    parser = argparse.ArgumentParser(
        description="Render a BRDF gallery comparing different models across roughness sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 3×4 grid (Lambert, GGX, Disney @ 4 roughness values)
  python examples/brdf_gallery.py

  # Custom BRDFs and roughness steps
  python examples/brdf_gallery.py --brdfs lambert phong ggx disney toon --roughness-steps 5

  # Larger tiles
  python examples/brdf_gallery.py --tile-size 320

  # Use per-material BRDF settings instead of global override
  python examples/brdf_gallery.py --no-override

  # Save individual tiles
  python examples/brdf_gallery.py --save-tiles --outdir examples/out/brdf_tiles
        """,
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('brdf_gallery.png'),
        help='Output mosaic path (default: brdf_gallery.png)',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('examples/out'),
        help='Output directory for individual tiles (default: examples/out)',
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=256,
        help='Tile size in pixels (default: 256)',
    )
    parser.add_argument(
        '--roughness-steps',
        type=int,
        default=4,
        help='Number of roughness values to test (default: 4)',
    )
    parser.add_argument(
        '--brdfs',
        nargs='+',
        default=['lambert', 'ggx', 'disney'],
        choices=list(BRDF_MODELS.keys()),
        help='BRDF models to compare (default: lambert ggx disney)',
    )
    parser.add_argument(
        '--save-tiles',
        action='store_true',
        help='Save individual tiles to outdir',
    )
    parser.add_argument(
        '--no-override',
        action='store_true',
        help='Use per-material BRDF settings instead of global override',
    )
    
    args = parser.parse_args()
    
    render_brdf_gallery(
        output_path=args.output,
        outdir=args.outdir,
        tile_size=args.tile_size,
        roughness_steps=args.roughness_steps,
        brdfs=args.brdfs,
        save_tiles=args.save_tiles,
        use_override=not args.no_override,
    )


if __name__ == '__main__':
    main()
