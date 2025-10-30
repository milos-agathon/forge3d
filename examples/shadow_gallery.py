#!/usr/bin/env python3
"""
Shadow Gallery Example (P7)

Demonstrates different shadow techniques side-by-side for comparison.
Shows the visual differences between Hard, PCF, PCSS, VSM, EVSM, MSM, and CSM shadows.

Usage:
    python examples/shadow_gallery.py
    python examples/shadow_gallery.py --output shadows.png
    python examples/shadow_gallery.py --map-res 2048
    python examples/shadow_gallery.py --techniques Hard PCF PCSS VSM
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List

try:
    import forge3d
    from forge3d.presets import studio_pbr
except ImportError:
    print("Error: forge3d not installed. Run 'maturin develop --release' first.")
    exit(1)


# Shadow technique configurations
SHADOW_TECHNIQUES = {
    'Hard': {
        'technique': 'Hard',
        'map_res': 1024,
        'bias': 0.001,
        'description': 'Hard shadows - single sample, crisp edges',
    },
    'PCF': {
        'technique': 'PCF',
        'map_res': 2048,
        'bias': 0.001,
        'pcf_radius': 2.0,
        'description': 'PCF - Percentage Closer Filtering, soft edges',
    },
    'PCSS': {
        'technique': 'PCSS',
        'map_res': 2048,
        'pcss_blocker_radius': 8.0,
        'pcss_filter_radius': 12.0,
        'light_size': 0.5,
        'moment_bias': 0.0005,
        'description': 'PCSS - Percentage Closer Soft Shadows, variable penumbra',
    },
    'VSM': {
        'technique': 'VSM',
        'map_res': 2048,
        'moment_bias': 0.0001,
        'description': 'VSM - Variance Shadow Maps, 2-moment filtering',
    },
    'EVSM': {
        'technique': 'EVSM',
        'map_res': 2048,
        'moment_bias': 0.0003,
        'description': 'EVSM - Exponential Variance, reduced light bleeding',
    },
    'MSM': {
        'technique': 'MSM',
        'map_res': 2048,
        'moment_bias': 0.0005,
        'description': 'MSM - Moment Shadow Maps, 4-moment accuracy',
    },
    'CSM': {
        'technique': 'CSM',
        'map_res': 2048,
        'cascades': 4,
        'bias': 0.002,
        'description': 'CSM - Cascaded Shadow Maps, large scene coverage',
    },
}


def render_shadow_comparison(
    techniques: List[str],
    output_path: str = "shadow_gallery.png",
    map_res: int = 2048,
    tile_size: int = 512,
    cols: int = 3
):
    """
    Render a gallery comparing different shadow techniques

    Args:
        techniques: List of shadow technique names to compare
        output_path: Output image path
        map_res: Shadow map resolution
        tile_size: Size of each tile in pixels
        cols: Number of columns in gallery grid
    """
    print(f"Rendering shadow gallery: {len(techniques)} techniques")
    print(f"  Shadow map resolution: {map_res}x{map_res}")
    print(f"  Tile size: {tile_size}x{tile_size}")

    # Calculate grid dimensions
    rows = (len(techniques) + cols - 1) // cols  # Ceiling division

    # Create full grid image
    full_width = cols * tile_size
    full_height = rows * tile_size
    gallery = np.zeros((full_height, full_width, 4), dtype=np.uint8)

    # Render each technique
    for idx, tech_name in enumerate(techniques):
        if tech_name not in SHADOW_TECHNIQUES:
            print(f"  Warning: Unknown technique '{tech_name}', skipping")
            continue

        row = idx // cols
        col = idx % cols

        print(f"  [{row},{col}] Rendering: {tech_name}")
        tech_config = SHADOW_TECHNIQUES[tech_name].copy()
        tech_config['map_res'] = map_res  # Override with user-specified resolution

        try:
            # Create renderer with this shadow technique
            config = studio_pbr()
            config['shadows'] = {k: v for k, v in tech_config.items()
                                if k != 'description'}

            # Note: This is a simplified example
            # Actual implementation would:
            # 1. Create renderer with proper scene
            # 2. Render the scene with shadows
            # 3. Extract the rendered image

            # renderer = forge3d.Renderer(tile_size, tile_size, **config)
            # tile_img = renderer.render_scene(...)

            # Placeholder: Create labeled tile
            tile_img = create_shadow_placeholder(
                tile_size,
                tech_name,
                tech_config.get('description', '')
            )

            # Place tile in gallery
            y_start = row * tile_size
            y_end = y_start + tile_size
            x_start = col * tile_size
            x_end = x_start + tile_size
            gallery[y_start:y_end, x_start:x_end] = tile_img

        except Exception as e:
            print(f"    Warning: Failed to render {tech_name}: {e}")

    # Save gallery
    print(f"Saving gallery to {output_path}")
    save_image(gallery, output_path)
    print(f"Done! Shadow gallery saved to {output_path}")
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


def save_image(img: np.ndarray, path: str):
    """Save RGBA image to file"""
    try:
        from PIL import Image
        Image.fromarray(img, mode='RGBA').save(path)
    except ImportError:
        try:
            forge3d.numpy_to_png(img, path)
        except AttributeError:
            np.save(path.replace('.png', '.npy'), img)
            print(f"  Warning: Saved as .npy file (PIL not available)")


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

    parser.add_argument('--output', '-o', default='shadow_gallery.png',
                       help='Output image path (default: shadow_gallery.png)')
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
        output_path=args.output,
        map_res=args.map_res,
        tile_size=args.tile_size,
        cols=args.cols
    )


if __name__ == '__main__':
    main()
