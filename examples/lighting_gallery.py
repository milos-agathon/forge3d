#!/usr/bin/env python3
"""
Lighting Gallery Example (P7)

Demonstrates different BRDF models and lighting configurations in a grid layout.
Each cell shows the same scene with different material/lighting combinations.

Usage:
    python examples/lighting_gallery.py
    python examples/lighting_gallery.py --output gallery.png
    python examples/lighting_gallery.py --rows 3 --cols 4 --tile-size 400
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import forge3d
    from forge3d.presets import studio_pbr, outdoor_sun, minimal
except ImportError:
    print("Error: forge3d not installed. Run 'maturin develop --release' first.")
    exit(1)


def create_test_scene():
    """
    Create a simple test scene with spheres of varying roughness/metallic

    Returns:
        Dictionary with scene geometry (vertices, faces, etc.)
    """
    # For this example, we'll create a simple procedural terrain or sphere
    # In practice, this would load actual geometry
    # This is a placeholder - actual implementation would use terrain or meshes
    return {
        'description': 'Simple test scene with spheres',
        # Actual geometry would go here
    }


def render_brdf_comparison(output_path: str = "lighting_gallery.png",
                           rows: int = 3,
                           cols: int = 4,
                           tile_size: int = 400):
    """
    Render a gallery comparing different BRDF models

    Args:
        output_path: Output image path
        rows: Number of rows in gallery grid
        cols: Number of columns in gallery grid
        tile_size: Size of each tile in pixels
    """
    print(f"Rendering lighting gallery: {rows}x{cols} grid, {tile_size}px tiles")

    # Define BRDF configurations to compare
    brdf_configs = [
        ('Lambert', {'brdf': 'lambert', 'roughness': 1.0, 'metallic': 0.0}),
        ('Phong', {'brdf': 'phong', 'roughness': 0.5, 'metallic': 0.0}),
        ('Oren-Nayar', {'brdf': 'oren-nayar', 'roughness': 0.6, 'metallic': 0.0}),
        ('GGX R=0.2', {'brdf': 'cooktorrance-ggx', 'roughness': 0.2, 'metallic': 0.0}),
        ('GGX R=0.5', {'brdf': 'cooktorrance-ggx', 'roughness': 0.5, 'metallic': 0.0}),
        ('GGX R=0.8', {'brdf': 'cooktorrance-ggx', 'roughness': 0.8, 'metallic': 0.0}),
        ('Disney', {'brdf': 'disney-principled', 'roughness': 0.4, 'metallic': 0.0}),
        ('Disney Metal', {'brdf': 'disney-principled', 'roughness': 0.2, 'metallic': 1.0}),
        ('Ashikhmin', {'brdf': 'ashikhmin-shirley', 'roughness': 0.5, 'metallic': 0.0}),
        ('Ward', {'brdf': 'ward', 'roughness': 0.4, 'metallic': 0.0}),
        ('Toon', {'brdf': 'toon', 'roughness': 0.5, 'metallic': 0.0}),
        ('Minnaert', {'brdf': 'minnaert', 'roughness': 0.6, 'metallic': 0.0}),
    ]

    # Create full grid image
    full_width = cols * tile_size
    full_height = rows * tile_size
    gallery = np.zeros((full_height, full_width, 4), dtype=np.uint8)

    # Render each tile
    tile_idx = 0
    for row in range(rows):
        for col in range(cols):
            if tile_idx >= len(brdf_configs):
                break

            name, shading = brdf_configs[tile_idx]
            print(f"  Rendering tile [{row},{col}]: {name}")

            # Create renderer for this tile with studio preset
            try:
                config = studio_pbr()
                config['shading'].update(shading)

                # Note: This is a simplified example
                # Actual implementation would:
                # 1. Create renderer with proper scene
                # 2. Render the scene
                # 3. Extract the rendered image
                # For now, we'll create a placeholder

                # renderer = forge3d.Renderer(tile_size, tile_size, **config)
                # tile_img = renderer.render_scene(...)

                # Placeholder: Create a gradient with label
                tile_img = create_placeholder_tile(tile_size, name, shading)

                # Place tile in gallery
                y_start = row * tile_size
                y_end = y_start + tile_size
                x_start = col * tile_size
                x_end = x_start + tile_size
                gallery[y_start:y_end, x_start:x_end] = tile_img

            except Exception as e:
                print(f"    Warning: Failed to render {name}: {e}")

            tile_idx += 1

    # Save gallery
    print(f"Saving gallery to {output_path}")
    save_image(gallery, output_path)
    print(f"Done! Gallery saved to {output_path}")


def create_placeholder_tile(size: int, name: str, shading: dict) -> np.ndarray:
    """
    Create a placeholder tile image with label

    Args:
        size: Tile size in pixels
        name: BRDF name
        shading: Shading configuration

    Returns:
        RGBA image array (size, size, 4)
    """
    # Create gradient based on roughness/metallic
    roughness = shading.get('roughness', 0.5)
    metallic = shading.get('metallic', 0.0)

    # Simple gradient visualization
    tile = np.zeros((size, size, 4), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            # Gradient from dark to bright based on roughness
            brightness = int((1.0 - roughness) * 200 + 55)

            if metallic > 0.5:
                # Metallic: cool tones
                tile[y, x] = [brightness // 2, brightness // 2, brightness, 255]
            else:
                # Dielectric: warm tones
                tile[y, x] = [brightness, brightness // 2, brightness // 4, 255]

    # Add label (simplified - would use actual text rendering)
    # For now, just add a border and corner marker
    tile[:5, :] = [255, 255, 255, 255]  # Top border
    tile[-5:, :] = [255, 255, 255, 255]  # Bottom border
    tile[:, :5] = [255, 255, 255, 255]  # Left border
    tile[:, -5:] = [255, 255, 255, 255]  # Right border

    return tile


def save_image(img: np.ndarray, path: str):
    """
    Save RGBA image to file

    Args:
        img: RGBA image array
        path: Output file path
    """
    try:
        # Try using PIL if available
        from PIL import Image
        Image.fromarray(img, mode='RGBA').save(path)
    except ImportError:
        # Fallback: use forge3d's numpy_to_png if available
        try:
            forge3d.numpy_to_png(img, path)
        except AttributeError:
            # Last resort: save as raw NumPy array
            np.save(path.replace('.png', '.npy'), img)
            print(f"  Warning: Saved as .npy file (PIL not available)")


def main():
    parser = argparse.ArgumentParser(description='Render BRDF comparison gallery')
    parser.add_argument('--output', '-o', default='lighting_gallery.png',
                       help='Output image path (default: lighting_gallery.png)')
    parser.add_argument('--rows', '-r', type=int, default=3,
                       help='Number of rows in gallery (default: 3)')
    parser.add_argument('--cols', '-c', type=int, default=4,
                       help='Number of columns in gallery (default: 4)')
    parser.add_argument('--tile-size', '-s', type=int, default=400,
                       help='Size of each tile in pixels (default: 400)')

    args = parser.parse_args()

    render_brdf_comparison(
        output_path=args.output,
        rows=args.rows,
        cols=args.cols,
        tile_size=args.tile_size
    )


if __name__ == '__main__':
    main()
