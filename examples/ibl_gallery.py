#!/usr/bin/env python3
"""
IBL Gallery Example (P7)

Demonstrates Image-Based Lighting with:
- HDR environment map rotation sweep
- Material roughness sweep showing reflection quality
- Metallic vs. dielectric materials

Usage:
    python examples/ibl_gallery.py --hdr assets/env.hdr
    python examples/ibl_gallery.py --hdr assets/env.hdr --output ibl_gallery.png
    python examples/ibl_gallery.py --hdr assets/env.hdr --mode rotation
    python examples/ibl_gallery.py --hdr assets/env.hdr --mode roughness
    python examples/ibl_gallery.py --hdr assets/env.hdr --mode metallic
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import forge3d
    from forge3d.presets import studio_pbr
except ImportError:
    print("Error: forge3d not installed. Run 'maturin develop --release' first.")
    exit(1)


def render_rotation_sweep(
    hdr_path: str,
    output_path: str = "ibl_rotation.png",
    tile_size: int = 400,
    rotation_steps: int = 8
):
    """
    Render IBL with environment rotation sweep

    Shows how the scene appearance changes as the HDR environment
    is rotated around the vertical axis.

    Args:
        hdr_path: Path to HDR environment map
        output_path: Output image path
        tile_size: Size of each tile in pixels
        rotation_steps: Number of rotation angles to show
    """
    print(f"Rendering IBL rotation sweep: {rotation_steps} angles")
    print(f"  HDR: {hdr_path}")

    cols = min(rotation_steps, 4)
    rows = (rotation_steps + cols - 1) // cols

    full_width = cols * tile_size
    full_height = rows * tile_size
    gallery = np.zeros((full_height, full_width, 4), dtype=np.uint8)

    for i in range(rotation_steps):
        rotation_deg = (i * 360.0) / rotation_steps
        row = i // cols
        col = i % cols

        print(f"  [{row},{col}] Rotation: {rotation_deg:.1f}°")

        try:
            config = studio_pbr(ibl_intensity=1.5)
            config['gi']['ibl_rotation'] = rotation_deg
            config['gi']['ibl_path'] = hdr_path

            # Placeholder tile
            tile_img = create_ibl_placeholder(
                tile_size,
                f"{rotation_deg:.0f}°",
                rotation_deg
            )

            y_start = row * tile_size
            y_end = y_start + tile_size
            x_start = col * tile_size
            x_end = x_start + tile_size
            gallery[y_start:y_end, x_start:x_end] = tile_img

        except Exception as e:
            print(f"    Warning: Failed at rotation {rotation_deg}: {e}")

    print(f"Saving to {output_path}")
    save_image(gallery, output_path)
    print("Done!")


def render_roughness_sweep(
    hdr_path: str,
    output_path: str = "ibl_roughness.png",
    tile_size: int = 400,
    roughness_steps: int = 10
):
    """
    Render material roughness sweep with IBL

    Shows how surface roughness affects IBL reflections,
    from mirror-like (roughness=0) to diffuse (roughness=1).

    Args:
        hdr_path: Path to HDR environment map
        output_path: Output image path
        tile_size: Size of each tile in pixels
        roughness_steps: Number of roughness values to show
    """
    print(f"Rendering IBL roughness sweep: {roughness_steps} steps")
    print(f"  HDR: {hdr_path}")

    cols = min(roughness_steps, 5)
    rows = (roughness_steps + cols - 1) // cols

    full_width = cols * tile_size
    full_height = rows * tile_size
    gallery = np.zeros((full_height, full_width, 4), dtype=np.uint8)

    for i in range(roughness_steps):
        roughness = i / max(1, roughness_steps - 1)  # 0.0 to 1.0
        row = i // cols
        col = i % cols

        print(f"  [{row},{col}] Roughness: {roughness:.2f}")

        try:
            config = studio_pbr(ibl_intensity=1.5, roughness=roughness)
            config['gi']['ibl_path'] = hdr_path
            config['shading']['metallic'] = 0.0  # Dielectric

            # Placeholder tile
            tile_img = create_roughness_placeholder(
                tile_size,
                f"R={roughness:.2f}",
                roughness
            )

            y_start = row * tile_size
            y_end = y_start + tile_size
            x_start = col * tile_size
            x_end = x_start + tile_size
            gallery[y_start:y_end, x_start:x_end] = tile_img

        except Exception as e:
            print(f"    Warning: Failed at roughness {roughness}: {e}")

    print(f"Saving to {output_path}")
    save_image(gallery, output_path)
    print("Done!")


def render_metallic_comparison(
    hdr_path: str,
    output_path: str = "ibl_metallic.png",
    tile_size: int = 400
):
    """
    Compare metallic vs. dielectric materials with IBL

    Shows the difference between metals (strong colored reflections)
    and dielectrics (weaker white reflections) at various roughness values.

    Args:
        hdr_path: Path to HDR environment map
        output_path: Output image path
        tile_size: Size of each tile in pixels
    """
    print(f"Rendering IBL metallic comparison")
    print(f"  HDR: {hdr_path}")

    # 2 rows (metallic vs. dielectric) × 5 cols (roughness values)
    rows = 2
    cols = 5
    roughness_values = [0.0, 0.2, 0.4, 0.6, 0.8]

    full_width = cols * tile_size
    full_height = rows * tile_size
    gallery = np.zeros((full_height, full_width, 4), dtype=np.uint8)

    for metallic_idx, metallic in enumerate([0.0, 1.0]):
        material_type = "Dielectric" if metallic < 0.5 else "Metallic"

        for roughness_idx, roughness in enumerate(roughness_values):
            row = metallic_idx
            col = roughness_idx

            label = f"{material_type} R={roughness:.1f}"
            print(f"  [{row},{col}] {label}")

            try:
                config = studio_pbr(
                    ibl_intensity=1.5,
                    roughness=roughness,
                    metallic=metallic
                )
                config['gi']['ibl_path'] = hdr_path

                # Placeholder tile
                tile_img = create_metallic_placeholder(
                    tile_size,
                    label,
                    roughness,
                    metallic
                )

                y_start = row * tile_size
                y_end = y_start + tile_size
                x_start = col * tile_size
                x_end = x_start + tile_size
                gallery[y_start:y_end, x_start:x_end] = tile_img

            except Exception as e:
                print(f"    Warning: Failed at {label}: {e}")

    print(f"Saving to {output_path}")
    save_image(gallery, output_path)
    print("Done!")


def create_ibl_placeholder(size: int, label: str, rotation: float) -> np.ndarray:
    """Create placeholder showing rotation effect"""
    tile = np.zeros((size, size, 4), dtype=np.uint8)

    # Create a radial gradient that rotates
    center = size // 2
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            angle = np.arctan2(dy, dx) + np.radians(rotation)
            radius = np.sqrt(dx*dx + dy*dy) / (size / 2)

            # Colored gradient based on angle
            hue = (angle + np.pi) / (2 * np.pi)
            brightness = int((1.0 - radius) * 200 + 55)

            if hue < 0.33:
                tile[y, x] = [brightness, brightness//2, brightness//4, 255]
            elif hue < 0.67:
                tile[y, x] = [brightness//2, brightness, brightness//4, 255]
            else:
                tile[y, x] = [brightness//4, brightness//2, brightness, 255]

    # Add label
    add_label_border(tile, label)
    return tile


def create_roughness_placeholder(size: int, label: str, roughness: float) -> np.ndarray:
    """Create placeholder showing roughness effect on reflections"""
    tile = np.ones((size, size, 4), dtype=np.uint8) * 128  # Mid gray

    # Simulate blurred reflection based on roughness
    center = size // 2
    blur_radius = int(roughness * 50) + 1

    for y in range(size):
        for x in range(size):
            dx = abs(x - center)
            dy = abs(y - center)
            dist = np.sqrt(dx*dx + dy*dy)

            # Sharp reflection at center, blurred based on roughness
            if dist < blur_radius:
                # Bright reflection
                brightness = int(255 - (dist / blur_radius) * 100)
                tile[y, x] = [brightness, brightness, brightness, 255]
            else:
                # Diffuse base color
                base = int(128 + roughness * 50)
                tile[y, x] = [base, base, base, 255]

    add_label_border(tile, label)
    return tile


def create_metallic_placeholder(size: int, label: str,
                                roughness: float, metallic: float) -> np.ndarray:
    """Create placeholder showing metallic vs. dielectric differences"""
    tile = np.ones((size, size, 4), dtype=np.uint8) * 100

    center = size // 2
    blur_radius = int(roughness * 50) + 1

    for y in range(size):
        for x in range(size):
            dx = abs(x - center)
            dy = abs(y - center)
            dist = np.sqrt(dx*dx + dy*dy)

            if dist < blur_radius * 2:
                brightness = int(200 - (dist / (blur_radius * 2)) * 150)

                if metallic > 0.5:
                    # Metallic: colored reflection (gold-like)
                    tile[y, x] = [brightness, int(brightness*0.8), int(brightness*0.3), 255]
                else:
                    # Dielectric: white reflection
                    tile[y, x] = [brightness, brightness, brightness, 255]

    add_label_border(tile, label)
    return tile


def add_label_border(tile: np.ndarray, label: str):
    """Add label area and border to tile"""
    tile[:30, :] = [80, 80, 120, 255]  # Top label area
    tile[:5, :] = [255, 255, 255, 255]  # Borders
    tile[-5:, :] = [255, 255, 255, 255]
    tile[:, :5] = [255, 255, 255, 255]
    tile[:, -5:] = [255, 255, 255, 255]


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
        description='Demonstrate IBL with environment rotation and roughness sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  rotation  - Show effect of HDR environment rotation (8 angles)
  roughness - Show effect of material roughness (0.0 to 1.0)
  metallic  - Compare metallic vs. dielectric at various roughness

Examples:
  # Rotation sweep
  python examples/ibl_gallery.py --hdr env.hdr --mode rotation

  # Roughness sweep for dielectrics
  python examples/ibl_gallery.py --hdr env.hdr --mode roughness

  # Metallic vs. dielectric comparison
  python examples/ibl_gallery.py --hdr env.hdr --mode metallic
        """
    )

    parser.add_argument('--hdr', '-i', required=True,
                       help='Path to HDR environment map (.hdr or .exr)')
    parser.add_argument('--output', '-o',
                       help='Output image path (auto-generated if not specified)')
    parser.add_argument('--mode', '-m',
                       choices=['rotation', 'roughness', 'metallic', 'all'],
                       default='all',
                       help='Gallery mode (default: all)')
    parser.add_argument('--tile-size', '-s', type=int, default=400,
                       help='Size of each tile in pixels (default: 400)')
    parser.add_argument('--rotation-steps', type=int, default=8,
                       help='Number of rotation angles (default: 8)')
    parser.add_argument('--roughness-steps', type=int, default=10,
                       help='Number of roughness steps (default: 10)')

    args = parser.parse_args()

    # Check HDR file exists
    hdr_path = Path(args.hdr)
    if not hdr_path.exists():
        print(f"Error: HDR file not found: {hdr_path}")
        print("Tip: You can download HDR environments from:")
        print("  - https://polyhaven.com/hdris")
        print("  - https://hdrihaven.com")
        exit(1)

    # Render requested modes
    if args.mode in ['rotation', 'all']:
        output = args.output or 'ibl_rotation.png'
        render_rotation_sweep(
            str(hdr_path),
            output,
            args.tile_size,
            args.rotation_steps
        )

    if args.mode in ['roughness', 'all']:
        output = args.output or 'ibl_roughness.png'
        render_roughness_sweep(
            str(hdr_path),
            output,
            args.tile_size,
            args.roughness_steps
        )

    if args.mode in ['metallic', 'all']:
        output = args.output or 'ibl_metallic.png'
        render_metallic_comparison(
            str(hdr_path),
            output,
            args.tile_size
        )


if __name__ == '__main__':
    main()
