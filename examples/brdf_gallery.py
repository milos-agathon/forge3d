#!/usr/bin/env python3
"""
P7-07: BRDF Gallery Generator (Milestone 0 Complete)

Renders a gallery mosaic of BRDF tiles using the offscreen renderer.
Compares different BRDF models across varying roughness values.

This example demonstrates:
- Offscreen BRDF tile rendering via forge3d.render_brdf_tile()
- CSV parsing for models and roughness values
- Milestone 0 debug modes: NDF-only, G-only, DFG-only, roughness visualization
- Deterministic PNG gallery generation with no tone mapping and exposure=1.0
- Light intensity tuned to prevent clipping (peak < 0.95)

Usage:
    # Default: GGX and Disney at 5 roughness values
    python examples/brdf_gallery.py
    
    # Custom models and roughness
    python examples/brdf_gallery.py --models lambert,phong,ggx,disney --roughness 0.1,0.3,0.5,0.7,0.9
    
    # NDF-only debug mode
    python examples/brdf_gallery.py --ndf-only
    
    # Milestone 0: G-only debug mode (outputs Smith G as grayscale)
    python examples/brdf_gallery.py --g-only
    
    # Milestone 0: DFG-only debug mode (outputs D*F*G pre-division)
    python examples/brdf_gallery.py --dfg-only
    
    # Milestone 0: Roughness visualization (validates uniform flow)
    python examples/brdf_gallery.py --roughness-visualize
    
    # Custom tile size
    python examples/brdf_gallery.py --tile-size 256 256
    
    # Custom output
    python examples/brdf_gallery.py --out reports/brdf_comparison.png
"""

import sys
from pathlib import Path

# Add python directory to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

import argparse
from typing import List, Optional
import numpy as np

try:
    import forge3d as f3d
except ImportError as e:
    print(f"Error: Could not import forge3d: {e}")
    print("Please build forge3d with: maturin develop --release")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Mosaic saving will be limited.")


def parse_csv_list(csv_str: str) -> List[str]:
    """Parse comma-separated values."""
    return [item.strip() for item in csv_str.split(',') if item.strip()]


def parse_csv_floats(csv_str: str) -> List[float]:
    """Parse comma-separated float values."""
    try:
        return [float(item.strip()) for item in csv_str.split(',') if item.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float in CSV: {e}")


def add_text_label(img: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
    """Add text label to an image using PIL.
    
    Args:
        img: RGBA image array
        text: Text to add
        position: 'top' or 'bottom'
        
    Returns:
        Labeled image array
    """
    if not HAS_PIL:
        return img
    
    pil_img = Image.fromarray(img, mode='RGBA')
    draw = ImageDraw.Draw(pil_img)
    
    try:
        # Try system font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            # Try another common font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            # Fallback to default
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Position text
    x = (pil_img.width - text_w) // 2
    if position == 'top':
        y = 8
    else:  # bottom
        y = pil_img.height - text_h - 8
    
    # Draw semi-transparent background
    padding = 4
    draw.rectangle(
        [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
        fill=(0, 0, 0, 200)
    )
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    return np.array(pil_img)


def create_mosaic(tiles: List[List[np.ndarray]], gap: int = 4) -> np.ndarray:
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
    mosaic = np.full((mosaic_h, mosaic_w, 4), [20, 20, 20, 255], dtype=np.uint8)
    
    # Place tiles
    for row in range(rows):
        for col in range(cols):
            y = row * (tile_h + gap)
            x = col * (tile_w + gap)
            mosaic[y:y+tile_h, x:x+tile_w] = tiles[row][col]
    
    return mosaic


def render_gallery(
    models: List[str],
    roughness_values: List[float],
    tile_size: tuple,
    ndf_only: bool,
    g_only: bool,
    dfg_only: bool,
    spec_only: bool,
    roughness_visualize: bool,
    exposure: float,
    light_intensity: float,
    output_path: Path,
    debug_dot_products: bool,
    mode: Optional[str],
    no_labels: bool,
) -> None:
    """Render BRDF gallery mosaic.
    
    Args:
        models: List of BRDF model names
        roughness_values: List of roughness values
        tile_size: (width, height) for each tile
        ndf_only: If True, render NDF-only mode
        g_only: If True, render G-only mode (Milestone 0)
        dfg_only: If True, render DFG-only mode (Milestone 0)
        roughness_visualize: If True, render roughness visualization (Milestone 0)
        exposure: Exposure multiplier (Milestone 4)
        light_intensity: Light intensity (Milestone 4)
        output_path: Path to save the mosaic
        debug_dot_products: If True, log min/max N·L and N·V values for debugging
    """
    width, height = tile_size
    
    # Determine active debug mode for display (mode overrides individual toggles)
    mode_display_map = {
        None: "Full BRDF",
        "full": "Full BRDF",
        "ndf": "NDF-only",
        "g": "G-only",
        "dfg": "DFG-only",
        "spec": "SPEC-only",
        "roughness": "Roughness Visualize",
    }
    if mode is not None:
        debug_mode = mode_display_map.get(mode, f"Unknown mode: {mode}")
    else:
        debug_mode = "Full BRDF"
        if ndf_only:
            debug_mode = "NDF-only"
        elif g_only:
            debug_mode = "G-only"
        elif dfg_only:
            debug_mode = "DFG-only"
        elif spec_only:
            debug_mode = "SPEC-only"
        elif roughness_visualize:
            debug_mode = "Roughness Visualize"
    
    print(f"Rendering BRDF Gallery:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Roughness values: {', '.join(f'{r:.2f}' for r in roughness_values)}")
    print(f"  Tile size: {width}×{height}")
    print(f"  Debug mode: {debug_mode}")
    print(f"  Output: {output_path}")
    print()
    
    # Model name mapping for display
    model_display_names = {
        'lambert': 'Lambert',
        'phong': 'Phong',
        'ggx': 'Cook-Torrance GGX',
        'disney': 'Disney Principled',
    }
    
    # Render tiles
    all_tiles = []
    
    for model in models:
        print(f"Rendering {model_display_names.get(model, model)}...")
        row_tiles = []
        
        for roughness in roughness_values:
            print(f"  Roughness {roughness:.2f}...", end=' ', flush=True)
            
            try:
                # Call offscreen renderer with Milestone 0 debug toggles and Milestone 4 params
                tile = f3d.render_brdf_tile(
                    model, roughness, width, height, 
                    ndf_only=ndf_only,
                    g_only=g_only,
                    dfg_only=dfg_only,
                    spec_only=spec_only,
                    roughness_visualize=roughness_visualize,
                    exposure=exposure,
                    light_intensity=light_intensity,
                    debug_dot_products=debug_dot_products,
                    mode=mode,
                )
                
                # Milestone 4: Optional labels - stamp model, r, and alpha=r^2 for audits
                if not no_labels:
                    display_name = model_display_names.get(model, model)
                    alpha = roughness * roughness
                    tile = add_text_label(tile, display_name, position='top')
                    tile = add_text_label(tile, f"r={roughness:.2f} α={alpha:.4f}", position='bottom')
                
                row_tiles.append(tile)
                print("✓")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                # Create error placeholder
                error_tile = np.full((height, width, 4), [100, 0, 0, 255], dtype=np.uint8)
                error_tile = add_text_label(error_tile, "ERROR", position='top')
                row_tiles.append(error_tile)
        
        all_tiles.append(row_tiles)
    
    # Create mosaic
    print("\nStitching mosaic...")
    mosaic = create_mosaic(all_tiles, gap=4)
    print(f"  Mosaic size: {mosaic.shape[1]}×{mosaic.shape[0]}")
    
    # Save mosaic
    print(f"\nSaving to {output_path}...")
    
    if HAS_PIL:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pil_img = Image.fromarray(mosaic, mode='RGBA')
        pil_img.save(output_path)
        print(f"✓ Saved BRDF gallery: {output_path}")
    else:
        print("✗ PIL not available, cannot save PNG")
        # Fallback: save as numpy
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, mosaic)
        print(f"  Saved as numpy array: {npy_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a BRDF gallery mosaic using offscreen rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default gallery
  python examples/brdf_gallery.py
  
  # Custom models (CSV)
  python examples/brdf_gallery.py --models lambert,phong,ggx,disney
  
  # Custom roughness sweep (CSV)
  python examples/brdf_gallery.py --roughness 0.1,0.3,0.5,0.7,0.9
  
  # NDF-only debug mode
  python examples/brdf_gallery.py --ndf-only
  
  # Custom tile size
  python examples/brdf_gallery.py --tile-size 512 512
  
  # Custom output path
  python examples/brdf_gallery.py --out reports/brdf_gallery.png
        """,
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='ggx,disney',
        help='Comma-separated BRDF models (default: ggx,disney)',
    )
    parser.add_argument(
        '--roughness',
        type=str,
        default='0.1,0.3,0.5,0.7,0.9',
        help='Comma-separated roughness values (default: 0.1,0.3,0.5,0.7,0.9)',
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=('WIDTH', 'HEIGHT'),
        help='Tile size in pixels (default: 256 256)',
    )
    parser.add_argument(
        '--ndf-only',
        action='store_true',
        help='Enable NDF-only debug mode',
    )
    parser.add_argument(
        '--g-only',
        action='store_true',
        help='Milestone 0: Enable G-only debug mode (outputs Smith G as grayscale)',
    )
    parser.add_argument(
        '--dfg-only',
        action='store_true',
        help='Milestone 0: Enable DFG-only debug mode (outputs D*F*G pre-division)',
    )
    parser.add_argument(
        '--spec-only',
        action='store_true',
        help='Milestone 0: Enable SPEC-only debug mode (outputs specular-only term)',
    )
    parser.add_argument(
        '--roughness-visualize',
        action='store_true',
        help='Milestone 0: Enable roughness visualization (outputs vec3(roughness))',
    )
    parser.add_argument(
        '--exposure',
        type=float,
        default=1.0,
        help='Milestone 4: Exposure multiplier (default: 1.0)',
    )
    parser.add_argument(
        '--light-intensity',
        type=float,
        default=0.8,
        help='Milestone 4: Light intensity (default: 0.8)',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'ndf', 'g', 'dfg', 'spec', 'roughness'],
        default=None,
        help='Milestone 4: Output mode selector (overrides individual toggles)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Milestone 4: Do not stamp labels on each tile',
    )
    parser.add_argument(
        '--debug-dot-products',
        action='store_true',
        help='Milestone 1: Log min/max N·L and N·V values for debugging',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('brdf_gallery.png'),
        help='Output path for mosaic PNG (default: brdf_gallery.png)',
    )
    
    args = parser.parse_args()
    
    # Parse CSV inputs
    models = parse_csv_list(args.models)
    roughness_values = parse_csv_floats(args.roughness)
    
    # Validate inputs
    valid_models = ['lambert', 'phong', 'ggx', 'disney']
    for model in models:
        if model not in valid_models:
            print(f"Error: Invalid model '{model}'. Valid models: {', '.join(valid_models)}")
            sys.exit(1)
    
    for roughness in roughness_values:
        if not (0.0 <= roughness <= 1.0):
            print(f"Warning: Roughness {roughness} outside [0, 1] range. Will be clamped.")
    
    if not models:
        print("Error: No models specified")
        sys.exit(1)
    
    if not roughness_values:
        print("Error: No roughness values specified")
        sys.exit(1)
    
    # Render gallery
    try:
        render_gallery(
            models=models,
            roughness_values=roughness_values,
            tile_size=tuple(args.tile_size),
            ndf_only=args.ndf_only,
            g_only=args.g_only,
            dfg_only=args.dfg_only,
            spec_only=args.spec_only,
            roughness_visualize=args.roughness_visualize,
            exposure=args.exposure,
            light_intensity=args.light_intensity,
            output_path=args.out,
            debug_dot_products=args.debug_dot_products,
            mode=args.mode,
            no_labels=args.no_labels,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
