#!/usr/bin/env python3
"""
S2 Demo: Nodata/mask to alpha channel propagation

This example demonstrates converting raster nodata values and masks
to alpha channels for proper transparency handling.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

def create_synthetic_raster_with_nodata(width=512, height=512, bands=3):
    """Create synthetic raster data with nodata regions."""
    print(f"Creating synthetic {bands}-band raster with nodata ({height}x{width})...")
    
    # Create coordinate grids  
    y_coords = np.linspace(-2, 2, height)
    x_coords = np.linspace(-2, 2, width)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Generate base data
    data = np.zeros((bands, height, width), dtype=np.float32)
    
    if bands >= 1:
        # Band 1: Mountain-like elevation (for terrain visualization)
        data[0] = np.exp(-(X**2 + Y**2)/2) + 0.3 * np.sin(X * 3) * np.cos(Y * 3)
    
    if bands >= 2:
        # Band 2: Temperature-like gradient
        data[1] = 0.5 + 0.5 * np.tanh(Y + 0.5 * np.sin(X * 2))
    
    if bands >= 3:
        # Band 3: Vegetation index-like
        r = np.sqrt(X**2 + Y**2)
        data[2] = np.where(r < 1.5, 0.8 - 0.3 * r, 0.1)
    
    # Normalize to [0, 1] range
    for b in range(bands):
        band_min, band_max = data[b].min(), data[b].max()
        if band_max > band_min:
            data[b] = (data[b] - band_min) / (band_max - band_min)
    
    return data


def create_complex_mask(data, seed=42):
    """Create a complex mask with various nodata patterns."""
    np.random.seed(seed)
    bands, height, width = data.shape
    
    # Create mask (True = valid, False = nodata)
    mask = np.ones((height, width), dtype=bool)
    
    # Pattern 1: Circular holes (representing water bodies)
    centers = [(height//4, width//4), (3*height//4, width//4), (height//2, 3*width//4)]
    radii = [height//8, height//12, height//10]
    
    y_indices, x_indices = np.ogrid[:height, :width]
    
    for (cy, cx), radius in zip(centers, radii):
        circle_mask = (x_indices - cx)**2 + (y_indices - cy)**2 < radius**2
        mask[circle_mask] = False
    
    # Pattern 2: Rectangular exclusion zones (representing urban areas)
    rect_zones = [
        (height//6, width//6, height//3, width//8),  # (top, left, height, width)
        (2*height//3, width//2, height//6, width//4),
    ]
    
    for top, left, rect_height, rect_width in rect_zones:
        bottom = min(top + rect_height, height)
        right = min(left + rect_width, width)
        mask[top:bottom, left:right] = False
    
    # Pattern 3: Scattered nodata pixels (sensor errors)
    n_scattered = int(0.02 * height * width)  # 2% scattered nodata
    scattered_y = np.random.randint(0, height, n_scattered)
    scattered_x = np.random.randint(0, width, n_scattered)
    mask[scattered_y, scattered_x] = False
    
    # Pattern 4: Edge artifacts (common in satellite imagery)
    edge_width = 15
    # Top and bottom edges
    mask[:edge_width, :] = False
    mask[-edge_width:, :] = False
    # Left and right edges (partial)
    mask[:, :edge_width//2] = False
    mask[:, -edge_width//2:] = False
    
    return mask


def demonstrate_mask_extraction():
    """Demonstrate mask extraction and alpha synthesis."""
    print("\n=== Mask Extraction Demo ===")
    
    try:
        from forge3d.adapters.rasterio_tiles import (
            extract_masks,
            windowed_read_with_alpha,
            create_synthetic_nodata_mask,
            is_rasterio_available
        )
    except ImportError as e:
        print(f"Could not import rasterio adapter: {e}")
        return None, None
    
    if not is_rasterio_available():
        print("rasterio is not available - demonstrating with synthetic data")
        return demonstrate_mock_mask_extraction()
    
    print("Note: This demo would work with real rasterio datasets with masks.")
    print("Example usage:")
    print("""
import rasterio
from rasterio.windows import Window
from forge3d.adapters.rasterio_tiles import extract_masks, windowed_read_with_alpha

# Open dataset with nodata values
with rasterio.open('data_with_nodata.tif') as dataset:
    # Extract masks
    masks = extract_masks(dataset)
    print(f"Dataset masks shape: {masks.shape}")
    
    # Read data with alpha channel derived from masks
    window = Window(0, 0, 256, 256)
    rgba_data = windowed_read_with_alpha(dataset, window, add_alpha=True)
    print(f"RGBA data shape: {rgba_data.shape}")
    
    # Nodata areas will have alpha=0, valid areas alpha=255
    nodata_pixels = np.sum(rgba_data[:, :, 3] == 0)
    valid_pixels = np.sum(rgba_data[:, :, 3] == 255)
    print(f"Transparent pixels: {nodata_pixels}")
    print(f"Opaque pixels: {valid_pixels}")
""")
    
    return None, None


def demonstrate_mock_mask_extraction():
    """Demonstrate mask extraction with synthetic data."""
    print("Demonstrating mask to alpha conversion with synthetic data...")
    
    # Create synthetic raster data
    width, height, bands = 512, 512, 3
    synthetic_data = create_synthetic_raster_with_nodata(width, height, bands)
    
    # Create complex mask
    mask = create_complex_mask(synthetic_data)
    
    # Apply mask to data (set nodata areas to 0)
    masked_data = synthetic_data.copy()
    for b in range(bands):
        masked_data[b][~mask] = 0.0  # Set nodata to 0
    
    print(f"\nSynthetic data: {bands} bands, {height}x{width} pixels")
    
    # Calculate mask statistics
    valid_pixels = np.sum(mask)
    total_pixels = height * width
    nodata_pixels = total_pixels - valid_pixels
    nodata_fraction = nodata_pixels / total_pixels
    
    print(f"Mask statistics:")
    print(f"  Valid pixels: {valid_pixels:,} ({100*(1-nodata_fraction):.1f}%)")
    print(f"  Nodata pixels: {nodata_pixels:,} ({100*nodata_fraction:.1f}%)")
    
    # Demonstrate alpha synthesis
    print(f"\n=== Alpha Channel Synthesis ===")
    
    # Convert to RGB format for visualization
    if bands >= 3:
        rgb_data = masked_data[:3]  # Take first 3 bands
    else:
        # Replicate single band
        rgb_data = np.repeat(masked_data[0:1], 3, axis=0)
    
    # Convert to uint8 and transpose to (H, W, C)
    rgb_uint8 = (rgb_data * 255).astype(np.uint8)
    rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))
    
    # Create alpha channel from mask
    alpha_channel = (mask * 255).astype(np.uint8)
    
    # Combine to RGBA
    rgba_hwc = np.concatenate([rgb_hwc, alpha_channel[:, :, np.newaxis]], axis=2)
    
    print(f"RGB data shape: {rgb_hwc.shape}")
    print(f"Alpha channel shape: {alpha_channel.shape}")
    print(f"RGBA data shape: {rgba_hwc.shape}")
    
    # Analyze alpha channel
    fully_transparent = np.sum(alpha_channel == 0)
    fully_opaque = np.sum(alpha_channel == 255)
    partially_transparent = total_pixels - fully_transparent - fully_opaque
    
    print(f"\nAlpha channel analysis:")
    print(f"  Fully transparent (alpha=0): {fully_transparent:,} pixels")
    print(f"  Fully opaque (alpha=255): {fully_opaque:,} pixels")
    print(f"  Partially transparent: {partially_transparent:,} pixels")
    
    return rgba_hwc, {
        'total_pixels': total_pixels,
        'valid_pixels': valid_pixels,
        'nodata_pixels': nodata_pixels,
        'nodata_fraction': nodata_fraction,
        'alpha_stats': {
            'transparent': fully_transparent,
            'opaque': fully_opaque,
            'partial': partially_transparent
        }
    }


def demonstrate_different_mask_types():
    """Demonstrate different types of masks and their alpha conversion."""
    print("\n=== Different Mask Types Demo ===")
    
    # Simple test data
    test_size = 64
    test_data = np.random.rand(3, test_size, test_size).astype(np.float32)
    
    mask_types = [
        ("Solid mask", np.ones((test_size, test_size), dtype=bool)),
        ("Checkerboard", ((np.arange(test_size)[:, None] + np.arange(test_size)) % 2).astype(bool)),
        ("Central circle", create_circular_mask(test_size, test_size//4)),
        ("Random 50%", np.random.rand(test_size, test_size) > 0.5),
    ]
    
    results = []
    
    for name, mask in mask_types:
        # Convert mask to alpha
        alpha = (mask * 255).astype(np.uint8)
        
        # Calculate statistics
        valid_count = np.sum(mask)
        total_count = test_size * test_size
        valid_fraction = valid_count / total_count
        
        stats = {
            'name': name,
            'size': f"{test_size}x{test_size}",
            'valid_pixels': valid_count,
            'valid_fraction': valid_fraction,
            'alpha_min': int(np.min(alpha)),
            'alpha_max': int(np.max(alpha)),
            'alpha_unique': len(np.unique(alpha))
        }
        
        results.append(stats)
        
        print(f"\n{name}:")
        print(f"  Valid pixels: {valid_count}/{total_count} ({valid_fraction:.1%})")
        print(f"  Alpha range: [{stats['alpha_min']}, {stats['alpha_max']}]")
        print(f"  Unique alpha values: {stats['alpha_unique']}")
    
    return results


def create_circular_mask(size, radius):
    """Create a circular mask."""
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    return mask


def save_demo_output(rgba_data, stats, output_path):
    """Save demonstration output as PNG."""
    if rgba_data is None:
        print("\nNo data to save.")
        return
    
    try:
        import forge3d
        print(f"\nSaving mask demo output to {output_path}...")
        forge3d.numpy_to_png(str(output_path), rgba_data)
        print(f"Saved RGBA visualization to {output_path}")
        print(f"Transparent areas will show as checkerboard in image viewers")
    except Exception:
        from ._png import write_png
        write_png(str(output_path), rgba_data)
        print(f"Saved {output_path} via fallback")


def save_report(stats, mask_types_results, output_path):
    """Save demonstration report as JSON."""
    import json
    
    report = {
        "demo_name": "S2: Nodata/Mask to Alpha Propagation Demo",
        "description": "Demonstration of mask extraction and alpha channel synthesis",
        "mask_statistics": stats,
        "mask_types_tested": mask_types_results,
        "features_demonstrated": [
            "Complex mask patterns (circular, rectangular, scattered)",
            "Alpha channel synthesis from boolean masks", 
            "RGBA output with transparency",
            "Mask statistics and analysis"
        ],
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved demonstration report to {report_path}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='S2: Mask to alpha propagation demo')
    parser.add_argument('--out', type=str, default='reports/s2_mask.png',
                       help='Output PNG path (default: reports/s2_mask.png)')
    args = parser.parse_args()
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Forge3D Workstream S2: Mask to Alpha Propagation Demo")
    print("=" * 55)
    
    # Demonstrate mask extraction
    rgba_data, stats = demonstrate_mask_extraction()
    
    # Demonstrate different mask types
    mask_types_results = demonstrate_different_mask_types()
    
    # Save outputs
    if rgba_data is not None and stats is not None:
        save_demo_output(rgba_data, stats, output_path)
        save_report(stats, mask_types_results, output_path)
    
    print(f"\n=== Demo Complete ===")
    print(f"Output: {output_path}")
    print("This demo shows how nodata masks become alpha channels.")
    print("The output PNG will have transparency in masked areas!")


if __name__ == "__main__":
    main()
