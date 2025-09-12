#!/usr/bin/env python3
"""
S1 Demo: Rasterio windowed reads and block iteration

This example demonstrates the basic windowed reading functionality
for raster data using the forge3d rasterio adapter.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

def create_synthetic_raster(width=1024, height=1024, bands=3):
    """Create synthetic raster data for demonstration."""
    print(f"Creating synthetic {bands}-band raster ({height}x{width})...")
    
    # Create coordinate grids
    y_coords = np.linspace(-1, 1, height)
    x_coords = np.linspace(-1, 1, width)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Generate synthetic bands with different patterns
    data = np.zeros((bands, height, width), dtype=np.float32)
    
    if bands >= 1:
        # Band 1: Radial gradient
        data[0] = np.sqrt(X*X + Y*Y)
    
    if bands >= 2:
        # Band 2: Sine wave pattern
        data[1] = np.sin(X * np.pi * 4) * np.cos(Y * np.pi * 4)
    
    if bands >= 3:
        # Band 3: Checkerboard pattern
        data[2] = ((X > 0).astype(float) - 0.5) * ((Y > 0).astype(float) - 0.5)
    
    # Add additional bands if requested
    for b in range(3, bands):
        # Random noise with decreasing amplitude
        amplitude = 1.0 / (1 + b - 3)
        data[b] = np.random.rand(height, width) * amplitude
    
    # Normalize to [0, 1] range
    for b in range(bands):
        band_min, band_max = data[b].min(), data[b].max()
        if band_max > band_min:
            data[b] = (data[b] - band_min) / (band_max - band_min)
    
    return data


def demonstrate_windowed_reading():
    """Demonstrate windowed reading functionality."""
    print("\n=== Windowed Reading Demo ===")
    
    try:
        from forge3d.adapters.rasterio_tiles import (
            windowed_read, 
            block_iterator, 
            get_dataset_info,
            validate_window,
            is_rasterio_available
        )
    except ImportError as e:
        print(f"Could not import rasterio adapter: {e}")
        print("Install with: pip install 'forge3d[raster]'")
        return None
    
    if not is_rasterio_available():
        print("rasterio is not available - using mock demonstration")
        return demonstrate_mock_windowed_reading()
    
    # In a real scenario, you would open an actual raster file:
    # with rasterio.open('path/to/raster.tif') as dataset:
    #     ...
    
    print("Note: This demo would work with real rasterio datasets.")
    print("Example usage:")
    print("""
import rasterio
from rasterio.windows import Window
from forge3d.adapters.rasterio_tiles import windowed_read, block_iterator

# Open dataset
with rasterio.open('data.tif') as dataset:
    # Define window (col_off, row_off, width, height)
    window = Window(100, 100, 256, 256)
    
    # Read windowed data
    data = windowed_read(dataset, window, out_shape=(128, 128))
    print(f"Read {data.shape} array from window")
    
    # Iterate over blocks
    for i, (block_window, block_data) in enumerate(block_iterator(dataset)):
        print(f"Block {i}: {block_window} -> {block_data.shape}")
        if i >= 5:  # Limit output
            break
""")
    
    return None


def demonstrate_mock_windowed_reading():
    """Demonstrate windowed reading with mock data."""
    print("Demonstrating windowed reading concept with synthetic data...")
    
    # Create synthetic raster
    width, height, bands = 1024, 1024, 3
    synthetic_data = create_synthetic_raster(width, height, bands)
    
    # Simulate windowed reading
    print(f"\nSynthetic raster: {bands} bands, {height}x{width} pixels")
    
    # Define windows to "read"
    windows = [
        {"name": "Top-left", "row": 0, "col": 0, "height": 256, "width": 256},
        {"name": "Center", "row": 384, "col": 384, "height": 256, "width": 256},
        {"name": "Bottom-right", "row": 768, "col": 768, "height": 256, "width": 256}
    ]
    
    results = []
    for window in windows:
        # Extract window data
        row_start = window["row"]
        row_end = row_start + window["height"]
        col_start = window["col"] 
        col_end = col_start + window["width"]
        
        window_data = synthetic_data[:, row_start:row_end, col_start:col_end]
        
        # Calculate statistics
        stats = {
            "name": window["name"],
            "shape": window_data.shape,
            "mean": float(np.mean(window_data)),
            "std": float(np.std(window_data)),
            "min": float(np.min(window_data)),
            "max": float(np.max(window_data)),
            "window": f"({row_start}:{row_end}, {col_start}:{col_end})"
        }
        results.append(stats)
        
        print(f"\nWindow '{stats['name']}' {stats['window']}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    return synthetic_data, results


def demonstrate_block_iteration():
    """Demonstrate block iteration concept."""
    print("\n=== Block Iteration Demo ===")
    
    # Create smaller synthetic raster for block demo
    width, height, bands = 512, 384, 1
    synthetic_data = create_synthetic_raster(width, height, bands)
    
    # Simulate block iteration with different block sizes
    block_sizes = [128, 64, 256]
    
    for block_size in block_sizes:
        print(f"\nBlock iteration with {block_size}x{block_size} blocks:")
        
        blocks_processed = 0
        total_pixels = 0
        
        # Iterate over blocks
        for row_start in range(0, height, block_size):
            for col_start in range(0, width, block_size):
                # Calculate actual block size (may be smaller at edges)
                actual_height = min(block_size, height - row_start)
                actual_width = min(block_size, width - col_start)
                
                # Extract block
                row_end = row_start + actual_height
                col_end = col_start + actual_width
                block_data = synthetic_data[:, row_start:row_end, col_start:col_end]
                
                blocks_processed += 1
                total_pixels += actual_height * actual_width
                
                if blocks_processed <= 3:  # Show first few blocks
                    print(f"  Block {blocks_processed}: ({row_start},{col_start}) -> "
                          f"{actual_height}x{actual_width} pixels, "
                          f"mean={np.mean(block_data):.3f}")
        
        print(f"  Total: {blocks_processed} blocks, {total_pixels} pixels")
        coverage = total_pixels / (width * height)
        print(f"  Coverage: {coverage:.1%} (should be 100%)")


def save_demo_output(data, results, output_path):
    """Save demonstration output as PNG using forge3d if available."""
    if data is None:
        print("\nNo data to save.")
        return
    
    try:
        import forge3d
        print(f"\nSaving demo output to {output_path}...")
        
        # Take the first 3 bands and convert to uint8
        if data.shape[0] >= 3:
            rgb_data = data[:3]  # RGB
        else:
            # Replicate single band to RGB
            rgb_data = np.repeat(data[0:1], 3, axis=0)
        
        # Convert to uint8 [0, 255] range
        rgb_uint8 = (rgb_data * 255).astype(np.uint8)
        
        # Transpose to (H, W, C) format
        rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))
        
        # Use forge3d to save PNG
        forge3d.numpy_to_png(str(output_path), rgb_hwc)
        print(f"Saved demo visualization to {output_path}")
        
    except ImportError:
        print("forge3d not available for saving PNG output")
    except Exception as e:
        print(f"Error saving output: {e}")


def save_report(results, output_path):
    """Save demonstration report as JSON."""
    import json
    
    report = {
        "demo_name": "S1: Rasterio Windowed Reads Demo",
        "description": "Demonstration of windowed reading and block iteration concepts",
        "synthetic_data": {
            "bands": 3,
            "height": 1024,
            "width": 1024,
            "data_type": "float32"
        },
        "windows_tested": len(results) if results else 0,
        "window_results": results or [],
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved demonstration report to {report_path}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='S1: Rasterio windowed reads demo')
    parser.add_argument('--out', type=str, default='reports/s1_windows.png',
                       help='Output PNG path (default: reports/s1_windows.png)')
    args = parser.parse_args()
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Forge3D Workstream S1: Rasterio Windowed Reads Demo")
    print("=" * 50)
    
    # Demonstrate windowed reading
    demo_data = demonstrate_windowed_reading()
    
    # Demonstrate block iteration
    demonstrate_block_iteration()
    
    # If we got mock data, save it
    if demo_data and isinstance(demo_data, tuple):
        synthetic_data, results = demo_data
        save_demo_output(synthetic_data, results, output_path)
        save_report(results, output_path)
    
    print(f"\n=== Demo Complete ===")
    print(f"Output: {output_path}")
    print("This demo shows the concepts behind windowed reading.")
    print("With rasterio installed, you could process real GeoTIFF files!")


if __name__ == "__main__":
    main()