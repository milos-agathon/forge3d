#!/usr/bin/env python3
"""
Datashader overlay demo for forge3d.

Demonstrates integration between Datashader's large-scale point rendering
and forge3d's overlay system. Creates synthetic point data (~1M points),
uses Datashader to aggregate/shade it, then composites as an overlay
in forge3d with proper coordinate alignment.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(0)

# Check for datashader availability
try:
    from forge3d.adapters import is_datashader_available, DatashaderAdapter
    if not is_datashader_available():
        print("Datashader is not available. Install with: pip install datashader")
        sys.exit(0)
    
    from forge3d.adapters import (
        rgba_view_from_agg, validate_alignment, 
        to_overlay_texture, shade_to_overlay
    )
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd
    
except ImportError as e:
    print(f"Failed to import datashader dependencies: {e}")
    print("Install with: pip install datashader pandas")
    sys.exit(0)


def generate_synthetic_points(n_points: int = 1_000_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic point data for demonstration.
    
    Creates a dataset with spatial clustering and value variation
    to showcase datashader's aggregation capabilities.
    """
    np.random.seed(seed)
    
    # Create several clusters of points
    n_clusters = 5
    points_per_cluster = n_points // n_clusters
    
    clusters = []
    for i in range(n_clusters):
        # Cluster centers distributed across extent
        center_x = -100 + (i * 50)  # -100 to 100
        center_y = -50 + (i * 25)   # -50 to 50
        
        # Generate points around cluster center
        cluster_x = np.random.normal(center_x, 15, points_per_cluster)
        cluster_y = np.random.normal(center_y, 10, points_per_cluster)
        
        # Add some value variation (could represent elevation, temperature, etc.)
        cluster_value = np.random.gamma(2, i + 1, points_per_cluster)
        
        clusters.append(pd.DataFrame({
            'x': cluster_x,
            'y': cluster_y, 
            'value': cluster_value
        }))
    
    # Combine all clusters
    df = pd.concat(clusters, ignore_index=True)
    
    # Add some random scatter outside clusters
    n_scatter = n_points - len(df)
    if n_scatter > 0:
        scatter_x = np.random.uniform(-120, 120, n_scatter)
        scatter_y = np.random.uniform(-70, 70, n_scatter)
        scatter_value = np.random.exponential(1, n_scatter)
        
        scatter_df = pd.DataFrame({
            'x': scatter_x,
            'y': scatter_y,
            'value': scatter_value
        })
        
        df = pd.concat([df, scatter_df], ignore_index=True)
    
    print(f"Generated {len(df)} synthetic points")
    print(f"X range: {df.x.min():.1f} to {df.x.max():.1f}")
    print(f"Y range: {df.y.min():.1f} to {df.y.max():.1f}")
    print(f"Value range: {df.value.min():.3f} to {df.value.max():.3f}")
    
    return df


def create_datashader_overlay(df: pd.DataFrame, 
                             extent: tuple = (-120, -70, 120, 70),
                             width: int = 800, 
                             height: int = 600) -> dict:
    """
    Create datashader aggregation and convert to forge3d overlay.
    
    Args:
        df: Point data with x, y, value columns
        extent: (xmin, ymin, xmax, ymax) coordinate bounds
        width: Canvas width in pixels
        height: Canvas height in pixels
        
    Returns:
        Dictionary with overlay texture data and metadata
    """
    print(f"Creating datashader canvas ({width}x{height}) for extent {extent}")
    
    # Create datashader canvas
    canvas = ds.Canvas(plot_width=width, plot_height=height, 
                      x_range=(extent[0], extent[2]), 
                      y_range=(extent[1], extent[3]))
    
    # Aggregate points by mean value
    print(f"Aggregating {len(df)} points...")
    start_time = time.time()
    agg = canvas.points(df, 'x', 'y', ds.mean('value'))
    agg_time = time.time() - start_time
    print(f"Aggregation completed in {agg_time:.3f}s")
    
    # Shade the aggregation
    print("Shading aggregation with viridis colormap...")
    start_time = time.time()
    img = tf.shade(agg, cmap='viridis', how='linear')
    shade_time = time.time() - start_time
    print(f"Shading completed in {shade_time:.3f}s")
    
    # Convert to RGBA overlay using forge3d adapter
    print("Converting to forge3d overlay...")
    start_time = time.time()
    
    # Use the convenience function
    overlay_data = shade_to_overlay(agg, extent, cmap='viridis', how='linear')
    
    convert_time = time.time() - start_time
    print(f"Conversion completed in {convert_time:.3f}s")
    
    # Add timing metadata
    overlay_data['performance'] = {
        'aggregation_time_s': agg_time,
        'shading_time_s': shade_time, 
        'conversion_time_s': convert_time,
        'total_time_s': agg_time + shade_time + convert_time,
        'points_processed': len(df),
        'points_per_second': len(df) / (agg_time + shade_time + convert_time)
    }
    
    return overlay_data


def create_base_terrain(width: int = 800, height: int = 600) -> np.ndarray:
    """
    Create a simple base terrain for the overlay demonstration.
    
    This provides a background context to show how the datashader
    overlay composites with other rendered content.
    """
    print(f"Creating base terrain ({width}x{height})...")
    
    # Simple procedural heightmap  
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32) 
    X, Y = np.meshgrid(x, y)
    
    # Create some gentle rolling terrain
    Z = (np.sin(X * 3) * np.cos(Y * 2) * 0.1 + 
         np.sin(X * 7) * np.sin(Y * 5) * 0.05 +
         (X**2 + Y**2) * 0.2)
    
    # Normalize to reasonable elevation range
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * 100  # 0-100m elevation
    
    return Z


def main():
    parser = argparse.ArgumentParser(description='Datashader overlay demo')
    parser.add_argument('--points', type=int, default=1_000_000,
                       help='Number of synthetic points to generate')
    parser.add_argument('--width', type=int, default=800,
                       help='Canvas width')
    parser.add_argument('--height', type=int, default=600, 
                       help='Canvas height')
    parser.add_argument('--output', type=str, default='examples/output',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Datashader Overlay Demo ===")
    print(f"Points: {args.points:,}")
    print(f"Canvas size: {args.width}x{args.height}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Generate synthetic point data
    df = generate_synthetic_points(args.points, args.seed)
    
    # Define coordinate extent (matches synthetic data range)
    extent = (-120, -70, 120, 70)  # xmin, ymin, xmax, ymax
    
    # Create datashader overlay
    start_total = time.time()
    overlay_data = create_datashader_overlay(df, extent, args.width, args.height)
    
    # Validate alignment (basic check)
    alignment_info = validate_alignment(extent, None, args.width, args.height)
    overlay_data['alignment'] = alignment_info
    
    print(f"Overlay created successfully:")
    print(f"  RGBA shape: {overlay_data['rgba'].shape}")
    print(f"  Data type: {overlay_data['rgba'].dtype}")
    print(f"  Memory size: {overlay_data['total_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Contiguous: {overlay_data['is_contiguous']}")
    print()
    
    # For demonstration purposes, save the overlay as PNG  
    # In a real application, this would be passed to forge3d's overlay system
    print("Saving overlay as PNG...")
    
    # Convert RGBA to RGB for PNG (remove alpha channel for demo)
    rgba = overlay_data['rgba']
    rgb = rgba[..., :3]  # Drop alpha channel
    
    # Save using forge3d's PNG utility
    output_png = output_dir / 'datashader_overlay_demo.png'
    f3d.numpy_to_png(str(output_png), rgb)
    print(f"Saved: {output_png}")
    
    # Save metadata JSON
    metadata = {
        'extent': extent,
        'width': args.width,
        'height': args.height,
        'points_processed': args.points,
        'pixel_width': overlay_data['pixel_width'],
        'pixel_height': overlay_data['pixel_height'],
        'performance': overlay_data['performance'],
        'alignment': overlay_data['alignment'],
        'forge3d_version': f3d.__version__,
        'canvas_config': {
            'cmap': 'viridis',
            'aggregation': 'mean',
            'shading': 'linear'
        }
    }
    
    output_json = output_dir / 'datashader_overlay_demo.json'
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {output_json}")
    
    total_time = time.time() - start_total
    print()
    print("=== Summary ===")
    print(f"Total runtime: {total_time:.3f}s")
    print(f"Points per second: {args.points / total_time:.0f}")
    print(f"Memory efficiency: {overlay_data['total_bytes'] / args.points:.1f} bytes/point")
    
    # Validate zero-copy where possible
    print(f"Zero-copy achieved: {overlay_data['shares_memory']}")
    
    # Check memory usage is within budget
    memory_mb = overlay_data['total_bytes'] / 1024 / 1024
    budget_mb = 512
    print(f"Memory usage: {memory_mb:.1f} MB / {budget_mb} MB budget")
    
    if memory_mb <= budget_mb:
        print("✅ Within memory budget")
    else:
        print("⚠️  Exceeds memory budget")
    
    print("OK")  # Success indicator for automated testing


if __name__ == '__main__':
    main()