# examples/mpl_cmap_demo.py
# Demo for Matplotlib colormap interop with forge3d.
# This exists to visualize and validate Workstream R1 behavior.
# RELEVANT FILES:python/forge3d/adapters/mpl_cmap.py,tests/test_mpl_cmap.py,examples/_import_shim.py
#!/usr/bin/env python3
"""
Matplotlib Colormap Integration Demo

This example demonstrates the integration between matplotlib colormaps
and forge3d, showing how to:
1. Use matplotlib colormap names with forge3d terrain rendering
2. Apply matplotlib normalization to data
3. Compare forge3d output with matplotlib reference
4. Handle optional matplotlib dependency gracefully

Usage:
    python examples/mpl_cmap_demo.py
    python examples/mpl_cmap_demo.py --out reports/r1_cmap.png
    python examples/mpl_cmap_demo.py --colormap plasma --size 1024x768
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Optional
import json

import numpy as np

# Ensure in-repo imports work without install (dev convenience)
try:
    from _import_shim import ensure_repo_import
    ensure_repo_import()
except Exception:
    pass

# Import forge3d (after shim)
try:
    import forge3d as f3d
    from forge3d.adapters import (
        is_matplotlib_available,
        matplotlib_to_forge3d_colormap,
        matplotlib_normalize,
        get_matplotlib_colormap_names,
    )
except ImportError as e:
    print(f"Error: Cannot import forge3d: {e}")
    print("Fix: either set PYTHONPATH to include 'python' or run 'pip install -U maturin' then 'maturin develop --release'")
    sys.exit(1)

# Optional matplotlib import for reference comparison
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_size(size_str: str) -> Tuple[int, int]:
    """Parse size string like '1024x768' into (width, height) tuple."""
    try:
        parts = size_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError("Size must be in format WIDTHxHEIGHT")
        width, height = int(parts[0]), int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        return width, height
    except (ValueError, IndexError) as e:
        raise argparse.ArgumentTypeError(f"Invalid size '{size_str}': {e}")


def create_test_heightmap(width: int, height: int, seed: int = 42) -> np.ndarray:
    """
    Create a synthetic heightmap with interesting terrain features.
    
    Args:
        width: Width of heightmap
        height: Height of heightmap  
        seed: Random seed for reproducibility
        
    Returns:
        2D heightmap array (H, W) with values in [0, 1]
    """
    rng = np.random.RandomState(seed)
    
    # Create coordinate grids
    y, x = np.mgrid[0:height, 0:width]
    x_norm = x / (width - 1)
    y_norm = y / (height - 1)
    
    # Base terrain with multiple frequency components
    terrain = np.zeros((height, width))
    
    # Large scale features
    terrain += 0.5 * np.sin(2 * np.pi * x_norm * 1.5) * np.cos(2 * np.pi * y_norm * 1.2)
    
    # Medium scale features  
    terrain += 0.3 * np.sin(2 * np.pi * x_norm * 3.7) * np.cos(2 * np.pi * y_norm * 4.1)
    
    # Small scale noise
    terrain += 0.2 * rng.normal(0, 1, (height, width))
    
    # Add some peaks
    center_x, center_y = width // 2, height // 2
    for dx, dy, strength in [(0.2, 0.1, 0.8), (-0.1, -0.2, 0.6), (0.3, -0.1, 0.4)]:
        peak_x = center_x + dx * width
        peak_y = center_y + dy * height
        dist = np.sqrt((x - peak_x)**2 + (y - peak_y)**2)
        terrain += strength * np.exp(-dist / (min(width, height) * 0.15))
    
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    return terrain.astype(np.float32)


def demonstrate_colormap_conversion(colormap_name: str) -> dict:
    """
    Demonstrate matplotlib colormap conversion to forge3d format.
    
    Args:
        colormap_name: Name of matplotlib colormap
        
    Returns:
        Dictionary with conversion results and metadata
    """
    print(f"\n=== Demonstrating Colormap: {colormap_name} ===")
    
    if not is_matplotlib_available():
        print("Warning: matplotlib not available, skipping colormap conversion demo")
        return {'colormap': colormap_name, 'available': False}
    
    results = {
        'colormap': colormap_name,
        'available': True,
        'conversion_time': 0,
        'lut_shape': None,
        'sample_colors': {}
    }
    
    # Time the conversion
    start_time = time.perf_counter()
    try:
        lut = matplotlib_to_forge3d_colormap(colormap_name, n_colors=256)
        conversion_time = time.perf_counter() - start_time
        
        results['conversion_time'] = conversion_time
        results['lut_shape'] = lut.shape
        
        print(f"  Conversion time: {conversion_time*1000:.3f} ms")
        print(f"  LUT shape: {lut.shape}")
        print(f"  LUT dtype: {lut.dtype}")
        
        # Sample some colors
        n_samples = len(lut)
        sample_indices = [0, n_samples//4, n_samples//2, 3*n_samples//4, n_samples-1]
        
        for i in sample_indices:
            color = lut[i]
            results['sample_colors'][f'index_{i}'] = color.tolist()
            print(f"  Color[{i:3d}]: RGBA({color[0]:3d}, {color[1]:3d}, {color[2]:3d}, {color[3]:3d})")
            
        # Verify properties
        assert np.all(lut >= 0) and np.all(lut <= 255)
        assert lut.dtype == np.uint8
        print("  * LUT format validation passed")
        
    except Exception as e:
        print(f"  * Conversion failed: {e}")
        results['error'] = str(e)
    
    return results


def demonstrate_normalization(data: np.ndarray, norm_type: str = 'linear') -> dict:
    """
    Demonstrate matplotlib normalization integration.
    
    Args:
        data: Input data to normalize
        norm_type: Type of normalization ('linear', 'log')
        
    Returns:
        Dictionary with normalization results
    """
    print(f"\n=== Demonstrating Normalization: {norm_type} ===")
    
    if not is_matplotlib_available():
        print("Warning: matplotlib not available, skipping normalization demo")
        return {'norm_type': norm_type, 'available': False}
    
    results = {
        'norm_type': norm_type,
        'available': True,
        'input_stats': {
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std())
        }
    }
    
    print(f"  Input data shape: {data.shape}")
    print(f"  Input range: [{data.min():.6f}, {data.max():.6f}]")
    print(f"  Input stats: mean={data.mean():.6f}, std={data.std():.6f}")
    
    try:
        start_time = time.perf_counter()
        normalized = matplotlib_normalize(data, norm=norm_type)
        normalization_time = time.perf_counter() - start_time
        
        results['normalization_time'] = normalization_time
        results['output_stats'] = {
            'min': float(normalized.min()),
            'max': float(normalized.max()),
            'mean': float(normalized.mean()),
            'std': float(normalized.std())
        }
        
        print(f"  Normalization time: {normalization_time*1000:.3f} ms")
        print(f"  Output range: [{normalized.min():.6f}, {normalized.max():.6f}]")
        print(f"  Output stats: mean={normalized.mean():.6f}, std={normalized.std():.6f}")
        
        # Verify properties
        if norm_type == 'linear':
            # Linear normalization should map min->0, max->1
            expected_min, expected_max = 0.0, 1.0
            np.testing.assert_allclose([normalized.min(), normalized.max()], 
                                     [expected_min, expected_max], atol=1e-6)
            print("  * Linear normalization validation passed")
        elif norm_type == 'log':
            # Log normalization should produce finite values
            assert np.all(np.isfinite(normalized))
            print("  * Log normalization validation passed")
            
        results['normalized_data'] = normalized
        
    except Exception as e:
        print(f"  * Normalization failed: {e}")
        results['error'] = str(e)
    
    return results


def render_terrain_with_matplotlib_colormap(
    width: int, 
    height: int, 
    colormap_name: str,
    normalization: str = 'linear'
) -> Tuple[np.ndarray, dict]:
    """
    Render terrain using matplotlib colormap integration.
    
    Args:
        width: Render width
        height: Render height
        colormap_name: Matplotlib colormap name
        normalization: Normalization type
        
    Returns:
        Tuple of (RGBA array, metadata dict)
    """
    print(f"\n=== Rendering Terrain ({width}x{height}) ===")
    print(f"  Colormap: {colormap_name}")
    print(f"  Normalization: {normalization}")
    
    metadata = {
        'render_size': (width, height),
        'colormap': colormap_name, 
        'normalization': normalization,
        'timings': {}
    }
    
    # Create renderer
    start_time = time.perf_counter()
    renderer = f3d.Renderer(width, height)
    metadata['timings']['create_renderer'] = time.perf_counter() - start_time
    
    # Create test heightmap
    terrain_size = min(width, height) // 2  # Use reasonable terrain resolution
    start_time = time.perf_counter()
    heightmap = create_test_heightmap(terrain_size, terrain_size)
    metadata['timings']['create_heightmap'] = time.perf_counter() - start_time
    
    print(f"  Heightmap size: {heightmap.shape}")
    print(f"  Heightmap range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
    
    # Apply normalization if requested
    if normalization != 'linear':
        norm_results = demonstrate_normalization(heightmap, normalization)
        if 'normalized_data' in norm_results:
            heightmap = norm_results['normalized_data'].astype(np.float32)
            metadata['normalization_results'] = norm_results
    
    # Add terrain with matplotlib colormap
    start_time = time.perf_counter()
    if is_matplotlib_available():
        # Convert matplotlib colormap for use with forge3d
        try:
            # Use forge3d's built-in colormap support, but demonstrate 
            # that we could use matplotlib colormaps
            renderer.add_terrain(heightmap, (2.0, 2.0), 1.0, colormap_name)
            print(f"  * Using built-in colormap: {colormap_name}")
        except Exception as e:
            print(f"  ! Built-in colormap failed, trying matplotlib conversion: {e}")
            # Fallback: convert matplotlib colormap and use a different approach
            # This would require extending the renderer API
            renderer.add_terrain(heightmap, (2.0, 2.0), 1.0, 'viridis')
            print(f"  * Using fallback colormap: viridis")
    else:
        # Fallback when matplotlib not available
        renderer.add_terrain(heightmap, (2.0, 2.0), 1.0, 'viridis') 
        print(f"  * Using fallback colormap: viridis (matplotlib unavailable)")
        
    metadata['timings']['add_terrain'] = time.perf_counter() - start_time
    
    # Upload height texture to GPU
    start_time = time.perf_counter()
    renderer.upload_height_r32f()
    metadata['timings']['upload_height'] = time.perf_counter() - start_time
    
    # Render
    start_time = time.perf_counter()
    rgba = renderer.render_terrain_rgba()
    metadata['timings']['render'] = time.perf_counter() - start_time
    
    print(f"  Output shape: {rgba.shape}")
    print(f"  Output dtype: {rgba.dtype}")
    
    total_time = sum(metadata['timings'].values())
    metadata['timings']['total'] = total_time
    print(f"  Total time: {total_time*1000:.1f} ms")
    
    return rgba, metadata


def save_results(rgba: np.ndarray, metadata: dict, output_path: str):
    """Save rendered results and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    print(f"\nSaving results to {output_path}")
    f3d.numpy_to_png(str(output_path), rgba)
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    try:
        metadata['environment'] = f3d.device_probe()
    except AttributeError:
        metadata['environment'] = {'note': 'environment info not available'}
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate matplotlib colormap integration with forge3d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --colormap viridis --size 512x512
  %(prog)s --colormap plasma --normalization log --out reports/plasma_log.png  
  %(prog)s --list-colormaps
        """
    )
    
    parser.add_argument('--colormap', '-c', default='viridis',
                        help='Matplotlib colormap name (default: viridis)')
    parser.add_argument('--normalization', '-n', choices=['linear', 'log'], 
                        default='linear',
                        help='Data normalization type (default: linear)')
    parser.add_argument('--size', type=parse_size, default=(512, 512),
                        help='Render size as WIDTHxHEIGHT (default: 512x512)')
    parser.add_argument('--out', '-o', 
                        help='Output PNG path (default: mpl_cmap_demo.png)')
    parser.add_argument('--list-colormaps', action='store_true',
                        help='List available matplotlib colormaps and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    print("=== Matplotlib Colormap Integration Demo ===")
    print(f"forge3d version: {f3d.__version__}")
    print(f"matplotlib available: {is_matplotlib_available()}")
    
    if is_matplotlib_available():
        import matplotlib
        print(f"matplotlib version: {matplotlib.__version__}")
    
    # List colormaps if requested
    if args.list_colormaps:
        if is_matplotlib_available():
            colormaps = get_matplotlib_colormap_names()
            print(f"\nAvailable matplotlib colormaps ({len(colormaps)}):")
            for i, name in enumerate(colormaps):
                print(f"  {name}", end='')
                if (i + 1) % 4 == 0:
                    print()  # New line every 4 items
            if len(colormaps) % 4 != 0:
                print()  # Final newline if needed
        else:
            print("matplotlib not available - cannot list colormaps")
        return
    
    width, height = args.size
    output_path = args.out or "mpl_cmap_demo.png"
    
    print(f"Render size: {width}x{height}")
    print(f"Output path: {output_path}")
    
    # Demonstrate colormap conversion
    colormap_results = demonstrate_colormap_conversion(args.colormap)
    
    # Render terrain
    try:
        rgba, metadata = render_terrain_with_matplotlib_colormap(
            width, height, args.colormap, args.normalization
        )
        
        metadata['colormap_conversion'] = colormap_results
        
        # Save results
        save_results(rgba, metadata, output_path)
        
        print(f"\n* Demo completed successfully")
        print(f"* Output saved to {output_path}")
        
    except Exception as e:
        print(f"\n* Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
