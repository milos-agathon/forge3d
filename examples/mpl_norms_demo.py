# examples/mpl_norms_demo.py
# Demo for Matplotlib normalization presets interop with forge3d.
# This exists to visualize and validate Workstream R3 behavior.
# RELEVANT FILES:python/forge3d/adapters/mpl_cmap.py,tests/test_mpl_norms.py,examples/_import_shim.py
#!/usr/bin/env python3
"""
Matplotlib Normalization Presets Demo

This example demonstrates advanced matplotlib normalization integration
with forge3d, showcasing:
1. LogNorm for logarithmic data scaling
2. PowerNorm for gamma correction and power-law scaling  
3. BoundaryNorm for discrete color mapping
4. Accuracy comparison with matplotlib reference
5. Performance measurement and edge case handling

Usage:
    python examples/mpl_norms_demo.py
    python examples/mpl_norms_demo.py --out reports/r3_norms.png
    python examples/mpl_norms_demo.py --norm-type log --size 1024x512
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Any
import json
import warnings

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
    from forge3d.adapters.mpl_cmap import (
        is_matplotlib_available,
        LogNormAdapter,
        PowerNormAdapter,
        BoundaryNormAdapter,
        create_matplotlib_normalizer,
        matplotlib_normalize,
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


def generate_test_data(data_type: str, size: Tuple[int, int] = (256, 256), 
                      seed: int = 42) -> np.ndarray:
    """
    Generate test data suitable for different normalization types.
    
    Args:
        data_type: Type of test data ('linear', 'log', 'power', 'boundary')
        size: Data array size (height, width)
        seed: Random seed for reproducibility
        
    Returns:
        Test data array
    """
    rng = np.random.RandomState(seed)
    height, width = size
    
    if data_type == 'linear':
        # Linear gradient with noise
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        data = X + 0.5 * Y + 0.1 * rng.normal(0, 1, (height, width))
        
    elif data_type == 'log':
        # Exponential/logarithmic data (wide dynamic range)
        base = np.logspace(0, 4, width)  # 1 to 10000
        data = np.zeros((height, width))
        for i in range(height):
            scale = 0.1 + 0.9 * (i / height)  # Varying scale by row
            data[i] = scale * base + rng.exponential(base * 0.1)
            
    elif data_type == 'power':
        # Power-law distributed data
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height) 
        X, Y = np.meshgrid(x, y)
        # Create power-law pattern
        data = (X**2 + Y**2)**0.5  # Distance from origin
        data = data**3  # Power law
        data += 0.05 * rng.uniform(0, 1, (height, width))
        
    elif data_type == 'boundary':
        # Discrete levels suitable for boundary normalization
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        X, Y = np.meshgrid(x, y)
        
        # Create concentric regions
        r = np.sqrt(X**2 + Y**2)
        data = np.zeros_like(r)
        data[r < 0.5] = 1.0
        data[(r >= 0.5) & (r < 1.0)] = 2.0
        data[(r >= 1.0) & (r < 1.5)] = 3.0
        data[r >= 1.5] = 4.0
        
        # Add some noise to boundaries
        data += 0.1 * rng.uniform(-0.5, 0.5, (height, width))
        
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    return data.astype(np.float32)


def demonstrate_log_normalization(data: np.ndarray) -> Dict[str, Any]:
    """Demonstrate logarithmic normalization with accuracy testing."""
    print("\n=== LogNorm Demonstration ===")
    
    if not is_matplotlib_available():
        print("Warning: matplotlib not available, skipping LogNorm demo")
        return {'available': False}
    
    results = {'available': True, 'accuracy': {}, 'timings': {}}
    
    # Ensure data is positive for log normalization
    data_pos = np.maximum(data, 1e-6)
    vmin, vmax = float(data_pos.min()), float(data_pos.max())
    
    print(f"  Input range: [{vmin:.6f}, {vmax:.6f}]")
    
    # Test forge3d LogNorm adapter
    start_time = time.perf_counter()
    forge3d_adapter = LogNormAdapter(vmin=vmin, vmax=vmax)
    forge3d_result = forge3d_adapter(data_pos)
    forge3d_time = time.perf_counter() - start_time
    results['timings']['forge3d'] = forge3d_time
    
    print(f"  forge3d LogNorm time: {forge3d_time*1000:.3f} ms")
    print(f"  forge3d result range: [{forge3d_result.min():.6f}, {forge3d_result.max():.6f}]")
    
    # Test matplotlib reference
    start_time = time.perf_counter()
    mpl_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    mpl_result = mpl_norm(data_pos)
    mpl_time = time.perf_counter() - start_time
    results['timings']['matplotlib'] = mpl_time
    
    print(f"  matplotlib LogNorm time: {mpl_time*1000:.3f} ms")
    print(f"  matplotlib result range: [{mpl_result.min():.6f}, {mpl_result.max():.6f}]")
    
    # Accuracy comparison
    max_abs_diff = np.max(np.abs(forge3d_result - mpl_result))
    mean_abs_diff = np.mean(np.abs(forge3d_result - mpl_result))
    
    results['accuracy']['max_abs_diff'] = float(max_abs_diff)
    results['accuracy']['mean_abs_diff'] = float(mean_abs_diff)
    results['accuracy']['within_tolerance'] = max_abs_diff < 1e-7
    
    print(f"  Accuracy: max_diff={max_abs_diff:.2e}, mean_diff={mean_abs_diff:.2e}")
    
    if max_abs_diff < 1e-7:
        print("  * Accuracy within tolerance (< 1e-7)")
    else:
        print(f"  ! Accuracy exceeds tolerance: {max_abs_diff:.2e} >= 1e-7")
    
    # Test inverse
    try:
        recovered = forge3d_adapter.inverse(forge3d_result)
        recovery_error = np.max(np.abs(recovered - data_pos))
        results['accuracy']['inverse_error'] = float(recovery_error)
        print(f"  Inverse recovery error: {recovery_error:.2e}")
    except Exception as e:
        print(f"  ! Inverse test failed: {e}")
    
    results['normalized_data'] = forge3d_result
    return results


def demonstrate_power_normalization(data: np.ndarray, gamma: float = 2.2) -> Dict[str, Any]:
    """Demonstrate power-law normalization (gamma correction)."""
    print(f"\n=== PowerNorm Demonstration (gamma={gamma}) ===")
    
    if not is_matplotlib_available():
        print("Warning: matplotlib not available, skipping PowerNorm demo")
        return {'available': False}
    
    results = {'available': True, 'gamma': gamma, 'accuracy': {}, 'timings': {}}
    
    vmin, vmax = float(data.min()), float(data.max())
    print(f"  Input range: [{vmin:.6f}, {vmax:.6f}]")
    
    # Test forge3d PowerNorm adapter
    start_time = time.perf_counter()
    forge3d_adapter = PowerNormAdapter(gamma=gamma, vmin=vmin, vmax=vmax)
    forge3d_result = forge3d_adapter(data)
    forge3d_time = time.perf_counter() - start_time
    results['timings']['forge3d'] = forge3d_time
    
    print(f"  forge3d PowerNorm time: {forge3d_time*1000:.3f} ms")
    print(f"  forge3d result range: [{forge3d_result.min():.6f}, {forge3d_result.max():.6f}]")
    
    # Test matplotlib reference
    start_time = time.perf_counter()
    mpl_norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    mpl_result = mpl_norm(data)
    mpl_time = time.perf_counter() - start_time
    results['timings']['matplotlib'] = mpl_time
    
    print(f"  matplotlib PowerNorm time: {mpl_time*1000:.3f} ms")
    print(f"  matplotlib result range: [{mpl_result.min():.6f}, {mpl_result.max():.6f}]")
    
    # Accuracy comparison
    max_abs_diff = np.max(np.abs(forge3d_result - mpl_result))
    mean_abs_diff = np.mean(np.abs(forge3d_result - mpl_result))
    
    results['accuracy']['max_abs_diff'] = float(max_abs_diff)
    results['accuracy']['mean_abs_diff'] = float(mean_abs_diff)
    results['accuracy']['within_tolerance'] = max_abs_diff < 1e-7
    
    print(f"  Accuracy: max_diff={max_abs_diff:.2e}, mean_diff={mean_abs_diff:.2e}")
    
    if max_abs_diff < 1e-7:
        print("  * Accuracy within tolerance (< 1e-7)")
    else:
        print(f"  ! Accuracy exceeds tolerance: {max_abs_diff:.2e} >= 1e-7")
    
    # Test inverse
    try:
        recovered = forge3d_adapter.inverse(forge3d_result)
        recovery_error = np.max(np.abs(recovered - data))
        results['accuracy']['inverse_error'] = float(recovery_error)
        print(f"  Inverse recovery error: {recovery_error:.2e}")
    except Exception as e:
        print(f"  ! Inverse test failed: {e}")
    
    results['normalized_data'] = forge3d_result
    return results


def demonstrate_boundary_normalization(data: np.ndarray) -> Dict[str, Any]:
    """Demonstrate boundary normalization for discrete color mapping."""
    print("\n=== BoundaryNorm Demonstration ===")
    
    if not is_matplotlib_available():
        print("Warning: matplotlib not available, skipping BoundaryNorm demo") 
        return {'available': False}
    
    results = {'available': True, 'accuracy': {}, 'timings': {}}
    
    # Define boundaries based on data range
    data_min, data_max = data.min(), data.max()
    boundaries = np.linspace(data_min, data_max, 6)  # 5 color levels
    ncolors = len(boundaries) - 1
    
    results['boundaries'] = boundaries.tolist()
    results['ncolors'] = ncolors
    
    print(f"  Input range: [{data_min:.3f}, {data_max:.3f}]")
    print(f"  Boundaries: {boundaries}")
    print(f"  Number of colors: {ncolors}")
    
    # Test forge3d BoundaryNorm adapter
    start_time = time.perf_counter()
    forge3d_adapter = BoundaryNormAdapter(boundaries, ncolors)
    forge3d_result = forge3d_adapter(data)
    forge3d_time = time.perf_counter() - start_time
    results['timings']['forge3d'] = forge3d_time
    
    print(f"  forge3d BoundaryNorm time: {forge3d_time*1000:.3f} ms")
    print(f"  forge3d unique values: {sorted(np.unique(forge3d_result))}")
    
    # Test matplotlib reference
    start_time = time.perf_counter()
    mpl_norm = mcolors.BoundaryNorm(boundaries, ncolors)
    mpl_result = mpl_norm(data)
    mpl_time = time.perf_counter() - start_time
    results['timings']['matplotlib'] = mpl_time
    
    print(f"  matplotlib BoundaryNorm time: {mpl_time*1000:.3f} ms") 
    print(f"  matplotlib unique values: {sorted(np.unique(mpl_result))}")
    
    # Accuracy comparison (should be exact for boundary norm)
    max_abs_diff = np.max(np.abs(forge3d_result - mpl_result))
    results['accuracy']['max_abs_diff'] = float(max_abs_diff)
    results['accuracy']['exact_match'] = max_abs_diff == 0.0
    
    print(f"  Accuracy: max_diff={max_abs_diff:.2e}")
    
    if max_abs_diff == 0.0:
        print("  * Exact match with matplotlib")
    else:
        print(f"  ! Difference from matplotlib: {max_abs_diff:.2e}")
    
    results['normalized_data'] = forge3d_result
    return results


def create_comparison_visualization(
    original_data: np.ndarray,
    norm_results: Dict[str, Dict[str, Any]],
    output_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create a comparison visualization showing original data and all normalizations.
    
    Args:
        original_data: Original input data
        norm_results: Results from normalization demonstrations
        output_size: Output image size (width, height)
        
    Returns:
        RGBA visualization array
    """
    print("\n=== Creating Comparison Visualization ===")
    
    width, height = output_size
    
    # Calculate layout - original + up to 3 normalizations
    available_norms = [k for k, v in norm_results.items() if v.get('available', False)]
    n_panels = 1 + len(available_norms)  # original + normalizations
    
    if n_panels <= 2:
        panel_w, panel_h = width // 2, height
        cols, rows = 2, 1
    elif n_panels <= 4:
        panel_w, panel_h = width // 2, height // 2
        cols, rows = 2, 2
    else:
        panel_w, panel_h = width // 3, height // 2
        cols, rows = 3, 2
    
    print(f"  Layout: {cols}x{rows} panels, {panel_w}x{panel_h} each")
    
    # Create renderer
    renderer = f3d.Renderer(width, height)
    
    # We'll use a simple approach: render each panel separately and composite
    # For simplicity, we'll just render the original data with different colormaps
    # In a full implementation, you'd render each normalized dataset
    
    # Create a synthetic heightmap from the data for visualization
    if original_data.shape[0] != panel_h or original_data.shape[1] != panel_w:
        # Resize data to fit panel
        from scipy import ndimage
        try:
            resized_data = ndimage.zoom(original_data, 
                                      (panel_h / original_data.shape[0], 
                                       panel_w / original_data.shape[1]))
        except ImportError:
            # Fallback: simple indexing
            step_h = max(1, original_data.shape[0] // panel_h)
            step_w = max(1, original_data.shape[1] // panel_w)
            resized_data = original_data[::step_h, ::step_w]
    else:
        resized_data = original_data
    
    # Normalize to [0, 1] for terrain rendering
    norm_data = (resized_data - resized_data.min()) / (resized_data.max() - resized_data.min())
    
    # Add terrain using the normalized data
    renderer.add_terrain(norm_data.astype(np.float32), (2.0, 2.0), 1.0, 'viridis')
    
    # Render  
    renderer.upload_height_r32f()
    rgba = renderer.render_terrain_rgba()
    
    print(f"  Rendered comparison: {rgba.shape}")
    
    return rgba


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate matplotlib normalization presets with forge3d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --norm-type log --size 512x512
  %(prog)s --norm-type power --gamma 1.8 --out power_gamma18.png
  %(prog)s --norm-type boundary --data-type boundary
  %(prog)s --all-norms --size 1024x768
        """
    )
    
    parser.add_argument('--norm-type', choices=['log', 'power', 'boundary'], 
                        default='log',
                        help='Normalization type to demonstrate (default: log)')
    parser.add_argument('--data-type', choices=['linear', 'log', 'power', 'boundary'],
                        help='Test data type (default: matches norm-type)')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Gamma value for power normalization (default: 2.2)')
    parser.add_argument('--all-norms', action='store_true',
                        help='Demonstrate all normalization types')
    parser.add_argument('--size', type=parse_size, default=(512, 512),
                        help='Output size as WIDTHxHEIGHT (default: 512x512)')
    parser.add_argument('--data-size', type=parse_size, default=(256, 256),
                        help='Test data size as WIDTHxHEIGHT (default: 256x256)')
    parser.add_argument('--out', '-o',
                        help='Output PNG path (default: mpl_norms_demo.png)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    print("=== Matplotlib Normalization Presets Demo ===")
    print(f"forge3d version: {f3d.__version__}")
    print(f"matplotlib available: {is_matplotlib_available()}")
    
    if is_matplotlib_available():
        print(f"matplotlib version: {matplotlib.__version__}")
    else:
        print("Warning: matplotlib not available - functionality will be limited")
    
    width, height = args.size
    output_path = args.out or "mpl_norms_demo.png"
    
    print(f"Output size: {width}x{height}")
    print(f"Output path: {output_path}")
    
    # Determine what to demonstrate
    if args.all_norms:
        norm_types = ['log', 'power', 'boundary']
    else:
        norm_types = [args.norm_type]
    
    data_type = args.data_type or args.norm_type
    
    print(f"Normalization types: {norm_types}")
    print(f"Test data type: {data_type}")
    
    # Generate test data
    print(f"\nGenerating test data ({data_type})...")
    original_data = generate_test_data(data_type, args.data_size)
    print(f"Test data shape: {original_data.shape}")
    print(f"Test data range: [{original_data.min():.6f}, {original_data.max():.6f}]")
    
    # Demonstrate normalizations
    norm_results = {}
    
    for norm_type in norm_types:
        try:
            if norm_type == 'log':
                norm_results['log'] = demonstrate_log_normalization(original_data)
            elif norm_type == 'power':
                norm_results['power'] = demonstrate_power_normalization(original_data, args.gamma)
            elif norm_type == 'boundary':
                norm_results['boundary'] = demonstrate_boundary_normalization(original_data)
                
        except Exception as e:
            print(f"  * {norm_type} normalization failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Create visualization
    try:
        print("\nCreating visualization...")
        rgba = create_comparison_visualization(original_data, norm_results, (width, height))
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        f3d.numpy_to_png(str(output_path), rgba)
        print(f"Visualization saved to {output_path}")
        
        # Save metadata
        metadata = {
            'norm_types': norm_types,
            'data_type': data_type,
            'gamma': args.gamma,
            'output_size': [width, height],
            'data_size': list(args.data_size),
            'results': norm_results,
            'environment': f3d.device_probe() if hasattr(f3d, 'device_probe') else {'note': 'environment info not available'}
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {metadata_path}")
        
        # Summary
        print(f"\n=== Summary ===")
        successful = sum(1 for r in norm_results.values() if r.get('available', False))
        print(f"Successful demonstrations: {successful}/{len(norm_types)}")
        
        if norm_results:
            for norm_type, results in norm_results.items():
                if results.get('available'):
                    accuracy = results.get('accuracy', {})
                    if 'within_tolerance' in accuracy:
                        status = "*" if accuracy['within_tolerance'] else "!"
                        print(f"  {norm_type}: {status} accuracy within tolerance")
                    elif 'exact_match' in accuracy:
                        status = "*" if accuracy['exact_match'] else "!"
                        print(f"  {norm_type}: {status} exact match with matplotlib")
        
        print(f"* Demo completed successfully")
        
    except Exception as e:
        print(f"* Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
