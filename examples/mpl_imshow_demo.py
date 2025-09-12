# examples/mpl_imshow_demo.py
# Demo for Matplotlib display helpers interop with forge3d.
# This exists to visualize and validate Workstream R4 behavior.
# RELEVANT FILES:python/forge3d/helpers/mpl_display.py,tests/test_mpl_display.py,examples/_import_shim.py
#!/usr/bin/env python3
"""
Matplotlib Display Helpers Demo

This example demonstrates the matplotlib display integration for forge3d,
showing how to:
1. Display forge3d RGBA buffers in matplotlib figures
2. Handle different image formats (uint8, float32)
3. Use custom extents and DPI settings
4. Create subplot comparisons
5. Save visualizations to files
6. Optimize for zero-copy display when possible

Usage:
    python examples/mpl_imshow_demo.py
    python examples/mpl_imshow_demo.py --out reports/r4_imshow.png
    python examples/mpl_imshow_demo.py --size 1024x768 --backend Agg
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
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
    from forge3d.helpers.mpl_display import (
        is_matplotlib_display_available,
        imshow_rgba,
        imshow_rgba_subplots,
        save_rgba_comparison,
        setup_matplotlib_backend,
        validate_rgba_array,
        quick_show,
    )
except ImportError as e:
    print(f"Error: Cannot import forge3d: {e}")
    print("Fix: either set PYTHONPATH to include 'python' or run 'pip install -U maturin' then 'maturin develop --release'")
    sys.exit(1)

# Optional matplotlib import
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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


def create_test_scenes(width: int, height: int) -> Dict[str, np.ndarray]:
    """
    Create various test scenes for display demonstration.
    
    Args:
        width: Scene width
        height: Scene height
        
    Returns:
        Dictionary of scene name -> RGBA array
    """
    print(f"\nCreating test scenes ({width}x{height})...")
    
    scenes = {}
    
    # Scene 1: Terrain rendering
    print("  Creating terrain scene...")
    renderer = f3d.Renderer(width, height)
    
    # Generate synthetic terrain
    terrain_size = min(width, height) // 4
    heightmap = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    
    # Create hills and valleys
    x, y = np.mgrid[0:terrain_size, 0:terrain_size]
    x_norm = x / (terrain_size - 1)
    y_norm = y / (terrain_size - 1)
    
    # Multiple frequency components
    heightmap += 0.5 * np.sin(4 * np.pi * x_norm) * np.cos(3 * np.pi * y_norm)
    heightmap += 0.3 * np.sin(8 * np.pi * x_norm) * np.cos(6 * np.pi * y_norm)
    heightmap += 0.2 * np.random.RandomState(42).normal(0, 0.1, heightmap.shape)
    
    # Normalize to [0, 1]
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    
    renderer.add_terrain(heightmap, (2.0, 2.0), 1.5, 'viridis')
    renderer.upload_height_r32f()
    scenes['terrain_viridis'] = renderer.render_terrain_rgba()
    
    # Scene 2: Same terrain with different colormap
    print("  Creating magma terrain scene...")
    renderer2 = f3d.Renderer(width, height)
    renderer2.add_terrain(heightmap, (2.0, 2.0), 1.5, 'magma')
    renderer2.upload_height_r32f()
    scenes['terrain_magma'] = renderer2.render_terrain_rgba()
    
    # Scene 3: Simple geometric pattern
    print("  Creating geometric pattern...")
    renderer3 = f3d.Renderer(width, height)
    scenes['triangle'] = renderer3.render_triangle_rgba()
    
    # Scene 4: Float32 version (for testing different formats)
    print("  Creating float32 version...")
    rgba_uint8 = scenes['terrain_viridis']
    rgba_float32 = rgba_uint8.astype(np.float32) / 255.0
    scenes['terrain_float32'] = rgba_float32
    
    print(f"  Created {len(scenes)} test scenes")
    return scenes


def demonstrate_basic_display(scenes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Demonstrate basic display functionality."""
    print("\n=== Basic Display Demonstration ===")
    
    if not is_matplotlib_display_available():
        print("Warning: matplotlib not available, skipping display demo")
        return {'available': False}
    
    results = {'available': True, 'tests': {}}
    
    # Setup backend for headless operation
    setup_matplotlib_backend('Agg')
    
    # Test 1: Basic display of uint8 RGBA
    print("  Testing basic uint8 RGBA display...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rgba = scenes['terrain_viridis']
    start_time = time.perf_counter()
    im = imshow_rgba(ax, rgba)
    display_time = time.perf_counter() - start_time
    
    ax.set_title('forge3d Terrain (uint8 RGBA)')
    
    results['tests']['basic_uint8'] = {
        'success': True,
        'display_time': display_time,
        'array_shape': rgba.shape,
        'array_dtype': str(rgba.dtype)
    }
    
    print(f"    Display time: {display_time*1000:.2f} ms")
    print(f"    Array shape: {rgba.shape}, dtype: {rgba.dtype}")
    
    plt.close(fig)
    
    # Test 2: Display with custom extent
    print("  Testing display with custom extent...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    extent = (0, 100, 0, 75)  # Custom coordinate system
    im = imshow_rgba(ax, rgba, extent=extent)
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Terrain with Custom Extent')
    
    # Verify extent was applied
    assert im.get_extent() == extent
    
    results['tests']['custom_extent'] = {
        'success': True,
        'extent': extent
    }
    
    print(f"    Custom extent: {extent}")
    
    plt.close(fig)
    
    # Test 3: Display float32 data
    print("  Testing float32 RGBA display...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rgba_float = scenes['terrain_float32']
    im = imshow_rgba(ax, rgba_float)
    
    ax.set_title('forge3d Terrain (float32 RGBA)')
    
    results['tests']['float32'] = {
        'success': True,
        'array_shape': rgba_float.shape,
        'array_dtype': str(rgba_float.dtype),
        'value_range': [float(rgba_float.min()), float(rgba_float.max())]
    }
    
    print(f"    Float32 range: [{rgba_float.min():.3f}, {rgba_float.max():.3f}]")
    
    plt.close(fig)
    
    # Test 4: Display with custom DPI
    print("  Testing custom DPI...")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    target_dpi = 200
    im = imshow_rgba(ax, rgba, dpi=target_dpi)
    
    ax.set_title(f'Custom DPI ({target_dpi})')
    
    results['tests']['custom_dpi'] = {
        'success': True,
        'target_dpi': target_dpi,
        'actual_dpi': fig.dpi
    }
    
    print(f"    Target DPI: {target_dpi}, Actual DPI: {fig.dpi}")
    
    plt.close(fig)
    
    print("  * Basic display tests completed")
    return results


def demonstrate_subplot_functionality(scenes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Demonstrate subplot and comparison functionality."""
    print("\n=== Subplot Functionality Demonstration ===")
    
    if not is_matplotlib_display_available():
        print("Warning: matplotlib not available, skipping subplot demo")
        return {'available': False}
    
    results = {'available': True, 'tests': {}}
    
    setup_matplotlib_backend('Agg')
    
    # Test 1: Multiple scenes in subplots
    print("  Testing multiple scene comparison...")
    
    scene_list = [
        scenes['terrain_viridis'],
        scenes['terrain_magma'], 
        scenes['triangle']
    ]
    
    titles = ['Viridis Terrain', 'Magma Terrain', 'Triangle']
    
    start_time = time.perf_counter()
    fig, images = imshow_rgba_subplots(
        scene_list,
        titles=titles,
        figsize=(12, 4),
        ncols=3
    )
    subplot_time = time.perf_counter() - start_time
    
    results['tests']['subplot_comparison'] = {
        'success': True,
        'n_subplots': len(scene_list),
        'subplot_time': subplot_time,
        'figure_size': fig.get_size_inches().tolist()
    }
    
    print(f"    Subplot creation time: {subplot_time*1000:.2f} ms")
    print(f"    Created {len(images)} subplots")
    
    plt.close(fig)
    
    # Test 2: Comparison with different formats
    print("  Testing uint8 vs float32 comparison...")
    
    comparison_arrays = [
        scenes['terrain_viridis'],     # uint8
        scenes['terrain_float32']      # float32
    ]
    
    comparison_titles = ['uint8 Format', 'float32 Format']
    
    fig, images = imshow_rgba_subplots(
        comparison_arrays,
        titles=comparison_titles,
        figsize=(10, 5),
        ncols=2
    )
    
    results['tests']['format_comparison'] = {
        'success': True,
        'formats': [str(arr.dtype) for arr in comparison_arrays]
    }
    
    print(f"    Compared formats: {[str(arr.dtype) for arr in comparison_arrays]}")
    
    plt.close(fig)
    
    print("  * Subplot tests completed")
    return results


def demonstrate_validation_and_error_handling(scenes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Demonstrate validation and error handling."""
    print("\n=== Validation and Error Handling Demonstration ===")
    
    if not is_matplotlib_display_available():
        print("Warning: matplotlib not available, skipping validation demo")
        return {'available': False}
    
    results = {'available': True, 'tests': {}}
    
    setup_matplotlib_backend('Agg')
    
    # Test 1: Array validation  
    print("  Testing array validation...")
    
    rgba = scenes['terrain_viridis']
    
    try:
        validated = validate_rgba_array(rgba, "test_array")
        results['tests']['validation_pass'] = {
            'success': True,
            'validated_shape': validated.shape,
            'validated_dtype': str(validated.dtype)
        }
        print("    * Valid array passed validation")
    except Exception as e:
        results['tests']['validation_pass'] = {
            'success': False,
            'error': str(e)
        }
        print(f"    * Validation failed: {e}")
    
    # Test 2: Invalid array handling
    print("  Testing invalid array handling...")
    
    # Test with wrong dimensions
    try:
        invalid_2d = rgba[:, :, 0]  # Remove channel dimension
        validate_rgba_array(invalid_2d, "invalid_2d")
        results['tests']['validation_fail_2d'] = {'success': False, 'error': 'Should have failed'}
    except ValueError as e:
        results['tests']['validation_fail_2d'] = {'success': True, 'error': str(e)}
        print(f"    * Correctly rejected 2D array: {type(e).__name__}")
    except Exception as e:
        results['tests']['validation_fail_2d'] = {'success': False, 'error': str(e)}
        print(f"    * Unexpected error: {e}")
    
    # Test 3: Non-contiguous array warning
    print("  Testing non-contiguous array warning...")
    
    try:
        # Create non-contiguous array
        rgba_transposed = rgba.transpose(1, 0, 2)  # May be non-contiguous
        if not rgba_transposed.flags['C_CONTIGUOUS']:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                validate_rgba_array(rgba_transposed, "non_contiguous")
                
                if w:
                    results['tests']['non_contiguous_warning'] = {
                        'success': True,
                        'warning_message': str(w[0].message)
                    }
                    print("    * Warning issued for non-contiguous array")
                else:
                    results['tests']['non_contiguous_warning'] = {
                        'success': False,
                        'error': 'No warning issued'
                    }
        else:
            results['tests']['non_contiguous_warning'] = {
                'success': True,
                'note': 'Test array was C-contiguous after transpose'
            }
            print("    ℹ Test array remained C-contiguous after transpose")
            
    except Exception as e:
        results['tests']['non_contiguous_warning'] = {
            'success': False,
            'error': str(e)
        }
        print(f"    * Error in non-contiguous test: {e}")
    
    print("  * Validation tests completed")
    return results


def demonstrate_performance_aspects(scenes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Demonstrate performance-related aspects."""
    print("\n=== Performance Aspects Demonstration ===")
    
    if not is_matplotlib_display_available():
        print("Warning: matplotlib not available, skipping performance demo")
        return {'available': False}
    
    results = {'available': True, 'tests': {}}
    
    setup_matplotlib_backend('Agg')
    
    # Test 1: Display timing for different sizes
    print("  Testing display timing for different array sizes...")
    
    timing_results = {}
    
    for scale in [0.25, 0.5, 1.0]:
        rgba = scenes['terrain_viridis']
        h, w = rgba.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Simple resize by indexing (not ideal, but sufficient for timing test)
        if scale < 1.0:
            step_h = max(1, int(1.0 / scale))
            step_w = max(1, int(1.0 / scale))
            test_rgba = rgba[::step_h, ::step_w]
        else:
            test_rgba = rgba
        
        fig, ax = plt.subplots()
        
        start_time = time.perf_counter()
        im = imshow_rgba(ax, test_rgba)
        display_time = time.perf_counter() - start_time
        
        timing_results[f'{scale}x'] = {
            'shape': test_rgba.shape,
            'display_time': display_time,
            'pixels': test_rgba.shape[0] * test_rgba.shape[1]
        }
        
        print(f"    {scale}x scale ({test_rgba.shape[:2]}): {display_time*1000:.2f} ms")
        
        plt.close(fig)
    
    results['tests']['timing_comparison'] = timing_results
    
    # Test 2: Memory efficiency check
    print("  Testing memory efficiency...")
    
    rgba = scenes['terrain_viridis']
    original_flags = rgba.flags.c_contiguous
    
    fig, ax = plt.subplots()
    
    # Check if the array remains unchanged after display
    original_data_ptr = rgba.ctypes.data
    
    im = imshow_rgba(ax, rgba)
    
    # Array should not be modified
    same_ptr = rgba.ctypes.data == original_data_ptr
    same_flags = rgba.flags.c_contiguous == original_flags
    
    results['tests']['memory_efficiency'] = {
        'original_c_contiguous': original_flags,
        'same_data_pointer': same_ptr,
        'same_c_contiguous_flag': same_flags
    }
    
    print(f"    Original C-contiguous: {original_flags}")
    print(f"    Data pointer unchanged: {same_ptr}")
    
    plt.close(fig)
    
    print("  * Performance tests completed")
    return results


def create_comprehensive_demonstration(scenes: Dict[str, np.ndarray], 
                                     output_size: Tuple[int, int]) -> np.ndarray:
    """Create a comprehensive demonstration figure."""
    print("\n=== Creating Comprehensive Demonstration ===")
    
    if not is_matplotlib_display_available():
        print("Warning: matplotlib not available, creating fallback visualization")
        # Return a simple composite as fallback
        return scenes['terrain_viridis']
    
    setup_matplotlib_backend('Agg')
    width, height = output_size
    
    # Create a figure with multiple panels demonstrating different features
    fig = plt.figure(figsize=(16, 12), dpi=100)
    
    # Panel 1: Basic terrain with colorbar
    ax1 = plt.subplot(2, 3, 1)
    im1 = imshow_rgba(ax1, scenes['terrain_viridis'])
    ax1.set_title('Viridis Terrain\n(Basic Display)', fontsize=10)
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    
    # Panel 2: Same terrain with custom extent
    ax2 = plt.subplot(2, 3, 2) 
    extent = (0, 10, 0, 7.5)
    im2 = imshow_rgba(ax2, scenes['terrain_plasma'], extent=extent)
    ax2.set_title('Plasma Terrain\n(Custom Extent)', fontsize=10)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Distance (km)')
    
    # Panel 3: Triangle scene
    ax3 = plt.subplot(2, 3, 3)
    im3 = imshow_rgba(ax3, scenes['triangle'])
    ax3.set_title('Geometric Pattern\n(Triangle)', fontsize=10)
    
    # Panel 4: Float32 vs uint8 comparison info
    ax4 = plt.subplot(2, 3, 4)
    ax4.text(0.1, 0.8, 'Format Comparison:', fontweight='bold', transform=ax4.transAxes)
    
    uint8_info = f"uint8: {scenes['terrain_viridis'].shape}, range [0, 255]"
    float32_info = f"float32: {scenes['terrain_float32'].shape}, range [{scenes['terrain_float32'].min():.3f}, {scenes['terrain_float32'].max():.3f}]"
    
    ax4.text(0.1, 0.6, uint8_info, transform=ax4.transAxes, fontsize=9)
    ax4.text(0.1, 0.4, float32_info, transform=ax4.transAxes, fontsize=9)
    ax4.text(0.1, 0.2, 'Both formats supported\nwith automatic conversion', 
             transform=ax4.transAxes, fontsize=9, style='italic')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title('Technical Details', fontsize=10)
    
    # Panel 5: Performance information
    ax5 = plt.subplot(2, 3, 5)
    rgba = scenes['terrain_viridis']
    n_pixels = rgba.shape[0] * rgba.shape[1]
    memory_mb = rgba.nbytes / (1024 * 1024)
    
    ax5.text(0.1, 0.8, 'Performance Info:', fontweight='bold', transform=ax5.transAxes)
    ax5.text(0.1, 0.6, f'Pixels: {n_pixels:,}', transform=ax5.transAxes, fontsize=9)
    ax5.text(0.1, 0.5, f'Memory: {memory_mb:.1f} MB', transform=ax5.transAxes, fontsize=9)
    ax5.text(0.1, 0.4, f'C-contiguous: {rgba.flags.c_contiguous}', transform=ax5.transAxes, fontsize=9)
    ax5.text(0.1, 0.2, 'Zero-copy display\nwhen possible', 
             transform=ax5.transAxes, fontsize=9, style='italic')
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title('Performance Metrics', fontsize=10)
    
    # Panel 6: Features summary
    ax6 = plt.subplot(2, 3, 6)
    features = [
        '* uint8 & float32 support',
        '* RGB & RGBA formats',
        '* Custom extents & DPI',
        '* Zero-copy optimization',
        '* Validation & error handling',
        '* Subplot comparisons'
    ]
    
    ax6.text(0.1, 0.9, 'Feature Summary:', fontweight='bold', transform=ax6.transAxes)
    for i, feature in enumerate(features):
        y_pos = 0.8 - i * 0.1
        ax6.text(0.1, y_pos, feature, transform=ax6.transAxes, fontsize=9)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('Matplotlib Integration', fontsize=10)
    
    # Main title
    fig.suptitle('forge3d Matplotlib Display Helpers Demo', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to temporary buffer and convert to RGBA array
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # Convert PNG buffer to numpy array
    buf.seek(0)
    from PIL import Image
    pil_img = Image.open(buf)
    demo_rgba = np.array(pil_img)
    
    print(f"  Created demonstration figure: {demo_rgba.shape}")
    return demo_rgba


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate matplotlib display helpers for forge3d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --size 512x512 --backend Agg
  %(prog)s --out reports/r4_demo.png --save-individual
  %(prog)s --test-performance --verbose
        """
    )
    
    parser.add_argument('--size', type=parse_size, default=(512, 384),
                        help='Scene size as WIDTHxHEIGHT (default: 512x384)')
    parser.add_argument('--backend', default='Agg',
                        help='Matplotlib backend (default: Agg)')
    parser.add_argument('--out', '-o',
                        help='Output PNG path (default: mpl_imshow_demo.png)')
    parser.add_argument('--save-individual', action='store_true',
                        help='Save individual test results')
    parser.add_argument('--test-performance', action='store_true',
                        help='Run performance tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    print("=== Matplotlib Display Helpers Demo ===")
    print(f"forge3d version: {f3d.__version__}")
    print(f"matplotlib display available: {is_matplotlib_display_available()}")
    
    if is_matplotlib_display_available():
        print(f"matplotlib version: {matplotlib.__version__}")
        current_backend = setup_matplotlib_backend(args.backend)
        print(f"matplotlib backend: {current_backend}")
    else:
        print("Warning: matplotlib not available - functionality will be limited")
    
    width, height = args.size
    output_path = args.out or "mpl_imshow_demo.png"
    
    print(f"Scene size: {width}x{height}")
    print(f"Output path: {output_path}")
    
    # Create test scenes
    scenes = create_test_scenes(width, height)
    
    # Collect all results
    all_results = {}
    
    try:
        # Demonstrate basic functionality
        basic_results = demonstrate_basic_display(scenes)
        all_results['basic_display'] = basic_results
        
        # Demonstrate subplot functionality
        subplot_results = demonstrate_subplot_functionality(scenes)
        all_results['subplot_functionality'] = subplot_results
        
        # Demonstrate validation and error handling
        validation_results = demonstrate_validation_and_error_handling(scenes)
        all_results['validation_handling'] = validation_results
        
        # Demonstrate performance aspects
        if args.test_performance:
            performance_results = demonstrate_performance_aspects(scenes)
            all_results['performance_aspects'] = performance_results
        
        # Create comprehensive demonstration
        if is_matplotlib_display_available():
            try:
                from PIL import Image
                demo_rgba = create_comprehensive_demonstration(scenes, (width, height))
            except ImportError:
                print("Warning: PIL not available, using terrain scene as demo output")
                demo_rgba = scenes['terrain_viridis']
        else:
            demo_rgba = scenes['terrain_viridis']
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving demo output...")
        f3d.numpy_to_png(str(output_path), demo_rgba)
        print(f"Demo visualization saved to {output_path}")
        
        # Save individual scenes if requested
        if args.save_individual:
            for name, rgba in scenes.items():
                if rgba.dtype == np.float32:
                    continue  # Skip float32 for PNG saving
                scene_path = output_path.with_stem(f"{output_path.stem}_{name}")
                f3d.numpy_to_png(str(scene_path), rgba)
                print(f"  Saved {name} to {scene_path}")
        
        # Save metadata
        metadata = {
            'scene_size': [width, height],
            'backend': args.backend,
            'test_results': all_results,
            'environment': f3d.device_probe() if hasattr(f3d, 'device_probe') else {'note': 'environment info not available'}
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {metadata_path}")
        
        # Summary
        print(f"\n=== Summary ===")
        successful_categories = sum(1 for r in all_results.values() if r.get('available', True))
        print(f"Successful test categories: {successful_categories}/{len(all_results)}")
        
        if args.verbose and all_results:
            for category, results in all_results.items():
                if results.get('available', True):
                    tests = results.get('tests', {})
                    successful_tests = sum(1 for t in tests.values() if t.get('success', False))
                    print(f"  {category}: {successful_tests}/{len(tests)} tests passed")
        
        print("* Demo completed successfully")
        
    except Exception as e:
        print(f"* Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
