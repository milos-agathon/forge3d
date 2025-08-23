#!/usr/bin/env python3
"""
Profile zero-copy pathways in forge3d rendering operations.

This tool renders at a specified size and measures timing while checking 
for expected zero-copy pathways between NumPy arrays and Rust backing stores.
It prints performance metrics and warnings if unexpected copies are detected.

Usage:
    python profile_copies.py --render-size 1024x1024
    python profile_copies.py --render-size 512x512 --terrain-size 256x256
    python profile_copies.py --help
"""

import argparse
import time
import sys
from typing import Tuple, Dict, Any, Optional

import numpy as np

# Import forge3d and validation helpers
try:
    import forge3d as f3d
    from forge3d._validate import ptr, is_c_contiguous, validate_zero_copy_path, check_zero_copy_compatibility
except ImportError as e:
    print(f"Error: Cannot import forge3d: {e}")
    print("Make sure forge3d is installed (run 'maturin develop --release')")
    sys.exit(1)


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


def time_operation(operation_name: str, func, *args, **kwargs):
    """Time an operation and return (result, elapsed_seconds)."""
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"  {operation_name}: {elapsed*1000:.2f} ms")
        return result, elapsed
    except Exception as e:
        end_time = time.perf_counter()
        elapsed = end_time - start_time  
        print(f"  {operation_name}: FAILED after {elapsed*1000:.2f} ms - {e}")
        raise


def profile_rgba_output(width: int, height: int) -> Dict[str, Any]:
    """Profile RGBA output zero-copy pathway."""
    print(f"\n=== Profiling RGBA Output ({width}x{height}) ===")
    
    results = {
        'width': width,
        'height': height,
        'operations': {},
        'zero_copy_status': 'unknown',
        'warnings': []
    }
    
    # Create renderer
    renderer, create_time = time_operation("Renderer creation", f3d.Renderer, width, height)
    results['operations']['create_renderer'] = create_time
    
    # Use test hook to get both array and pointer for zero-copy validation
    try:
        rgba_array, rust_ptr = renderer.render_triangle_rgba_with_ptr()
        numpy_ptr = ptr(rgba_array)
        
        print(f"  NumPy array pointer: 0x{numpy_ptr:x}")
        print(f"  Rust buffer pointer: 0x{rust_ptr:x}")
        
        # Validate zero-copy
        if numpy_ptr == rust_ptr:
            print("  RGBA zero-copy validated - pointers match")
            results['zero_copy_status'] = 'zero_copy_confirmed'
        else:
            print(f"  RGBA zero-copy failed - pointer mismatch (offset: {abs(numpy_ptr - rust_ptr)} bytes)")
            results['zero_copy_status'] = 'copy_detected'
            results['warnings'].append(f"RGBA pointer mismatch: numpy=0x{numpy_ptr:x}, rust=0x{rust_ptr:x}")
        
        # Validate output properties
        if rgba_array.shape != (height, width, 4):
            results['warnings'].append(f"Unexpected RGBA shape: {rgba_array.shape}, expected {(height, width, 4)}")
        if rgba_array.dtype != np.uint8:
            results['warnings'].append(f"Unexpected RGBA dtype: {rgba_array.dtype}, expected uint8")
        if not is_c_contiguous(rgba_array):
            results['warnings'].append("RGBA output is not C-contiguous")
            
    except AttributeError:
        # Fallback for non-test builds
        print("  Test hooks not available (not a test build)")
        rgba_array = renderer.render_triangle_rgba()
        numpy_ptr = ptr(rgba_array)
        print(f"  NumPy array pointer: 0x{numpy_ptr:x}")
        results['zero_copy_status'] = 'test_hooks_unavailable'
        results['warnings'].append("Cannot validate zero-copy without test hooks")
    
    return results


def profile_height_input(terrain_width: int, terrain_height: int, render_width: int, render_height: int) -> Dict[str, Any]:
    """Profile height input zero-copy pathway."""
    print(f"\n=== Profiling Height Input ({terrain_width}x{terrain_height} terrain, {render_width}x{render_height} render) ===")
    
    results = {
        'terrain_width': terrain_width,
        'terrain_height': terrain_height,
        'render_width': render_width, 
        'render_height': render_height,
        'operations': {},
        'zero_copy_status': 'unknown',
        'warnings': []
    }
    
    # Create renderer
    renderer = f3d.Renderer(render_width, render_height)
    
    # Create test heightmap (C-contiguous float32 for zero-copy path)
    print("  Creating test heightmap...")
    heightmap_f32 = np.random.RandomState(42).rand(terrain_height, terrain_width).astype(np.float32)
    
    # Ensure C-contiguous
    if not is_c_contiguous(heightmap_f32):
        heightmap_f32 = np.ascontiguousarray(heightmap_f32)
        results['warnings'].append("Had to make heightmap contiguous")
    
    input_ptr = ptr(heightmap_f32)
    print(f"  Input heightmap pointer: 0x{input_ptr:x}")
    
    # Check input compatibility
    compat = check_zero_copy_compatibility(heightmap_f32, "heightmap")
    if not compat['compatible']:
        results['warnings'].extend([f"Heightmap issue: {issue}" for issue in compat['issues']])
    else:
        print("  Input heightmap is zero-copy compatible")
    
    # Add terrain (should use zero-copy path for float32 C-contiguous)
    _, add_time = time_operation("add_terrain", renderer.add_terrain, 
                                heightmap_f32, (1.0, 1.0), 1.0, "viridis")
    results['operations']['add_terrain'] = add_time
    
    # Validate zero-copy for height input
    try:
        captured_ptr = renderer.debug_last_height_src_ptr()
        
        print(f"  Input heightmap pointer: 0x{input_ptr:x}")
        print(f"  Captured source pointer: 0x{captured_ptr:x}")
        
        if input_ptr == captured_ptr:
            print("  Height input zero-copy validated - pointers match")
            results['zero_copy_status'] = 'zero_copy_confirmed'
        else:
            print(f"  Height input zero-copy failed - pointer mismatch (offset: {abs(input_ptr - captured_ptr)} bytes)")
            results['zero_copy_status'] = 'copy_detected'
            results['warnings'].append(f"Height pointer mismatch: input=0x{input_ptr:x}, captured=0x{captured_ptr:x}")
            
    except AttributeError:
        # Fallback for non-test builds
        print("  Test hooks not available (not a test build)")
        results['zero_copy_status'] = 'test_hooks_unavailable'
        results['warnings'].append("Cannot validate height zero-copy without test hooks")
    
    # Upload to GPU
    _, upload_time = time_operation("upload_height_r32f", renderer.upload_height_r32f)
    results['operations']['upload_height'] = upload_time
    
    # Read back for validation
    readback, readback_time = time_operation("read_full_height_texture", renderer.read_full_height_texture)
    results['operations']['readback_height'] = readback_time
    
    readback_ptr = ptr(readback)
    print(f"  Readback heightmap pointer: 0x{readback_ptr:x}")
    
    # Validate roundtrip accuracy
    if readback.shape != heightmap_f32.shape:
        results['warnings'].append(f"Shape mismatch: input {heightmap_f32.shape}, readback {readback.shape}")
    else:
        max_diff = np.max(np.abs(readback - heightmap_f32))
        print(f"  Roundtrip max difference: {max_diff:.8f}")
        if max_diff > 1e-6:
            results['warnings'].append(f"Large roundtrip error: {max_diff}")
        else:
            print("  Roundtrip accuracy is good")
            
    # Input and readback should have different pointers (different purposes)
    if input_ptr == readback_ptr:
        results['warnings'].append("Input and readback have same pointer (unexpected)")
    else:
        print("  Input and readback use different buffers (expected)")
    
    return results


def profile_module_functions(width: int, height: int) -> Dict[str, Any]:
    """Profile module-level convenience functions."""
    print(f"\n=== Profiling Module Functions ({width}x{height}) ===")
    
    results = {
        'width': width,
        'height': height, 
        'operations': {},
        'warnings': [],
        'zero_copy_status': 'test_hooks_unavailable'  # Module functions don't have pointer validation
    }
    
    # Test module-level render_triangle_rgba
    rgba, render_time = time_operation("render_triangle_rgba", f3d.render_triangle_rgba, width, height)
    results['operations']['module_render_rgba'] = render_time
    
    # Validate output
    if rgba.shape != (height, width, 4):
        results['warnings'].append(f"Module function shape mismatch: {rgba.shape}")
    if rgba.dtype != np.uint8:
        results['warnings'].append(f"Module function dtype mismatch: {rgba.dtype}")
        
    # Check zero-copy compatibility
    compat = check_zero_copy_compatibility(rgba, "module_rgba_output")
    if not compat['compatible']:
        results['warnings'].extend([f"Module output issue: {issue}" for issue in compat['issues']])
    else:
        print("  Module function output is zero-copy compatible")
    
    ptr_val = ptr(rgba)
    print(f"  Module function output pointer: 0x{ptr_val:x}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile zero-copy pathways in forge3d rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --render-size 1024x1024
  %(prog)s --render-size 512x512 --terrain-size 128x128
  %(prog)s --render-size 256x256 --terrain-size 64x64 --verbose
        """
    )
    
    parser.add_argument('--render-size', type=parse_size, default=(512, 512),
                        help='Render buffer size as WIDTHxHEIGHT (default: 512x512)')
    parser.add_argument('--terrain-size', type=parse_size, 
                        help='Terrain heightmap size as WIDTHxHEIGHT (default: same as render-size)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed timing and pointer information')
    parser.add_argument('--skip-terrain', action='store_true',
                        help='Skip terrain input profiling (RGBA output only)')
    
    args = parser.parse_args()
    
    render_width, render_height = args.render_size
    terrain_width, terrain_height = args.terrain_size or args.render_size
    
    print("=== forge3d Zero-Copy Profiler ===")
    print(f"Render size: {render_width}x{render_height}")
    if not args.skip_terrain:
        print(f"Terrain size: {terrain_width}x{terrain_height}")
    
    # Collect all results
    all_results = {}
    total_warnings = []
    
    try:
        # Profile RGBA output
        rgba_results = profile_rgba_output(render_width, render_height)
        all_results['rgba_output'] = rgba_results
        total_warnings.extend(rgba_results['warnings'])
        
        # Profile height input
        if not args.skip_terrain:
            height_results = profile_height_input(terrain_width, terrain_height, render_width, render_height)
            all_results['height_input'] = height_results
            total_warnings.extend(height_results['warnings'])
        
        # Profile module functions
        module_results = profile_module_functions(render_width, render_height)
        all_results['module_functions'] = module_results
        total_warnings.extend(module_results['warnings'])
        
        # Summary
        print(f"\n=== Summary ===")
        
        # Check if zero-copy was validated successfully
        zero_copy_confirmed = True
        for category, results in all_results.items():
            if results.get('zero_copy_status') == 'zero_copy_confirmed':
                continue
            elif results.get('zero_copy_status') == 'test_hooks_unavailable':
                # Don't fail if test hooks unavailable  
                continue
            else:
                zero_copy_confirmed = False
                break
        
        if len(total_warnings) == 0 and zero_copy_confirmed:
            print("Zero-copy pathways validated successfully") 
            print("zero-copy OK")
            status = 0
        elif len(total_warnings) == 0:
            print("No warnings detected")
            if zero_copy_confirmed:
                print("zero-copy OK")
            status = 0
        else:
            print(f"{len(total_warnings)} warnings detected:")
            for warning in total_warnings:
                print(f"  - {warning}")
            status = 1
            
        # Performance summary
        print(f"\nPerformance summary:")
        for category, results in all_results.items():
            if 'operations' in results:
                total_time = sum(results['operations'].values()) * 1000  # convert to ms
                print(f"  {category}: {total_time:.2f} ms total")
        
        if args.verbose:
            print(f"\nDetailed results: {all_results}")
            
    except Exception as e:
        print(f"\nProfiling failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        status = 2
    
    sys.exit(status)


if __name__ == '__main__':
    main()