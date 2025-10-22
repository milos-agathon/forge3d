#!/usr/bin/env python3
"""
Quick validation script for terrain draping implementation.
Run this to verify the complete pipeline works.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from forge3d.terrain import drape_landcover, estimate_memory_usage
    print("✓ Successfully imported forge3d.terrain")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("Run: maturin develop --release")
    sys.exit(1)


def test_basic_functionality():
    """Test basic terrain draping with synthetic data."""
    print("\n" + "="*60)
    print("Testing Basic Terrain Draping Functionality")
    print("="*60)
    
    # Create synthetic DEM (256x256 with Gaussian hill)
    print("\n1. Creating synthetic DEM (256x256)...")
    x = np.linspace(-5, 5, 256)
    y = np.linspace(-5, 5, 256)
    X, Y = np.meshgrid(x, y)
    dem = (100 * np.exp(-(X**2 + Y**2) / 5)).astype(np.float32)
    print(f"   DEM shape: {dem.shape}, range: {dem.min():.1f} to {dem.max():.1f}")
    
    # Create synthetic land-cover (gradient from blue to green)
    print("\n2. Creating synthetic land-cover (256x256 RGBA)...")
    landcover = np.zeros((256, 256, 4), dtype=np.uint8)
    
    # Blue water at bottom, green terrain at top
    for i in range(256):
        ratio = i / 256
        landcover[i, :, 0] = int((1 - ratio) * 30)      # Red decreases
        landcover[i, :, 1] = int(ratio * 180 + (1-ratio) * 144)  # Green increases
        landcover[i, :, 2] = int((1 - ratio) * 255)     # Blue decreases
        landcover[i, :, 3] = 255                         # Full alpha
    
    print(f"   Land-cover shape: {landcover.shape}")
    
    # Estimate memory
    print("\n3. Estimating GPU memory usage...")
    mem_info = estimate_memory_usage(dem.shape)
    print(f"   DEM texture: {mem_info['heightmap_r32f'] / (1024**2):.2f} MB")
    print(f"   Land-cover texture: {mem_info['landcover_rgba8'] / (1024**2):.2f} MB")
    print(f"   Total: {mem_info['total_mb']:.2f} MB")
    print(f"   Exceeds 512MB budget: {mem_info['exceeds_512mb_budget']}")
    
    # Render
    print("\n4. Rendering terrain with GPU draping...")
    print("   Output: 640x480, height_scale=2.0")
    
    try:
        result = drape_landcover(
            dem,
            landcover,
            width=640,
            height=480,
            height_scale=2.0,
            camera_theta=45.0,
            camera_phi=30.0,
        )
        print(f"   ✓ Render successful!")
        print(f"   Result shape: {result.shape}, dtype: {result.dtype}")
        
        # Validate output
        assert result.shape == (480, 640, 4), f"Expected (480, 640, 4), got {result.shape}"
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
        
        # Check for non-empty output
        non_white = np.any(result[:, :, :3] != 255, axis=2)
        filled_pixels = non_white.sum()
        total_pixels = 480 * 640
        fill_pct = (filled_pixels / total_pixels) * 100
        
        print(f"   Non-background pixels: {filled_pixels:,} / {total_pixels:,} ({fill_pct:.1f}%)")
        
        if fill_pct < 5:
            print(f"   ⚠️  Warning: Output appears mostly empty ({fill_pct:.1f}% filled)")
        
        # Check color diversity
        unique_colors = len(np.unique(result.reshape(-1, 4), axis=0))
        print(f"   Unique colors: {unique_colors:,}")
        
        # Save test output
        output_path = Path(__file__).parent / "terrain_drape_test_output.png"
        try:
            from PIL import Image
            Image.fromarray(result).save(output_path)
            print(f"\n5. ✓ Saved test output to: {output_path}")
        except ImportError:
            print(f"\n5. ⚠️  PIL not available, skipping save")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Render failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_validation():
    """Test input validation."""
    print("\n" + "="*60)
    print("Testing Input Validation")
    print("="*60)
    
    dem = np.random.rand(100, 100).astype(np.float32)
    landcover = np.zeros((100, 100, 4), dtype=np.uint8)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Mismatched dimensions
    print("\n1. Testing mismatched dimensions...")
    try:
        bad_landcover = np.zeros((50, 50, 4), dtype=np.uint8)
        drape_landcover(dem, bad_landcover)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        if "same dimensions" in str(e):
            print(f"   ✓ Correctly rejected: {e}")
            tests_passed += 1
        else:
            print(f"   ✗ Wrong error message: {e}")
    
    # Test 2: Wrong landcover shape
    print("\n2. Testing wrong land-cover shape (RGB instead of RGBA)...")
    try:
        bad_landcover = np.zeros((100, 100, 3), dtype=np.uint8)
        drape_landcover(dem, bad_landcover)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        if "RGBA" in str(e):
            print(f"   ✓ Correctly rejected: {e}")
            tests_passed += 1
        else:
            print(f"   ✗ Wrong error message: {e}")
    
    # Test 3: Non-finite values
    print("\n3. Testing non-finite DEM values...")
    try:
        bad_dem = dem.copy()
        bad_dem[50, 50] = np.nan
        drape_landcover(bad_dem, landcover)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        if "non-finite" in str(e):
            print(f"   ✓ Correctly rejected: {e}")
            tests_passed += 1
        else:
            print(f"   ✗ Wrong error message: {e}")
    
    print(f"\nValidation tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def main():
    print("\n" + "="*60)
    print("Terrain Draping Implementation Validation")
    print("="*60)
    
    results = []
    
    # Test 1: Basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test 2: Input validation
    results.append(("Input Validation", test_input_validation()))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All validation tests passed!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest tests/test_terrain_drape.py -v")
        print("  2. Try the example: python examples/switzerland_landcover_drape.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
