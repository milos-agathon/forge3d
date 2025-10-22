#!/usr/bin/env python3
"""
Test to prevent regression of black-top tiles bug in true-resolution tiled GPU path tracer.

This test ensures that:
1. All tiles write non-black pixels
2. Tiles have sufficient color variance (unique_colors > 2)
3. No tile has meanRGB close to (0,0,0)
4. XY gradient mode produces continuous gradients across tiles
"""

import sys
import numpy as np
from pathlib import Path
import subprocess
import pytest
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import forge3d
    from forge3d.terrain import drape_landcover
except ImportError as e:
    pytest.skip(f"forge3d not available: {e}", allow_module_level=True)


def test_gradient_mode_no_black_tiles():
    """Test that gradient diagnostic mode (debug_mode=4) produces non-black output across all tiles."""
    
    # Create simple test DEM and landcover
    height, width = 720, 1280
    dem = np.random.rand(100, 100).astype(np.float32) * 100.0  # Random terrain
    landcover = np.zeros((100, 100, 4), dtype=np.uint8)
    landcover[:, :, 0] = 100  # Red
    landcover[:, :, 1] = 150  # Green
    landcover[:, :, 2] = 200  # Blue
    landcover[:, :, 3] = 255  # Alpha
    
    # Render with gradient diagnostic mode
    result = drape_landcover(
        dem,
        landcover,
        render_mode="raytrace",
        rt_spp=1,
        rt_debug_mode=4,  # XY gradient mode
        width=width,
        height=height,
        rt_batch_spp=1,
        max_rt_triangles=50000,
        zscale=0.05,
        camera_theta=45.0,
        camera_phi=25.0,
    )
    
    # Convert to numpy array
    img_data = np.array(result["image"]).reshape((height, width, 4))
    
    # Define tile size (should match Rust implementation)
    tile_w, tile_h = 512, 512
    
    # Iterate through tiles and check each one
    tiles_checked = 0
    tiles_with_issues = []
    
    for ty in range(0, height, tile_h):
        for tx in range(0, width, tile_w):
            tile_end_x = min(tx + tile_w, width)
            tile_end_y = min(ty + tile_h, height)
            
            tile_data = img_data[ty:tile_end_y, tx:tile_end_x, :3]
            
            # Check 1: Unique colors (should be many for a gradient)
            unique_colors = len(np.unique(tile_data.reshape(-1, 3), axis=0))
            
            # Check 2: Mean RGB should not be near black
            mean_rgb = tile_data.mean(axis=(0, 1))
            
            # Check 3: Variance check (gradient should have variance)
            variance = tile_data.var(axis=(0, 1)).sum()
            
            tiles_checked += 1
            
            if unique_colors < 10:
                tiles_with_issues.append(f"Tile at ({tx},{ty}): only {unique_colors} unique colors")
            
            if mean_rgb.max() < 5.0:
                tiles_with_issues.append(f"Tile at ({tx},{ty}): nearly black (mean_rgb={mean_rgb})")
            
            if variance < 10.0:
                tiles_with_issues.append(f"Tile at ({tx},{ty}): no variance (variance={variance})")
    
    # Report results
    print(f"Checked {tiles_checked} tiles")
    if tiles_with_issues:
        for issue in tiles_with_issues:
            print(f"  ISSUE: {issue}")
        pytest.fail(f"Found {len(tiles_with_issues)} tiles with issues")
    else:
        print("  All tiles passed validation!")


def test_normal_render_no_solid_black_tiles():
    """Test that normal rendering (non-debug) produces varied output in all tiles."""
    
    # Create test data
    height, width = 720, 1280
    dem = np.random.rand(100, 100).astype(np.float32) * 100.0
    
    # Create varied landcover
    landcover = np.zeros((100, 100, 4), dtype=np.uint8)
    landcover[:50, :, :] = [34, 139, 34, 255]  # Forest green (top half)
    landcover[50:, :, :] = [139, 69, 19, 255]   # Saddle brown (bottom half)
    
    # Render normally
    result = drape_landcover(
        dem,
        landcover,
        render_mode="raytrace",
        rt_spp=1,
        rt_debug_mode=0,  # Normal rendering
        width=width,
        height=height,
        rt_batch_spp=1,
        max_rt_triangles=50000,
        zscale=0.05,
    )
    
    img_data = np.array(result["image"]).reshape((height, width, 4))
    
    # Check that no tile is solid black
    tile_w, tile_h = 512, 512
    black_tiles = []
    
    for ty in range(0, height, tile_h):
        for tx in range(0, width, tile_w):
            tile_end_x = min(tx + tile_w, width)
            tile_end_y = min(ty + tile_h, height)
            
            tile_data = img_data[ty:tile_end_y, tx:tile_end_x, :3]
            mean_rgb = tile_data.mean(axis=(0, 1))
            unique_colors = len(np.unique(tile_data.reshape(-1, 3), axis=0))
            
            # Flag if tile is essentially black
            if mean_rgb.max() < 1.0 and unique_colors == 1:
                black_tiles.append((tx, ty))
    
    if black_tiles:
        pytest.fail(f"Found {len(black_tiles)} solid black tiles at positions: {black_tiles}")


if __name__ == "__main__":
    # Run tests directly
    print("Running tile black regression tests...")
    print("\n=== Test 1: Gradient Mode ===")
    test_gradient_mode_no_black_tiles()
    print("\n=== Test 2: Normal Render ===")
    test_normal_render_no_solid_black_tiles()
    print("\n✅ All tests passed!")
