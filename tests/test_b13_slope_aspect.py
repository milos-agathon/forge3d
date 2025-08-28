# B13-BEGIN:slope-aspect-tests
"""Tests for B13: Slope/aspect computation using finite differences"""
import os
import pytest
import numpy as np

# Skip if terrain tests are not enabled
SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1")

def test_slope_aspect_flat_surface():
    """Test slope/aspect computation on a flat surface"""
    import forge3d as f3d
    
    # Create a TerrainSpike instance for analysis
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create flat height field (3x3 grid)
    heights = np.full(9, 10.0, dtype=np.float32)  # All same height
    
    # Compute slopes and aspects
    slopes, aspects = ts.slope_aspect_compute(heights, 3, 3, dx=1.0, dy=1.0)
    
    # All slopes should be near zero for flat surface
    assert len(slopes) == 9
    assert len(aspects) == 9
    assert np.all(slopes < 0.1), f"Slopes should be near zero for flat surface, got max={slopes.max()}"

def test_slope_aspect_simple_ramp():
    """Test slope/aspect computation on a simple east-facing ramp"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create east-facing ramp: heights increase from west to east
    heights = np.array([
        0.0, 1.0, 2.0,  # Bottom row
        0.0, 1.0, 2.0,  # Middle row  
        0.0, 1.0, 2.0,  # Top row
    ], dtype=np.float32)
    
    slopes, aspects = ts.slope_aspect_compute(heights, 3, 3, dx=1.0, dy=1.0)
    
    # Center point should have significant slope
    center_slope = slopes[4]  # Middle point (1,1)
    assert center_slope > 30.0, f"Ramp should have significant slope, got {center_slope}"
    
    # Aspect should be roughly westward (around 270 degrees) - steepest descent direction
    # Note: aspect points in direction of steepest descent, so east-facing slope has west aspect
    center_aspect = aspects[4]
    assert 225.0 < center_aspect < 315.0, f"East-facing ramp should have westward aspect (steepest descent), got {center_aspect}"

def test_slope_aspect_north_facing_ramp():
    """Test slope/aspect computation on a north-facing ramp"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create north-facing ramp: heights increase from south to north
    heights = np.array([
        0.0, 0.0, 0.0,  # Bottom row (south)
        1.0, 1.0, 1.0,  # Middle row  
        2.0, 2.0, 2.0,  # Top row (north)
    ], dtype=np.float32)
    
    slopes, aspects = ts.slope_aspect_compute(heights, 3, 3, dx=1.0, dy=1.0)
    
    # Center point should have significant slope
    center_slope = slopes[4]
    assert center_slope > 30.0, f"Ramp should have significant slope, got {center_slope}"
    
    # Aspect should be roughly northward (around 0/360 degrees)
    center_aspect = aspects[4]
    # North can be near 0° or 360°, so check both ranges
    assert (center_aspect < 45.0) or (center_aspect > 315.0), \
           f"North-facing ramp should have northward aspect, got {center_aspect}"

def test_slope_aspect_input_validation():
    """Test input validation for slope/aspect computation"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Test wrong array size
    heights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    with pytest.raises(Exception, match="Heights array length .* does not match dimensions"):
        ts.slope_aspect_compute(heights, 3, 3)
    
    # Test non-contiguous array - create a 3x3 array then make it non-contiguous
    heights_2d = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]], dtype=np.float32)
    heights_non_contig = heights_2d[:, ::2].flatten()  # Non-contiguous, gives 9 elements
    with pytest.raises(Exception, match="Heights array must be C-contiguous"):
        ts.slope_aspect_compute(heights_non_contig, 3, 3)

def test_slope_aspect_various_spacings():
    """Test slope/aspect computation with different dx/dy spacings"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Simple gradient in X direction
    heights = np.array([
        0.0, 2.0, 4.0,
        0.0, 2.0, 4.0,
        0.0, 2.0, 4.0,
    ], dtype=np.float32)
    
    # Test with different spacings
    slopes1, aspects1 = ts.slope_aspect_compute(heights, 3, 3, dx=1.0, dy=1.0)
    slopes2, aspects2 = ts.slope_aspect_compute(heights, 3, 3, dx=2.0, dy=1.0)
    
    # With larger dx, slope should be smaller (same height change over larger distance)
    center_slope1 = slopes1[4]
    center_slope2 = slopes2[4]
    assert center_slope2 < center_slope1, \
           f"Larger dx should result in smaller slope, got {center_slope1} vs {center_slope2}"
    
    # Aspect should remain similar (both eastward)
    center_aspect1 = aspects1[4]
    center_aspect2 = aspects2[4]
    assert abs(center_aspect1 - center_aspect2) < 10.0, \
           f"Aspect should be similar, got {center_aspect1} vs {center_aspect2}"

def test_slope_aspect_accuracy_requirement():
    """Test that slope/aspect meets ≤0.5° RMSE requirement on synthetic data"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create a known analytical surface: plane with known slope
    # z = ax + by + c, where a=0.1, b=0.0 (slope in x direction only)
    width, height = 5, 5
    a, b, c = 0.1, 0.0, 0.0
    dx, dy = 1.0, 1.0
    
    heights = np.zeros(width * height, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            world_x = x * dx
            world_y = y * dy
            heights[idx] = a * world_x + b * world_y + c
    
    slopes, aspects = ts.slope_aspect_compute(heights, width, height, dx, dy)
    
    # For a plane z = ax + by + c:
    # dz/dx = a, dz/dy = b
    # slope = atan(sqrt(a² + b²))
    # aspect = atan2(-b, -a) + π/2 (converted to geographic)
    
    expected_slope_rad = np.arctan(np.sqrt(a*a + b*b))
    expected_slope_deg = np.degrees(expected_slope_rad)
    
    # Check interior points (avoid boundary effects)
    for y in range(1, height-1):
        for x in range(1, width-1):
            idx = y * width + x
            slope_error = abs(slopes[idx] - expected_slope_deg)
            assert slope_error <= 0.5, \
                   f"Slope error {slope_error:.3f}° at ({x},{y}) exceeds 0.5° requirement"

def test_slope_aspect_return_types():
    """Test that slope/aspect computation returns correct types and shapes"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create simple height field (3x3 grid)
    heights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    
    slopes, aspects = ts.slope_aspect_compute(heights, 3, 3, dx=1.0, dy=1.0)
    
    # Check types and shapes
    assert isinstance(slopes, np.ndarray), "Slopes should be NumPy array"
    assert isinstance(aspects, np.ndarray), "Aspects should be NumPy array"
    assert slopes.dtype == np.float32, f"Slopes should be float32, got {slopes.dtype}"
    assert aspects.dtype == np.float32, f"Aspects should be float32, got {aspects.dtype}"
    assert slopes.shape == (9,), f"Slopes should have shape (9,), got {slopes.shape}"
    assert aspects.shape == (9,), f"Aspects should have shape (9,), got {aspects.shape}"
    
    # Check value ranges
    assert np.all((slopes >= 0.0) & (slopes <= 90.0)), "Slopes should be in [0°, 90°]"
    assert np.all((aspects >= 0.0) & (aspects <= 360.0)), "Aspects should be in [0°, 360°]"

# B13-END:slope-aspect-tests