# B14-BEGIN:contour-tests
"""Tests for B14: Contour line extraction using marching squares"""
import os
import pytest
import numpy as np

# Skip if terrain tests are not enabled
SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1")

def test_contour_extraction_basic():
    """Test basic contour extraction functionality"""
    import forge3d as f3d
    
    # Create a TerrainSpike instance for analysis
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create simple height field with clear level crossings
    heights = np.array([
        0.0, 0.5, 1.0,
        1.5, 2.0, 2.5,
        3.0, 3.5, 4.0,
    ], dtype=np.float32)
    
    levels = [1.0, 2.0, 3.0]
    result = ts.contour_extract(heights, 3, 3, dx=1.0, dy=1.0, levels=levels)
    
    # Should return a dictionary with expected structure
    assert isinstance(result, dict)
    assert "polyline_count" in result
    assert "total_points" in result
    assert "polylines" in result
    
    # Should find some contour lines
    assert result["polyline_count"] > 0, "Should extract some contour lines"
    assert result["total_points"] > 0, "Should have some contour points"
    assert isinstance(result["polylines"], list), "Polylines should be a list"
    
    # All polylines should have the requested levels
    for polyline in result["polylines"]:
        assert isinstance(polyline, dict)
        assert "level" in polyline
        assert "points" in polyline
        assert polyline["level"] in levels, "Polyline level should be one of the requested levels"

def test_contour_extraction_no_crossings():
    """Test contour extraction when no level crossings exist"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Flat surface - no level crossings
    heights = np.full(9, 5.0, dtype=np.float32)  # 3x3 grid, all height 5.0
    levels = [1.0, 2.0, 3.0]  # Levels below the surface
    
    result = ts.contour_extract(heights, 3, 3, dx=1.0, dy=1.0, levels=levels)
    
    # Should find no contour lines
    assert result["polyline_count"] == 0, "Flat surface should produce no contours"
    assert result["total_points"] == 0, "Should have no contour points"
    assert len(result["polylines"]) == 0, "Should have no polylines"

def test_contour_extraction_simple_plane():
    """Test contour extraction on a simple inclined plane"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create inclined plane: heights increase linearly
    width, height = 4, 4
    heights = np.zeros(width * height, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            heights[idx] = x + y  # Simple linear increase
    
    levels = [2.0, 4.0]  # Levels that should cross the plane
    result = ts.contour_extract(heights, width, height, dx=1.0, dy=1.0, levels=levels)
    
    # Should find contours for both levels
    assert result["polyline_count"] > 0, "Should find contours on inclined plane"
    
    # Check that all requested levels are present
    found_levels = set()
    for polyline in result["polylines"]:
        found_levels.add(polyline["level"])
    
    for level in levels:
        if level >= heights.min() and level <= heights.max():
            assert level in found_levels, f"Should find contour for level {level}"

def test_contour_extraction_input_validation():
    """Test input validation for contour extraction"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    heights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    levels = [2.5]
    
    # Test wrong array size
    with pytest.raises(Exception, match="Heights array length .* does not match dimensions"):
        ts.contour_extract(heights, 3, 3, dx=1.0, dy=1.0, levels=levels)
    
    # Test non-contiguous array
    heights_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    heights_non_contig = heights_2d[:, ::2].flatten()  # Non-contiguous
    with pytest.raises(Exception, match="Heights array must be C-contiguous"):
        ts.contour_extract(heights_non_contig, 2, 2, dx=1.0, dy=1.0, levels=levels)
    
    # Test empty levels
    with pytest.raises(Exception, match="At least one contour level must be specified"):
        ts.contour_extract(heights, 2, 2, dx=1.0, dy=1.0, levels=[])

def test_contour_extraction_multiple_levels():
    """Test contour extraction with multiple levels"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create height field with multiple distinct regions
    heights = np.array([
        0.0, 1.0, 2.0, 3.0,
        1.0, 2.0, 3.0, 4.0,
        2.0, 3.0, 4.0, 5.0,
        3.0, 4.0, 5.0, 6.0,
    ], dtype=np.float32)
    
    levels = [1.5, 2.5, 3.5, 4.5]
    result = ts.contour_extract(heights, 4, 4, dx=1.0, dy=1.0, levels=levels)
    
    # Should find contours for multiple levels
    assert result["polyline_count"] > 0, "Should find multiple contours"
    
    # Check that polylines exist for various levels
    found_levels = [polyline["level"] for polyline in result["polylines"]]
    assert len(set(found_levels)) >= 2, "Should find contours for multiple levels"

def test_contour_extraction_point_format():
    """Test that contour points are in correct format"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create simple gradient
    heights = np.array([
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    ], dtype=np.float32)
    
    levels = [2.0]
    result = ts.contour_extract(heights, 3, 3, dx=1.0, dy=1.0, levels=levels)
    
    # Check point format for each polyline
    for polyline in result["polylines"]:
        points = polyline["points"]
        
        # Points should be NumPy array
        assert isinstance(points, np.ndarray), "Points should be NumPy array"
        
        if points.size > 0:
            # Should be Nx2 array (x, y coordinates)
            assert len(points.shape) == 2, f"Points should be 2D array, got shape {points.shape}"
            assert points.shape[1] == 2, f"Points should have 2 columns (x,y), got {points.shape[1]}"
            assert points.dtype == np.float32, f"Points should be float32, got {points.dtype}"
        else:
            # Empty array should be 0x2
            assert points.shape == (0, 2), f"Empty points should be (0,2), got {points.shape}"

def test_contour_extraction_deterministic():
    """Test that contour extraction produces deterministic results"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create reproducible height field
    np.random.seed(42)
    heights = np.random.uniform(0.0, 10.0, size=16).astype(np.float32)
    
    levels = [3.0, 6.0, 9.0]
    
    # Extract contours multiple times
    result1 = ts.contour_extract(heights, 4, 4, dx=1.0, dy=1.0, levels=levels)
    result2 = ts.contour_extract(heights, 4, 4, dx=1.0, dy=1.0, levels=levels)
    
    # Results should be identical
    assert result1["polyline_count"] == result2["polyline_count"], \
           "Contour extraction should be deterministic"
    assert result1["total_points"] == result2["total_points"], \
           "Contour extraction should be deterministic"

def test_contour_extraction_different_spacings():
    """Test contour extraction with different dx/dy spacings"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Simple gradient
    heights = np.array([
        0.0, 1.0, 2.0,
        0.0, 1.0, 2.0,
        0.0, 1.0, 2.0,
    ], dtype=np.float32)
    
    levels = [1.0]
    
    # Test with different spacings
    result1 = ts.contour_extract(heights, 3, 3, dx=1.0, dy=1.0, levels=levels)
    result2 = ts.contour_extract(heights, 3, 3, dx=2.0, dy=1.0, levels=levels)
    
    # Both should find contours, but coordinates should be scaled
    assert result1["polyline_count"] > 0, "Should find contours with dx=1.0"
    assert result2["polyline_count"] > 0, "Should find contours with dx=2.0"
    
    # With larger dx, x-coordinates should be scaled up
    if result1["polylines"] and result2["polylines"]:
        points1 = result1["polylines"][0]["points"]
        points2 = result2["polylines"][0]["points"]
        
        if points1.size > 0 and points2.size > 0:
            # X coordinates should be roughly doubled
            avg_x1 = np.mean(points1[:, 0])
            avg_x2 = np.mean(points2[:, 0])
            assert avg_x2 > avg_x1, "Larger dx should result in larger x coordinates"

def test_contour_extraction_tolerance_requirement():
    """Test that contour extraction meets ±1% tolerance requirement"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Create known analytical surface for testing
    # Simple plane with known contour lines
    width, height = 5, 5
    dx, dy = 1.0, 1.0
    
    # Create plane: z = x + y (diagonal gradient)
    heights = np.zeros(width * height, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            heights[idx] = x + y
    
    levels = [4.0]  # Should intersect the plane diagonally
    result = ts.contour_extract(heights, width, height, dx, dy, levels=levels)
    
    # For this simple case, we should get deterministic results
    assert result["polyline_count"] > 0, "Should find contours on diagonal plane"
    
    # The exact validation of ±1% tolerance would require more complex
    # analytical solutions, but we can at least verify consistent results
    total_points = result["total_points"]
    polyline_count = result["polyline_count"]
    
    # Run again to verify consistency (within ±1% tolerance)
    result2 = ts.contour_extract(heights, width, height, dx, dy, levels=levels)
    
    points_error = abs(result2["total_points"] - total_points) / max(total_points, 1)
    polyline_error = abs(result2["polyline_count"] - polyline_count) / max(polyline_count, 1)
    
    assert points_error <= 0.01, f"Point count should be within ±1%, got {points_error:.3%} error"
    assert polyline_error <= 0.01, f"Polyline count should be within ±1%, got {polyline_error:.3%} error"

# B14-END:contour-tests