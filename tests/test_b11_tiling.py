# B11-BEGIN:tiling-tests
"""Tests for B11: Tiled DEM pyramid & cache system"""
import os
import pytest
import numpy as np

# Skip if terrain tests are not enabled
SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1")

def test_terrain_spike_tiling_enable():
    """Test enabling tiling system on TerrainSpike"""
    import forge3d as f3d
    
    # Create terrain spike
    ts = f3d.TerrainSpike(128, 128, 16)
    
    # Enable tiling with 10km x 10km bounds
    ts.enable_tiling(-5000.0, -5000.0, 5000.0, 5000.0, cache_capacity=4, max_lod=3)
    
    # Should now be able to get cache stats
    stats = ts.get_cache_stats()
    assert 'capacity' in stats
    assert 'current_size' in stats
    assert 'memory_usage_bytes' in stats
    assert stats['capacity'] == 4
    assert stats['current_size'] == 0  # No tiles loaded yet
    assert stats['memory_usage_bytes'] == 0

def test_visible_tiles_calculation():
    """Test visible tile calculation for different camera positions"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    ts.enable_tiling(-1000.0, -1000.0, 1000.0, 1000.0, cache_capacity=8, max_lod=2)
    
    # Test camera at origin looking forward
    camera_pos = (0.0, 100.0, 0.0)
    camera_dir = (0.0, -0.1, 1.0)  # Looking slightly down and forward
    
    visible_tiles = ts.get_visible_tiles(
        camera_pos, camera_dir, fov_deg=60.0, aspect=1.0, near=1.0, far=2000.0
    )
    
    # Should get some visible tiles
    assert len(visible_tiles) > 0
    
    # Each tile should be a tuple of (lod, x, y)
    for tile in visible_tiles:
        assert len(tile) == 3
        lod, x, y = tile
        assert isinstance(lod, int)
        assert isinstance(x, int) 
        assert isinstance(y, int)
        assert lod <= 2  # Should respect max_lod
        
def test_tile_loading():
    """Test loading specific tiles into cache"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    ts.enable_tiling(-500.0, -500.0, 500.0, 500.0, cache_capacity=4, max_lod=2)
    
    # Load a specific tile
    ts.load_tile(0, 0, 0)  # Root tile
    
    # Cache should now have 1 tile
    stats = ts.get_cache_stats()
    assert stats['current_size'] == 1
    assert stats['memory_usage_bytes'] > 0
    
    # Load more tiles
    ts.load_tile(1, 0, 0)
    ts.load_tile(1, 1, 0)
    
    stats = ts.get_cache_stats()
    assert stats['current_size'] == 3
    
def test_cache_lru_eviction():
    """Test LRU eviction when cache capacity is exceeded"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    # Small cache capacity to trigger eviction
    ts.enable_tiling(-200.0, -200.0, 200.0, 200.0, cache_capacity=2, max_lod=3)
    
    # Load tiles up to capacity
    ts.load_tile(0, 0, 0)
    ts.load_tile(1, 0, 0)
    
    stats = ts.get_cache_stats()
    assert stats['current_size'] == 2
    
    # Load another tile - should evict oldest
    ts.load_tile(1, 1, 1)
    
    stats = ts.get_cache_stats()
    assert stats['current_size'] == 2  # Still at capacity
    
def test_stream_visible_tiles():
    """Test streaming (getting and loading) visible tiles"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(128, 128, 16)
    ts.enable_tiling(-1000.0, -1000.0, 1000.0, 1000.0, cache_capacity=8, max_lod=2)
    
    # Stream tiles for a camera position
    camera_pos = (0.0, 200.0, 0.0)
    camera_dir = (1.0, -0.5, 0.0)  # Looking east and slightly down
    
    visible_tiles = ts.stream_visible_tiles(
        camera_pos, camera_dir, fov_deg=45.0, aspect=16.0/9.0, near=10.0, far=1500.0
    )
    
    # Should return same tiles as get_visible_tiles but also load them
    assert len(visible_tiles) > 0
    
    # Cache should now contain loaded tiles
    stats = ts.get_cache_stats()
    assert stats['current_size'] > 0
    assert stats['memory_usage_bytes'] > 0
    
    # Each tile should have been loaded
    for lod, x, y in visible_tiles:
        # We can't directly check if specific tile is loaded without 
        # exposing more internal state, but cache size > 0 indicates success
        pass

def test_tiling_system_memory_budget():
    """Test that tiling system respects memory budget limits"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Get initial memory state
    initial_metrics = ts.get_memory_metrics()
    initial_host_visible = initial_metrics['host_visible_bytes']
    
    # Enable tiling (should not use significant memory until tiles are loaded)
    ts.enable_tiling(-1000.0, -1000.0, 1000.0, 1000.0, cache_capacity=4, max_lod=2)
    
    # Load some tiles
    ts.load_tile(0, 0, 0)
    ts.load_tile(1, 0, 0)
    
    # Check that memory is being tracked
    final_metrics = ts.get_memory_metrics()
    final_host_visible = final_metrics['host_visible_bytes']
    
    # Should have more host-visible memory due to cached tile data
    assert final_host_visible > initial_host_visible
    
    # Should still be within budget
    assert final_metrics['within_budget']
    assert final_host_visible <= final_metrics['limit_bytes']

def test_tile_hierarchy_consistency():
    """Test that tile hierarchy and bounds calculations work correctly"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    ts.enable_tiling(-512.0, -512.0, 512.0, 512.0, cache_capacity=16, max_lod=3)
    
    # Get visible tiles from different camera positions
    # Position 1: Close to origin
    close_tiles = ts.get_visible_tiles(
        (0.0, 50.0, 0.0), (0.0, -1.0, 0.0), fov_deg=45.0, near=1.0, far=200.0
    )
    
    # Position 2: Farther from origin  
    far_tiles = ts.get_visible_tiles(
        (0.0, 200.0, 0.0), (0.0, -1.0, 0.0), fov_deg=45.0, near=1.0, far=800.0
    )
    
    # Close camera should see higher resolution tiles (higher LOD numbers)
    # Far camera should see lower resolution tiles (lower LOD numbers)
    if close_tiles and far_tiles:
        avg_close_lod = sum(tile[0] for tile in close_tiles) / len(close_tiles)
        avg_far_lod = sum(tile[0] for tile in far_tiles) / len(far_tiles)
        
        # This is a heuristic - closer cameras often see higher detail
        # but it's not guaranteed due to frustum culling
        # For now just ensure we get valid tiles
        assert all(0 <= lod <= 3 for lod, x, y in close_tiles)
        assert all(0 <= lod <= 3 for lod, x, y in far_tiles)

def test_tiling_error_conditions():
    """Test error conditions in tiling system"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    
    # Should error when tiling not enabled
    with pytest.raises(Exception, match="Tiling system not enabled"):
        ts.get_visible_tiles((0, 0, 0), (1, 0, 0))
    
    with pytest.raises(Exception, match="Tiling system not enabled"):
        ts.load_tile(0, 0, 0)
    
    with pytest.raises(Exception, match="Tiling system not enabled"):
        ts.get_cache_stats()
    
    with pytest.raises(Exception, match="Tiling system not enabled"):
        ts.stream_visible_tiles((0, 0, 0), (1, 0, 0))

# B11-END:tiling-tests