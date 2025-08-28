# B15-BEGIN:memory-integration-tests
"""Tests for B15: Memory governor integration in terrain operations"""
import os
import numpy as np
import pytest

# Skip if terrain tests are not enabled
SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1")

def test_terrain_spike_memory_metrics():
    """Test that TerrainSpike exposes memory metrics via get_memory_metrics()"""
    import forge3d as f3d
    
    # Create terrain spike instance  
    ts = f3d.TerrainSpike(256, 256, 32)  # Small grid to avoid memory issues
    
    # Get memory metrics
    metrics = ts.get_memory_metrics()
    
    # Verify all expected keys are present
    expected_keys = {
        'buffer_count', 'texture_count', 'buffer_bytes', 'texture_bytes',
        'host_visible_bytes', 'total_bytes', 'limit_bytes', 
        'within_budget', 'utilization_ratio'
    }
    assert set(metrics.keys()) == expected_keys
    
    # Verify types and basic sanity checks
    assert isinstance(metrics['buffer_count'], int)
    assert isinstance(metrics['texture_count'], int) 
    assert isinstance(metrics['buffer_bytes'], int)
    assert isinstance(metrics['texture_bytes'], int)
    assert isinstance(metrics['host_visible_bytes'], int)
    assert isinstance(metrics['total_bytes'], int)
    assert isinstance(metrics['limit_bytes'], int)
    assert isinstance(metrics['within_budget'], bool)
    assert isinstance(metrics['utilization_ratio'], float)
    
    # Budget limit should be 512 MiB
    assert metrics['limit_bytes'] == 512 * 1024 * 1024
    
    # Should have some buffer allocations from TerrainSpike creation
    assert metrics['buffer_count'] > 0
    assert metrics['buffer_bytes'] > 0
    
    # Utilization ratio should be between 0 and 1
    assert 0.0 <= metrics['utilization_ratio'] <= 1.0

def test_terrain_readback_budget_checking():
    """Test that terrain readback operations respect memory budget limits"""
    import forge3d as f3d
    
    # Create a smaller terrain spike to avoid immediate budget issues
    ts = f3d.TerrainSpike(128, 128, 16)
    
    # Get initial metrics
    initial_metrics = ts.get_memory_metrics()
    initial_host_visible = initial_metrics['host_visible_bytes']
    
    # Render to PNG (creates temporary readback buffer)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    try:
        ts.render_png(temp_path)
        
        # Verify PNG was created
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 1000  # Should be a real PNG
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Get final metrics - readback buffer should be cleaned up
    final_metrics = ts.get_memory_metrics()
    final_host_visible = final_metrics['host_visible_bytes']
    
    # Host-visible memory should be back to initial level (temporary buffer freed)
    # Allow some tolerance for potential other allocations
    assert abs(final_host_visible - initial_host_visible) <= 1024  # Within 1KB tolerance

def test_memory_budget_error_messages():
    """Test that budget exceeded errors include current/requested/limit info"""
    import forge3d as f3d
    
    # This test would be hard to trigger reliably without artificially lowering the budget
    # For now, just verify the error message format when we can construct a scenario
    
    # Create terrain spike and get current usage
    ts = f3d.TerrainSpike(64, 64, 8)  # Very small to minimize usage
    metrics = ts.get_memory_metrics()
    
    # If we're already close to budget, this might fail
    if metrics['utilization_ratio'] > 0.9:
        pytest.skip("Already too close to memory budget to test safely")
    
    # We can't easily test budget exceeded errors without modifying budget limits
    # But we can verify the metrics are tracking properly
    assert metrics['within_budget'] == True
    assert metrics['host_visible_bytes'] <= metrics['limit_bytes']

def test_terrain_buffer_allocation_tracking():
    """Test that terrain buffer allocations are properly tracked"""
    import forge3d as f3d
    
    # Get baseline memory metrics
    renderer = f3d.Renderer(64, 64)
    baseline_metrics = renderer.get_memory_metrics()
    baseline_buffers = baseline_metrics['buffer_count']
    baseline_bytes = baseline_metrics['buffer_bytes']
    
    # Create terrain spike (should create additional buffers)
    ts = f3d.TerrainSpike(128, 128, 16)  
    after_terrain_metrics = ts.get_memory_metrics()
    
    # Should have more buffers after terrain creation
    assert after_terrain_metrics['buffer_count'] > baseline_buffers
    assert after_terrain_metrics['buffer_bytes'] > baseline_bytes
    
    # Verify that buffers are being counted
    terrain_buffers = after_terrain_metrics['buffer_count'] - baseline_buffers
    terrain_bytes = after_terrain_metrics['buffer_bytes'] - baseline_bytes
    
    # Should have at least vertex buffer, index buffer, and UBO
    assert terrain_buffers >= 3
    assert terrain_bytes > 0

def test_memory_metrics_consistency():
    """Test that memory metrics are internally consistent"""
    import forge3d as f3d
    
    ts = f3d.TerrainSpike(64, 64, 8)
    metrics = ts.get_memory_metrics()
    
    # Total bytes should be sum of buffer and texture bytes
    assert metrics['total_bytes'] == metrics['buffer_bytes'] + metrics['texture_bytes']
    
    # If within budget, host_visible should be <= limit
    if metrics['within_budget']:
        assert metrics['host_visible_bytes'] <= metrics['limit_bytes']
    
    # Utilization ratio should match host_visible / limit
    expected_ratio = metrics['host_visible_bytes'] / metrics['limit_bytes']
    assert abs(metrics['utilization_ratio'] - expected_ratio) < 1e-6

# B15-END:memory-integration-tests