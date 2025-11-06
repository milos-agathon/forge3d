#!/usr/bin/env python3
"""
P7-06: Test for render_brdf_tile Python shim
Tests that the top-level Python function correctly delegates to native module.
"""
import sys
import pytest
import numpy as np

# Test import from top-level module
try:
    import forge3d as f3d
    TOP_LEVEL_IMPORT_OK = True
except ImportError:
    TOP_LEVEL_IMPORT_OK = False

# Check if native module is available
try:
    import forge3d._forge3d as f3d_native
    NATIVE_AVAILABLE = hasattr(f3d_native, 'render_brdf_tile')
except (ImportError, AttributeError):
    NATIVE_AVAILABLE = False


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
def test_render_brdf_tile_in_all():
    """Test that render_brdf_tile is exported in __all__."""
    assert hasattr(f3d, 'render_brdf_tile'), "render_brdf_tile should be available"
    assert 'render_brdf_tile' in f3d.__all__, "render_brdf_tile should be in __all__"


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_exit_criteria():
    """P7-06 Exit Criteria: f3d.render_brdf_tile("ggx", 0.5, 256, 256, True) returns (256,256,4) uint8."""
    result = f3d.render_brdf_tile("ggx", 0.5, 256, 256, True)
    
    # Exit criteria checks
    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert result.shape == (256, 256, 4), f"Expected (256, 256, 4), got {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    
    # Additional validation
    assert result.sum() > 0, "Result should not be all zeros"


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_all_models():
    """Test that the shim works for all supported BRDF models."""
    models = ["lambert", "phong", "ggx", "disney"]
    
    for model in models:
        result = f3d.render_brdf_tile(model, 0.5, 64, 64, False)
        assert result.shape == (64, 64, 4), f"Model {model} failed"
        assert result.dtype == np.uint8, f"Model {model} wrong dtype"


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_ndf_mode():
    """Test NDF-only mode through the Python shim."""
    # NDF-only mode
    result = f3d.render_brdf_tile("ggx", 0.3, 64, 64, ndf_only=True)
    assert result.shape == (64, 64, 4)
    
    # Full BRDF mode
    result2 = f3d.render_brdf_tile("ggx", 0.3, 64, 64, ndf_only=False)
    assert result2.shape == (64, 64, 4)


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_error_propagation():
    """Test that errors from native module are properly propagated."""
    # Invalid model should raise error
    with pytest.raises((RuntimeError, ValueError)):
        f3d.render_brdf_tile("invalid_model", 0.5, 64, 64, False)


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
@pytest.mark.skipif(NATIVE_AVAILABLE, reason="Test only when native unavailable")
def test_render_brdf_tile_no_native_error():
    """Test that calling without native module raises clear RuntimeError."""
    # When native module is not available, should raise RuntimeError
    with pytest.raises(RuntimeError, match="requires the native module"):
        f3d.render_brdf_tile("ggx", 0.5, 64, 64, False)


@pytest.mark.skipif(not TOP_LEVEL_IMPORT_OK, reason="forge3d not importable")
@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_png_workflow():
    """Test complete workflow: render + save as PNG."""
    # Render a tile
    tile = f3d.render_brdf_tile("ggx", 0.5, 128, 128, False)
    
    # Verify it's ready for PNG export
    assert tile.shape == (128, 128, 4)
    assert tile.dtype == np.uint8
    assert tile.flags['C_CONTIGUOUS'], "Array should be C-contiguous for PNG export"
    
    # Note: Actual PNG save would use f3d.numpy_to_png("output.png", tile)
    # but we skip the file I/O in the test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
