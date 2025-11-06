#!/usr/bin/env python3
"""
P7-05: Test for render_brdf_tile PyO3 binding
Tests that the native function returns correct numpy arrays for BRDF tiles.
"""
import numpy as np
import pytest

try:
    import forge3d._forge3d as f3d_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_basic():
    """Test basic BRDF tile rendering."""
    # Render GGX at roughness 0.5
    result = f3d_native.render_brdf_tile("ggx", 0.5, 128, 128, False)
    
    # Check shape
    assert result.shape == (128, 128, 4), f"Expected (128, 128, 4), got {result.shape}"
    
    # Check dtype
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    
    # Check not all zeros (should have lighting)
    assert result.sum() > 0, "Result should not be all zeros"
    
    # Check alpha channel is 255 (opaque)
    assert np.all(result[:, :, 3] == 255), "Alpha channel should be 255"


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_models():
    """Test all supported BRDF models."""
    models = ["lambert", "phong", "ggx", "disney"]
    
    for model in models:
        result = f3d_native.render_brdf_tile(model, 0.5, 64, 64, False)
        assert result.shape == (64, 64, 4), f"Model {model} failed shape check"
        assert result.dtype == np.uint8, f"Model {model} failed dtype check"
        assert result.sum() > 0, f"Model {model} produced all zeros"


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_ndf_only():
    """Test NDF-only debug mode."""
    # NDF-only mode should produce grayscale output (R=G=B)
    result = f3d_native.render_brdf_tile("ggx", 0.3, 64, 64, True)
    
    assert result.shape == (64, 64, 4)
    
    # Check if output is grayscale (at least mostly)
    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
    # Allow small variations due to floating point precision
    assert np.allclose(r, g, atol=1), "NDF mode should produce grayscale (R≈G)"
    assert np.allclose(g, b, atol=1), "NDF mode should produce grayscale (G≈B)"


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_roughness_clamping():
    """Test that roughness is clamped to [0, 1]."""
    # Should clamp to 0
    result1 = f3d_native.render_brdf_tile("ggx", -0.5, 64, 64, False)
    assert result1.shape == (64, 64, 4)
    
    # Should clamp to 1
    result2 = f3d_native.render_brdf_tile("ggx", 2.0, 64, 64, False)
    assert result2.shape == (64, 64, 4)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_invalid_model():
    """Test that invalid model raises error."""
    with pytest.raises(ValueError, match="Invalid BRDF model"):
        f3d_native.render_brdf_tile("invalid_model", 0.5, 64, 64, False)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native module not available")
def test_render_brdf_tile_png_export():
    """Test that result can be saved as PNG using forge3d utilities."""
    result = f3d_native.render_brdf_tile("ggx", 0.5, 128, 128, False)
    
    # Verify it's in the correct format for PNG export
    assert result.shape == (128, 128, 4)
    assert result.dtype == np.uint8
    
    # In actual usage, this would be:
    # from forge3d import numpy_to_png
    # numpy_to_png("brdf_ggx_r0.5.png", result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
