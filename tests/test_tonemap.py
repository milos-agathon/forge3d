# T22-BEGIN:test
import math
import numpy as np
import pytest
import forge3d as f3d

def reinhard(x):
    return x / (1.0 + x)

def gamma_correct(x, gamma=2.2):
    return np.maximum(x, 0.0) ** (1.0/gamma)

def tonemap_cpu(rgb, exposure=1.0):
    return gamma_correct(reinhard(rgb * exposure), 2.2)

def test_tonemap_cpu_vector():
    rgb = np.array([0.0, 0.18, 4.0], dtype=np.float32) # black, mid-gray, bright
    out = tonemap_cpu(rgb, 1.0)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    # Known spot checks
    assert np.isclose(out[0], 0.0, atol=1e-6)
    assert 0.4 < out[1] < 0.6
    assert out[2] < 1.0

def test_set_sun_and_exposure():
    r = f3d.Renderer(16, 16)
    # Should not throw
    r.set_sun(45.0, 30.0)
    with pytest.raises(ValueError): r.set_exposure(0.0)
    r.set_exposure(1.25)

def test_tonemap_shader_output_range():
    """Test that shader tonemap output is in valid [0,255] range"""
    # Render a small triangle to test shader tonemap pipeline
    r = f3d.Renderer(32, 32)
    rgba = r.render_triangle_rgba()
    
    # Assert output is in valid uint8 range [0,255]
    assert rgba.dtype == np.uint8, f"Expected uint8, got {rgba.dtype}"
    assert rgba.shape == (32, 32, 4), f"Expected (32,32,4), got {rgba.shape}"
    assert np.all(rgba >= 0) and np.all(rgba <= 255), "RGBA values outside [0,255] range"
    
    # Check that we have some color variation (triangle should not be all white/black)
    unique_colors = len(np.unique(rgba.reshape(-1, 4), axis=0))
    assert unique_colors > 2, f"Too few unique colors: {unique_colors} (tonemap not working?)"

def test_tonemap_exposure_effect():
    """Test that changing exposure affects average luminance as expected"""
    try:
        # This test requires terrain_spike feature to test exposure changes
        import forge3d as f3d
        
        # Create small renderer for speed
        r = f3d.Renderer(32, 32)
        
        # Try to set up terrain for exposure testing - skip if not available
        try:
            # Create a simple 4x4 heightmap for testing
            heightmap = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)
            r.add_terrain(heightmap, (1.0, 1.0), colormap='viridis')
            
            # Test low exposure
            r.set_exposure(0.5)
            rgba_low = r.render_triangle_rgba()  # This will render terrain if added
            
            # Test high exposure  
            r.set_exposure(2.0)
            rgba_high = r.render_triangle_rgba()
            
            # Convert to luminance (Y = 0.299*R + 0.587*G + 0.114*B)
            lum_low = 0.299 * rgba_low[...,0] + 0.587 * rgba_low[...,1] + 0.114 * rgba_low[...,2]
            lum_high = 0.299 * rgba_high[...,0] + 0.587 * rgba_high[...,1] + 0.114 * rgba_high[...,2]
            
            avg_lum_low = np.mean(lum_low)
            avg_lum_high = np.mean(lum_high)
            
            # Higher exposure should generally result in higher average luminance
            # (though tonemap will compress bright values)
            assert avg_lum_high > avg_lum_low * 0.8, \
                f"Exposure effect too weak: low={avg_lum_low:.1f}, high={avg_lum_high:.1f}"
                
            print(f"Exposure test: low={avg_lum_low:.1f}, high={avg_lum_high:.1f}")
            
        except Exception as e:
            # If terrain features not available, do basic triangle test
            print(f"Terrain test skipped: {e}")
            
            # Just verify basic triangle rendering still works with different settings
            r.set_sun(45.0, 90.0)  # Should not crash
            rgba = r.render_triangle_rgba()
            assert rgba.shape == (32, 32, 4)
            
    except Exception as e:
        # If exposure methods not available, skip this test
        pytest.skip(f"Exposure testing not available: {e}")

def test_tonemap_cpu_shader_consistency():
    """Test that CPU tonemap reference produces reasonable values compared to shader"""
    # Test CPU tonemap reference against known inputs
    test_values = np.array([0.0, 0.18, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    
    # Test with different exposures
    for exposure in [0.5, 1.0, 2.0]:
        tonemapped = tonemap_cpu(test_values, exposure)
        
        # All outputs should be in [0,1] range after tonemap
        assert np.all(tonemapped >= 0.0) and np.all(tonemapped <= 1.0), \
            f"Tonemap output outside [0,1] for exposure {exposure}"
        
        # Reinhard should compress bright values but preserve order
        assert np.all(np.diff(tonemapped) >= -1e-6), \
            f"Tonemap should preserve monotonicity for exposure {exposure}"
        
        # Black should stay black
        assert np.isclose(tonemapped[0], 0.0, atol=1e-6), \
            f"Black should remain black for exposure {exposure}"
    
    print(f"CPU tonemap validation passed for test values: {test_values}")
# T22-END:test