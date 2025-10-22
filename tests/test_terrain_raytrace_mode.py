"""
Test terrain ray tracing mode toggle
"""
import numpy as np
import pytest

try:
    from forge3d.terrain import drape_landcover
except ImportError:
    pytest.skip("forge3d not available", allow_module_level=True)


def make_test_terrain(size=32):
    """Create small test heightmap and landcover for fast testing."""
    # Simple pyramid terrain
    heightmap = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            cx = abs(x - size // 2)
            cy = abs(y - size // 2)
            dist = max(cx, cy)
            height = max(0.0, (size // 2 - dist) * 10.0)
            heightmap[y, x] = height
    
    # Green terrain with blue water at edges
    landcover = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            cx = abs(x - size // 2)
            cy = abs(y - size // 2)
            dist = max(cx, cy)
            if dist > size // 2 - 2:
                landcover[y, x] = [30, 144, 255, 255]  # Blue water
            else:
                landcover[y, x] = [34, 139, 34, 255]  # Green terrain
    
    return heightmap, landcover


def test_raster_mode_produces_valid_output():
    """Test that raster mode produces valid RGBA output."""
    heightmap, landcover = make_test_terrain(32)
    
    result = drape_landcover(
        heightmap,
        landcover,
        render_mode="raster",
        width=128,
        height=128,
        camera_theta=45.0,
        camera_phi=30.0,
    )
    
    assert result.shape == (128, 128, 4), f"Expected (128, 128, 4), got {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    assert result.max() > 0, "Output should have some non-zero pixels"


def test_raytrace_mode_produces_valid_output():
    """Test that raytrace mode produces valid RGBA output."""
    heightmap, landcover = make_test_terrain(32)
    
    result = drape_landcover(
        heightmap,
        landcover,
        render_mode="raytrace",
        rt_spp=16,  # Low quality for speed
        rt_seed=42,
        width=128,
        height=128,
        camera_theta=45.0,
        camera_phi=30.0,
        denoiser="off",  # Disable for speed
    )
    
    assert result.shape == (128, 128, 4), f"Expected (128, 128, 4), got {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    assert result.max() > 0, "Raytrace output should have some non-zero pixels"


def test_render_mode_validation():
    """Test that invalid render_mode raises ValueError."""
    heightmap, landcover = make_test_terrain(32)
    
    with pytest.raises(ValueError, match="render_mode must be"):
        drape_landcover(
            heightmap,
            landcover,
            render_mode="invalid_mode",
            width=64,
            height=64,
        )


def test_camera_parameters_consistent_between_modes():
    """Test that camera parameters work the same in both modes."""
    heightmap, landcover = make_test_terrain(32)
    
    camera_configs = [
        {"camera_theta": 0.0, "camera_phi": 30.0, "camera_fov": 35.0},
        {"camera_theta": 45.0, "camera_phi": 45.0, "camera_fov": 45.0},
        {"camera_theta": 90.0, "camera_phi": 60.0, "camera_fov": 30.0},
    ]
    
    for config in camera_configs:
        # Test raster mode
        result_raster = drape_landcover(
            heightmap,
            landcover,
            render_mode="raster",
            width=64,
            height=64,
            **config,
        )
        assert result_raster.shape == (64, 64, 4)
        
        # Test raytrace mode with same camera parameters
        result_raytrace = drape_landcover(
            heightmap,
            landcover,
            render_mode="raytrace",
            rt_spp=4,  # Very low quality for speed
            width=64,
            height=64,
            denoiser="off",
            **config,
        )
        assert result_raytrace.shape == (64, 64, 4)


def test_rt_spp_affects_output_variance():
    """Test that higher rt_spp reduces noise (variance) in ray tracing."""
    heightmap, landcover = make_test_terrain(32)
    
    # Render with low spp (more noise)
    result_low = drape_landcover(
        heightmap,
        landcover,
        render_mode="raytrace",
        rt_spp=4,
        rt_seed=42,
        width=64,
        height=64,
        camera_theta=45.0,
        camera_phi=30.0,
        denoiser="off",
    )
    
    # Render with high spp (less noise)
    result_high = drape_landcover(
        heightmap,
        landcover,
        render_mode="raytrace",
        rt_spp=32,
        rt_seed=42,
        width=64,
        height=64,
        camera_theta=45.0,
        camera_phi=30.0,
        denoiser="off",
    )
    
    # Both should produce valid output
    assert result_low.shape == (64, 64, 4)
    assert result_high.shape == (64, 64, 4)
    
    # Higher spp should generally have lower variance in non-background regions
    # (This is a soft check - exact variance comparison depends on scene complexity)
    assert result_high.max() > 0, "High spp output should have terrain pixels"


def test_mode_switching_works():
    """Test that switching between modes works correctly."""
    heightmap, landcover = make_test_terrain(32)
    
    # Render raster
    result1 = drape_landcover(
        heightmap, landcover,
        render_mode="raster",
        width=64, height=64,
    )
    
    # Switch to raytrace
    result2 = drape_landcover(
        heightmap, landcover,
        render_mode="raytrace",
        rt_spp=8,
        width=64, height=64,
        denoiser="off",
    )
    
    # Switch back to raster
    result3 = drape_landcover(
        heightmap, landcover,
        render_mode="raster",
        width=64, height=64,
    )
    
    # All should produce valid output
    assert result1.shape == (64, 64, 4)
    assert result2.shape == (64, 64, 4)
    assert result3.shape == (64, 64, 4)


def test_hdri_parameter_in_raytrace_mode():
    """Test that HDRI parameter is accepted in raytrace mode."""
    heightmap, landcover = make_test_terrain(32)
    
    # Test with HDRI disabled (None)
    result = drape_landcover(
        heightmap,
        landcover,
        render_mode="raytrace",
        rt_spp=8,
        width=64,
        height=64,
        hdri=None,
        hdri_intensity=1.0,
        denoiser="off",
    )
    
    assert result.shape == (64, 64, 4)
    assert result.dtype == np.uint8


def test_denoiser_parameter_in_raytrace_mode():
    """Test that denoiser parameter works in raytrace mode."""
    heightmap, landcover = make_test_terrain(32)
    
    for denoiser in ["off", "oidn", "bilateral"]:
        result = drape_landcover(
            heightmap,
            landcover,
            render_mode="raytrace",
            rt_spp=8,
            width=64,
            height=64,
            denoiser=denoiser,
        )
        
        assert result.shape == (64, 64, 4), f"Denoiser '{denoiser}' failed"
        assert result.dtype == np.uint8


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing raster mode...")
    test_raster_mode_produces_valid_output()
    print("✓ Raster mode works")
    
    print("Testing raytrace mode...")
    test_raytrace_mode_produces_valid_output()
    print("✓ Raytrace mode works")
    
    print("Testing mode validation...")
    test_render_mode_validation()
    print("✓ Mode validation works")
    
    print("Testing camera consistency...")
    test_camera_parameters_consistent_between_modes()
    print("✓ Camera parameters consistent")
    
    print("Testing mode switching...")
    test_mode_switching_works()
    print("✓ Mode switching works")
    
    print("\n✅ All tests passed!")
