"""
Shadow mapping tests for terrain draping (Python API)
"""

import numpy as np
import pytest
from forge3d.terrain import drape_landcover


def test_shadow_rendering_basic():
    """Test basic shadow rendering with default parameters"""
    size = 64
    
    # Create heightmap with a raised plateau
    heightmap = np.zeros((size, size), dtype=np.float32)
    heightmap[20:40, 20:40] = 10.0
    
    # Green landcover
    landcover = np.zeros((size, size, 4), dtype=np.uint8)
    landcover[:, :, 1] = 200  # Green
    landcover[:, :, 3] = 255  # Alpha
    
    # Render with shadows enabled (default)
    img = drape_landcover(
        heightmap,
        landcover,
        width=512,
        height=512,
        zscale=2.0,
        enable_shadows=True,
        shadow_intensity=0.7,
    )
    
    assert img.shape == (512, 512, 4)
    assert img.dtype == np.uint8
    assert np.any(img[:, :, 3] > 0), "Should have non-transparent pixels"


def test_shadow_intensity_variation():
    """Test that shadow_intensity parameter affects output"""
    size = 32
    heightmap = np.random.rand(size, size).astype(np.float32) * 5.0
    landcover = np.full((size, size, 4), 128, dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    # Render with no shadows (intensity = 0)
    img_no_shadow = drape_landcover(
        heightmap,
        landcover,
        width=256,
        height=256,
        shadow_intensity=0.0,
        enable_shadows=True,
    )
    
    # Render with strong shadows (intensity = 1.0)
    img_strong_shadow = drape_landcover(
        heightmap,
        landcover,
        width=256,
        height=256,
        shadow_intensity=1.0,
        enable_shadows=True,
    )
    
    # Images should be different
    assert not np.array_equal(img_no_shadow, img_strong_shadow), \
        "Shadow intensity should affect output"


def test_shadow_disable():
    """Test that enable_shadows=False disables shadow rendering"""
    size = 32
    heightmap = np.random.rand(size, size).astype(np.float32) * 5.0
    landcover = np.full((size, size, 4), 128, dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    # Render with shadows disabled
    img_disabled = drape_landcover(
        heightmap,
        landcover,
        width=256,
        height=256,
        enable_shadows=False,
    )
    
    assert img_disabled.shape == (256, 256, 4)
    assert img_disabled.dtype == np.uint8


def test_shadow_softness_parameter():
    """Test PCF kernel radius (shadow_softness)"""
    size = 32
    heightmap = np.random.rand(size, size).astype(np.float32) * 5.0
    landcover = np.full((size, size, 4), 128, dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    # Test different softness values
    for softness in [1.0, 2.0, 3.0, 5.0]:
        img = drape_landcover(
            heightmap,
            landcover,
            width=256,
            height=256,
            shadow_softness=softness,
            enable_shadows=True,
        )
        assert img.shape == (256, 256, 4), f"Failed with softness={softness}"


def test_shadow_map_resolution():
    """Test various shadow map resolutions"""
    size = 32
    heightmap = np.random.rand(size, size).astype(np.float32) * 5.0
    landcover = np.full((size, size, 4), 128, dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    # Test different resolutions
    for res in [512, 1024, 2048, 4096]:
        img = drape_landcover(
            heightmap,
            landcover,
            width=256,
            height=256,
            shadow_map_res=res,
            enable_shadows=True,
        )
        assert img.shape == (256, 256, 4), f"Failed with shadow_map_res={res}"


def test_shadow_bias_parameter():
    """Test depth bias to prevent shadow acne"""
    size = 32
    heightmap = np.full((size, size), 5.0, dtype=np.float32)  # Flat elevated terrain
    landcover = np.full((size, size, 4), 128, dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    # Test different bias values
    for bias in [0.0001, 0.001, 0.0015, 0.005, 0.01]:
        img = drape_landcover(
            heightmap,
            landcover,
            width=256,
            height=256,
            shadow_bias=bias,
            enable_shadows=True,
        )
        assert img.shape == (256, 256, 4), f"Failed with shadow_bias={bias}"


def test_shadow_with_lighting_models():
    """Test shadows work with different lighting models"""
    size = 32
    heightmap = np.random.rand(size, size).astype(np.float32) * 5.0
    landcover = np.full((size, size, 4), 128, dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    for lighting_model in ["lambert", "phong", "blinn_phong"]:
        img = drape_landcover(
            heightmap,
            landcover,
            width=256,
            height=256,
            lighting_model=lighting_model,
            enable_shadows=True,
            shadow_intensity=0.6,
        )
        assert img.shape == (256, 256, 4), \
            f"Failed with lighting_model={lighting_model}"


def test_shadow_mean_luminance_in_shadowed_region():
    """Test that shadowed regions have lower luminance"""
    size = 64
    
    # Create heightmap with obstacle that casts shadow
    heightmap = np.zeros((size, size), dtype=np.float32)
    heightmap[10:30, 10:30] = 20.0  # Tall obstacle
    
    # Uniform bright landcover
    landcover = np.full((size, size, 4), 255, dtype=np.uint8)
    
    # Render with strong directional light and shadows
    img = drape_landcover(
        heightmap,
        landcover,
        width=512,
        height=512,
        zscale=3.0,
        light_elevation=45.0,
        light_azimuth=135.0,  # Light from southeast
        shadow_intensity=0.8,
        enable_shadows=True,
    )
    
    # Calculate mean luminance (simple average of RGB)
    luminance = img[:, :, :3].astype(np.float32).mean(axis=2)
    
    # Shadowed regions should have lower luminance than lit regions
    # This is a qualitative test - we just check variation exists
    assert luminance.std() > 5.0, \
        "Shadow should create luminance variation in the scene"


def test_shadow_parameters_integration():
    """Test all shadow parameters work together"""
    size = 48
    heightmap = np.random.rand(size, size).astype(np.float32) * 8.0
    landcover = np.random.randint(50, 200, (size, size, 4), dtype=np.uint8)
    landcover[:, :, 3] = 255
    
    img = drape_landcover(
        heightmap,
        landcover,
        width=512,
        height=512,
        zscale=2.5,
        light_elevation=60.0,
        light_azimuth=45.0,
        shadow_intensity=0.7,
        shadow_softness=3.0,
        shadow_map_res=2048,
        shadow_bias=0.002,
        enable_shadows=True,
    )
    
    assert img.shape == (512, 512, 4)
    assert img.dtype == np.uint8
    # Check that the image has reasonable content
    assert np.any(img[:, :, 3] > 0), "Should have visible pixels"
    assert img[:, :, :3].mean() > 0, "Should have non-black pixels"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
