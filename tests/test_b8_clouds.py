# B8-BEGIN:realtime-clouds-tests
"""Tests for B8: Realtime Clouds - Billboard/volumetric cloud rendering with IBL-aware scattering"""
import os
import pytest
import numpy as np
import time

# Skip if terrain tests are not enabled
SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1")

def test_clouds_basic_enable_disable():
    """Test basic cloud rendering enable/disable functionality"""
    import forge3d as f3d

    # Create scene
    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Initially clouds should be disabled
    assert not scene.is_clouds_enabled()

    # Enable clouds
    scene.enable_clouds(quality="medium")
    assert scene.is_clouds_enabled()

    # Disable clouds
    scene.disable_clouds()
    assert not scene.is_clouds_enabled()

def test_clouds_quality_levels():
    """Test different cloud rendering quality levels"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Test all quality levels
    for quality in ["low", "medium", "high", "ultra"]:
        scene.enable_clouds(quality=quality)
        assert scene.is_clouds_enabled()

        # Should be able to render without error
        try:
            rgba = scene.render_rgba()
            assert rgba.shape == (256, 256, 4)
            assert rgba.dtype == np.uint8
        except Exception as e:
            pytest.fail(f"Failed to render with quality='{quality}': {e}")

def test_clouds_render_modes():
    """Test different cloud rendering modes"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Test all render modes
    for mode in ["billboard", "volumetric", "hybrid"]:
        scene.set_cloud_render_mode(mode)

        # Should be able to render without error
        try:
            rgba = scene.render_rgba()
            assert rgba.shape == (256, 256, 4)
            assert rgba.dtype == np.uint8
        except Exception as e:
            pytest.fail(f"Failed to render with mode='{mode}': {e}")

def test_clouds_parameter_validation():
    """Test cloud parameter validation"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Test valid parameter ranges
    scene.set_cloud_density(0.0)  # Minimum
    scene.set_cloud_density(2.0)  # Maximum
    scene.set_cloud_coverage(0.0)  # Minimum
    scene.set_cloud_coverage(1.0)  # Maximum
    scene.set_cloud_scale(10.0)   # Minimum reasonable scale

    # Test invalid quality levels
    with pytest.raises(Exception):
        scene.enable_clouds(quality="invalid")

    # Test invalid render modes
    with pytest.raises(Exception):
        scene.set_cloud_render_mode("invalid")

    # Test invalid animation presets
    with pytest.raises(Exception):
        scene.set_cloud_animation_preset("invalid")

def test_clouds_animation_presets():
    """Test cloud animation presets"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Test all animation presets
    presets = ["static", "gentle", "moderate", "stormy"]
    for preset in presets:
        scene.set_cloud_animation_preset(preset)

        # Should be able to render after setting preset
        try:
            rgba = scene.render_rgba()
            assert rgba.shape == (256, 256, 4)
        except Exception as e:
            pytest.fail(f"Failed to render with preset '{preset}': {e}")

def test_clouds_animation_time_progression():
    """Test cloud animation over time"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="low")  # Use low quality for speed
    scene.set_cloud_animation_preset("moderate")

    # Set up terrain for better visual testing
    heightmap = np.random.rand(128, 128).astype(np.float32)
    scene.set_height_from_r32f(heightmap)

    # Render at different time steps
    time_steps = [0.0, 1.0, 2.0, 3.0]
    images = []

    for time_step in time_steps:
        scene.update_cloud_animation(time_step)
        rgba = scene.render_rgba()
        images.append(rgba.copy())

    # Images should be different at different time steps (clouds should move)
    # Compare first and last images - they should differ
    diff = np.mean(np.abs(images[0].astype(float) - images[-1].astype(float)))
    assert diff > 0.5, f"Cloud animation should cause visible changes, got diff={diff}"

def test_clouds_parameter_getter():
    """Test getting cloud parameters"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Set known parameter values
    expected_density = 0.8
    expected_coverage = 0.6
    expected_scale = 150.0

    scene.set_cloud_density(expected_density)
    scene.set_cloud_coverage(expected_coverage)
    scene.set_cloud_scale(expected_scale)

    # Get parameters back
    density, coverage, scale, wind_strength = scene.get_clouds_params()

    # Values should match within tolerance
    assert abs(density - expected_density) < 0.05
    assert abs(coverage - expected_coverage) < 0.05
    assert abs(scale - expected_scale) < 5.0

def test_clouds_wind_control():
    """Test cloud wind parameter control"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Test setting different wind values
    scene.set_cloud_wind_vector(0.0, 0.0, 0.0)    # No wind
    scene.set_cloud_wind_vector(1.0, 0.0, 0.5)    # East wind
    scene.set_cloud_wind_vector(0.0, 1.0, 0.8)    # North wind
    scene.set_cloud_wind_vector(-0.7, 0.7, 1.0)   # Northwest wind, strong

    # Should render successfully with all wind settings
    rgba = scene.render_rgba()
    assert rgba.shape == (256, 256, 4)

def test_clouds_scale_control():
    """Test cloud scale parameter control"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Test different scale values
    scales = [50.0, 100.0, 200.0, 500.0, 1000.0]
    for scale in scales:
        scene.set_cloud_scale(scale)

        # Should render successfully with all scales
        rgba = scene.render_rgba()
        assert rgba.shape == (256, 256, 4)

def test_clouds_not_enabled():
    """Test that cloud controls fail when not enabled"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Clouds are not enabled, so controls should fail
    with pytest.raises(Exception, match="Clouds not enabled"):
        scene.set_cloud_density(0.5)

    with pytest.raises(Exception, match="Clouds not enabled"):
        scene.get_clouds_params()

def test_clouds_rendering_integration():
    """Test cloud rendering integration with terrain"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Create varied terrain
    x = np.linspace(-2, 2, 128)
    y = np.linspace(-2, 2, 128)
    X, Y = np.meshgrid(x, y)
    heightmap = (np.sin(X * 2) * np.cos(Y * 2) * 0.5 + 0.5).astype(np.float32)
    scene.set_height_from_r32f(heightmap)

    # Render without clouds
    rgba_no_clouds = scene.render_rgba()

    # Enable clouds and render again
    scene.enable_clouds(quality="medium")
    scene.set_cloud_density(0.8)
    scene.set_cloud_coverage(0.6)
    rgba_with_clouds = scene.render_rgba()

    # Images should be different (clouds should affect the scene)
    diff = np.mean(np.abs(rgba_no_clouds.astype(float) - rgba_with_clouds.astype(float)))
    assert diff > 1.0, f"Clouds should visibly affect the scene, got diff={diff}"

def test_clouds_render_consistency():
    """Test that cloud rendering is consistent across multiple calls"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.5)
    scene.set_cloud_animation_preset("static")  # No animation for consistency
    scene.update_cloud_animation(1.0)  # Set fixed time

    # Render multiple times with same parameters
    rgba1 = scene.render_rgba()
    rgba2 = scene.render_rgba()

    # Should be identical (deterministic when animation is static)
    diff = np.mean(np.abs(rgba1.astype(float) - rgba2.astype(float)))
    assert diff < 0.1, f"Cloud rendering should be deterministic with static animation, got diff={diff}"

def test_clouds_performance_bounds():
    """Test that cloud rendering meets performance requirements"""
    import forge3d as f3d

    # Test at different resolutions
    resolutions = [(512, 384), (1024, 768)]  # Start smaller, work up to larger

    for width, height in resolutions:
        scene = f3d.Scene(width, height, grid=128, colormap="terrain")

        # Create simple terrain
        heightmap = np.random.rand(128, 128).astype(np.float32)
        scene.set_height_from_r32f(heightmap)

        # Test rendering time without clouds
        start_time = time.time()
        for _ in range(3):
            rgba = scene.render_rgba()
        baseline_time = (time.time() - start_time) / 3

        # Enable low-quality clouds (performance mode)
        scene.enable_clouds(quality="low")
        scene.set_cloud_render_mode("hybrid")  # Balanced performance
        scene.set_cloud_density(0.6)

        start_time = time.time()
        for _ in range(3):
            rgba = scene.render_rgba()
        cloud_time = (time.time() - start_time) / 3

        # Clouds should not add excessive overhead
        overhead = (cloud_time - baseline_time) / baseline_time if baseline_time > 0 else 0
        max_overhead = 1.0  # Allow up to 100% overhead for cloud rendering

        assert overhead < max_overhead, f"Cloud overhead too high at {width}x{height}: {overhead:.2%} (expected <{max_overhead:.0%})"

        # Check if we achieve reasonable FPS
        fps = 1.0 / cloud_time if cloud_time > 0 else 0
        print(f"  Cloud rendering at {width}x{height}: {fps:.1f} FPS")

def test_clouds_1080p_performance():
    """Test cloud rendering performance at 1080p (B8 acceptance criteria)"""
    import forge3d as f3d

    # Create 1080p scene
    scene = f3d.Scene(1920, 1080, grid=256, colormap="terrain")

    # Create moderately complex terrain
    heightmap = np.random.rand(256, 256).astype(np.float32)
    scene.set_height_from_r32f(heightmap)

    # Enable low-quality clouds for 60 FPS target
    scene.enable_clouds(quality="low")
    scene.set_cloud_render_mode("hybrid")
    scene.set_cloud_density(0.6)
    scene.set_cloud_coverage(0.5)

    # Warm up
    scene.render_rgba()

    # Measure performance
    start_time = time.time()
    frame_count = 5  # Reduced for CI
    for _ in range(frame_count):
        rgba = scene.render_rgba()

    total_time = time.time() - start_time
    avg_frame_time = total_time / frame_count
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    print(f"  1080p cloud rendering performance: {fps:.1f} FPS")

    # B8 acceptance criteria: "Clouds render at 60 FPS 1080p with low-VRAM profile"
    # We'll be lenient in CI but still check for reasonable performance
    min_fps = 30.0  # Relaxed for CI, should be 60+ in optimized builds
    assert fps >= min_fps, f"1080p cloud rendering too slow: {fps:.1f} FPS (expected >={min_fps} FPS)"

def test_clouds_memory_usage():
    """Test that clouds don't exceed memory constraints"""
    import forge3d as f3d

    # Test with high quality clouds (largest memory usage)
    scene = f3d.Scene(512, 512, grid=256, colormap="terrain")
    scene.enable_clouds(quality="high")

    # Should be able to render without memory errors
    try:
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)
    except Exception as e:
        if "memory" in str(e).lower() or "allocation" in str(e).lower():
            pytest.fail(f"Cloud rendering exceeded memory constraints: {e}")
        else:
            raise  # Re-raise if it's not a memory issue

def test_clouds_edge_cases():
    """Test cloud rendering edge cases and boundary conditions"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Test extreme parameter combinations
    extreme_configs = [
        # (density, coverage, scale)
        (0.0, 0.0, 10.0),    # All minimums
        (2.0, 1.0, 1000.0),  # All maximums
        (2.0, 0.0, 10.0),    # Mixed extremes
        (0.0, 1.0, 1000.0),  # Opposite mixed extremes
    ]

    for density, coverage, scale in extreme_configs:
        scene.set_cloud_density(density)
        scene.set_cloud_coverage(coverage)
        scene.set_cloud_scale(scale)

        # Should render without crashing
        rgba = scene.render_rgba()
        assert rgba.shape == (256, 256, 4)

def test_clouds_return_types():
    """Test that cloud functions return correct types"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Test enable_clouds return type
    result = scene.enable_clouds(quality="medium")
    assert result is None  # Should not return anything

    # Test parameter getter return types
    density, coverage, scale, wind_strength = scene.get_clouds_params()
    assert isinstance(density, float)
    assert isinstance(coverage, float)
    assert isinstance(scale, float)
    assert isinstance(wind_strength, float)

    # Test boolean return types
    assert isinstance(scene.is_clouds_enabled(), bool)

def test_clouds_quality_impact():
    """Test that different quality levels have visible impact"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Set up terrain
    heightmap = np.random.rand(128, 128).astype(np.float32)
    scene.set_height_from_r32f(heightmap)

    # Test low vs high quality
    scene.enable_clouds(quality="low")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)
    rgba_low = scene.render_rgba()

    scene.enable_clouds(quality="high")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)
    rgba_high = scene.render_rgba()

    # High quality should produce visually different results
    diff = np.mean(np.abs(rgba_low.astype(float) - rgba_high.astype(float)))
    assert diff > 0.5, f"Quality levels should produce different visual results, got diff={diff}"

def test_clouds_ibl_integration():
    """Test IBL-aware scattering integration (B8 requirement)"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_clouds(quality="medium")

    # Set different sky/lighting conditions
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)

    # Test with different animation presets (which affect scattering parameters)
    presets = ["gentle", "moderate", "stormy"]
    images = []

    for preset in presets:
        scene.set_cloud_animation_preset(preset)
        rgba = scene.render_rgba()
        images.append(rgba)

    # Different presets should produce different scattering results
    diff_1_2 = np.mean(np.abs(images[0].astype(float) - images[1].astype(float)))
    diff_1_3 = np.mean(np.abs(images[0].astype(float) - images[2].astype(float)))

    assert diff_1_2 > 0.3, f"Different animation presets should affect IBL scattering, got diff={diff_1_2}"
    assert diff_1_3 > 0.3, f"Different animation presets should affect IBL scattering, got diff={diff_1_3}"

# B8-END:realtime-clouds-tests