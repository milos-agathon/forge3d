# B7-BEGIN:cloud-shadows-tests
"""Tests for B7: Cloud Shadow Overlay - 2D shadow texture modulation over terrain"""
import os
import pytest
import numpy as np

# Skip if terrain tests are not enabled
SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1")

def test_cloud_shadows_basic_enable_disable():
    """Test basic cloud shadow enable/disable functionality"""
    import forge3d as f3d

    # Create scene with terrain
    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Initially cloud shadows should be disabled
    assert not scene.is_cloud_shadows_enabled()

    # Enable cloud shadows
    scene.enable_cloud_shadows(quality="low")
    assert scene.is_cloud_shadows_enabled()

    # Disable cloud shadows
    scene.disable_cloud_shadows()
    assert not scene.is_cloud_shadows_enabled()

def test_cloud_shadows_quality_levels():
    """Test different cloud shadow quality levels"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Test all quality levels
    for quality in ["low", "medium", "high"]:
        scene.enable_cloud_shadows(quality=quality)
        assert scene.is_cloud_shadows_enabled()

        # Should be able to render without error
        try:
            rgba = scene.render_rgba()
            assert rgba.shape == (256, 256, 4)
            assert rgba.dtype == np.uint8
        except Exception as e:
            pytest.fail(f"Failed to render with quality='{quality}': {e}")

def test_cloud_shadows_parameter_validation():
    """Test cloud shadow parameter validation"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Test valid parameter ranges
    scene.set_cloud_density(0.0)  # Minimum
    scene.set_cloud_density(1.0)  # Maximum
    scene.set_cloud_coverage(0.0)  # Minimum
    scene.set_cloud_coverage(1.0)  # Maximum
    scene.set_cloud_shadow_intensity(0.0)  # Minimum
    scene.set_cloud_shadow_intensity(1.0)  # Maximum
    scene.set_cloud_shadow_softness(0.0)  # Minimum
    scene.set_cloud_shadow_softness(1.0)  # Maximum

    # Test invalid parameter ranges
    with pytest.raises(Exception):
        scene.set_cloud_density(-0.1)  # Below minimum
    with pytest.raises(Exception):
        scene.set_cloud_density(1.1)   # Above maximum
    with pytest.raises(Exception):
        scene.set_cloud_coverage(-0.1)  # Below minimum
    with pytest.raises(Exception):
        scene.set_cloud_coverage(1.1)   # Above maximum

def test_cloud_shadows_animation_presets():
    """Test cloud animation presets"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Test all animation presets
    presets = ["calm", "windy", "stormy"]
    for preset in presets:
        scene.set_cloud_animation_preset(preset)

        # Should be able to render after setting preset
        try:
            rgba = scene.render_rgba()
            assert rgba.shape == (256, 256, 4)
        except Exception as e:
            pytest.fail(f"Failed to render with preset '{preset}': {e}")

def test_cloud_shadows_animation_time_progression():
    """Test cloud shadow animation over time"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="low")  # Use low quality for speed
    scene.set_cloud_animation_preset("windy")

    # Set up a simple terrain with variation
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
    # Compare first and last images - they should differ significantly
    diff = np.mean(np.abs(images[0].astype(float) - images[-1].astype(float)))
    assert diff > 1.0, f"Cloud animation should cause visible changes, got diff={diff}"

def test_cloud_shadows_parameter_getter():
    """Test getting cloud shadow parameters"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Set known parameter values
    expected_density = 0.7
    expected_coverage = 0.4
    expected_intensity = 0.8
    expected_softness = 0.3

    scene.set_cloud_density(expected_density)
    scene.set_cloud_coverage(expected_coverage)
    scene.set_cloud_shadow_intensity(expected_intensity)
    scene.set_cloud_shadow_softness(expected_softness)

    # Get parameters back
    density, coverage, intensity, softness = scene.get_cloud_params()

    # Values should match within tolerance
    assert abs(density - expected_density) < 0.01
    assert abs(coverage - expected_coverage) < 0.01
    assert abs(intensity - expected_intensity) < 0.01
    assert abs(softness - expected_softness) < 0.01

def test_cloud_shadows_speed_control():
    """Test cloud speed parameter control"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Test setting different speed values
    scene.set_cloud_speed(0.0, 0.0)    # No movement
    scene.set_cloud_speed(0.1, 0.05)   # Moderate movement
    scene.set_cloud_speed(-0.05, 0.1)  # Negative/positive combination

    # Should render successfully with all speed settings
    rgba = scene.render_rgba()
    assert rgba.shape == (256, 256, 4)

def test_cloud_shadows_scale_control():
    """Test cloud scale parameter control"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Test different scale values
    scales = [0.5, 1.0, 2.0, 4.0]
    for scale in scales:
        scene.set_cloud_scale(scale)

        # Should render successfully with all scales
        rgba = scene.render_rgba()
        assert rgba.shape == (256, 256, 4)

def test_cloud_shadows_wind_effects():
    """Test cloud wind parameter control"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Test different wind directions and strengths
    wind_configs = [
        (0.0, 0.0),     # No wind
        (45.0, 1.0),    # Northeast wind, moderate strength
        (180.0, 2.0),   # South wind, strong
        (270.0, 0.5),   # West wind, light
    ]

    for direction, strength in wind_configs:
        scene.set_cloud_wind(direction, strength)

        # Should render successfully with all wind settings
        rgba = scene.render_rgba()
        assert rgba.shape == (256, 256, 4)

def test_cloud_shadows_debug_mode():
    """Test cloud shadow debug visualization"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Enable clouds-only mode
    scene.set_cloud_show_clouds_only(True)
    rgba_clouds_only = scene.render_rgba()

    # Disable clouds-only mode
    scene.set_cloud_show_clouds_only(False)
    rgba_normal = scene.render_rgba()

    # Images should be different
    diff = np.mean(np.abs(rgba_clouds_only.astype(float) - rgba_normal.astype(float)))
    assert diff > 10.0, f"Clouds-only mode should produce different image, got diff={diff}"

def test_cloud_shadows_quality_not_enabled():
    """Test that cloud shadow controls fail when not enabled"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Cloud shadows are not enabled, so controls should fail
    with pytest.raises(Exception, match="Cloud shadows are not enabled"):
        scene.set_cloud_density(0.5)

    with pytest.raises(Exception, match="Cloud shadows are not enabled"):
        scene.get_cloud_params()

def test_cloud_shadows_terrain_integration():
    """Test cloud shadows integration with terrain rendering"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Create varied terrain
    x = np.linspace(-2, 2, 128)
    y = np.linspace(-2, 2, 128)
    X, Y = np.meshgrid(x, y)
    heightmap = (np.sin(X * 2) * np.cos(Y * 2) * 0.5 + 0.5).astype(np.float32)
    scene.set_height_from_r32f(heightmap)

    # Render without cloud shadows
    rgba_no_clouds = scene.render_rgba()

    # Enable cloud shadows and render again
    scene.enable_cloud_shadows(quality="medium")
    scene.set_cloud_density(0.8)
    scene.set_cloud_shadow_intensity(0.7)
    rgba_with_clouds = scene.render_rgba()

    # Images should be different (cloud shadows should affect lighting)
    diff = np.mean(np.abs(rgba_no_clouds.astype(float) - rgba_with_clouds.astype(float)))
    assert diff > 2.0, f"Cloud shadows should visibly affect terrain, got diff={diff}"

def test_cloud_shadows_render_consistency():
    """Test that cloud shadow rendering is consistent across multiple calls"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")
    scene.set_cloud_density(0.6)
    scene.set_cloud_coverage(0.5)
    scene.update_cloud_animation(1.0)  # Set fixed time

    # Render multiple times with same parameters
    rgba1 = scene.render_rgba()
    rgba2 = scene.render_rgba()

    # Should be identical (deterministic)
    diff = np.mean(np.abs(rgba1.astype(float) - rgba2.astype(float)))
    assert diff < 0.1, f"Cloud shadow rendering should be deterministic, got diff={diff}"

def test_cloud_shadows_performance_bounds():
    """Test that cloud shadow rendering meets performance requirements"""
    import forge3d as f3d
    import time

    scene = f3d.Scene(512, 512, grid=256, colormap="terrain")

    # Create moderately complex terrain
    heightmap = np.random.rand(256, 256).astype(np.float32)
    scene.set_height_from_r32f(heightmap)

    # Test rendering time without cloud shadows
    start_time = time.time()
    for _ in range(3):
        rgba = scene.render_rgba()
    baseline_time = (time.time() - start_time) / 3

    # Enable cloud shadows and test rendering time
    scene.enable_cloud_shadows(quality="medium")
    scene.set_cloud_density(0.7)

    start_time = time.time()
    for _ in range(3):
        rgba = scene.render_rgba()
    cloud_time = (time.time() - start_time) / 3

    # Cloud shadows should not add more than 50% overhead
    overhead = (cloud_time - baseline_time) / baseline_time
    assert overhead < 0.5, f"Cloud shadow overhead too high: {overhead:.2%} (expected <50%)"

def test_cloud_shadows_memory_usage():
    """Test that cloud shadows don't exceed memory constraints"""
    import forge3d as f3d

    # Test with high quality clouds (largest memory usage)
    scene = f3d.Scene(512, 512, grid=256, colormap="terrain")
    scene.enable_cloud_shadows(quality="high")

    # Should be able to render without memory errors
    try:
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)
    except Exception as e:
        if "memory" in str(e).lower() or "allocation" in str(e).lower():
            pytest.fail(f"Cloud shadows exceeded memory constraints: {e}")
        else:
            raise  # Re-raise if it's not a memory issue

def test_cloud_shadows_edge_cases():
    """Test cloud shadow edge cases and boundary conditions"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")
    scene.enable_cloud_shadows(quality="medium")

    # Test extreme parameter combinations
    extreme_configs = [
        # (density, coverage, intensity, softness)
        (0.0, 0.0, 0.0, 0.0),  # All minimums
        (1.0, 1.0, 1.0, 1.0),  # All maximums
        (1.0, 0.0, 1.0, 0.0),  # Mixed extremes
        (0.0, 1.0, 0.0, 1.0),  # Opposite mixed extremes
    ]

    for density, coverage, intensity, softness in extreme_configs:
        scene.set_cloud_density(density)
        scene.set_cloud_coverage(coverage)
        scene.set_cloud_shadow_intensity(intensity)
        scene.set_cloud_shadow_softness(softness)

        # Should render without crashing
        rgba = scene.render_rgba()
        assert rgba.shape == (256, 256, 4)

def test_cloud_shadows_return_types():
    """Test that cloud shadow functions return correct types"""
    import forge3d as f3d

    scene = f3d.Scene(256, 256, grid=128, colormap="terrain")

    # Test enable_cloud_shadows return type
    result = scene.enable_cloud_shadows(quality="medium")
    assert result is None  # Should not return anything

    # Test parameter getter return types
    density, coverage, intensity, softness = scene.get_cloud_params()
    assert isinstance(density, float)
    assert isinstance(coverage, float)
    assert isinstance(intensity, float)
    assert isinstance(softness, float)

    # Test boolean return types
    assert isinstance(scene.is_cloud_shadows_enabled(), bool)

# B7-END:cloud-shadows-tests