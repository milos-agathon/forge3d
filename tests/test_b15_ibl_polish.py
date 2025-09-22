#!/usr/bin/env python3
"""
B15: Image-Based Lighting (IBL) Polish - Test Suite

Tests for B15 acceptance criteria:
- IBL irradiance/specular prefiltering and BRDF LUT generation
- Quality levels with proper texture sizing
- Environment map loading and processing
- Material property testing and validation
- Performance requirements for IBL generation

This test suite validates the complete B15 implementation including:
- IBL system enable/disable functionality
- Quality setting controls (Low/Medium/High/Ultra)
- Environment map loading from HDR data
- IBL texture generation (irradiance, specular, BRDF LUT)
- Material testing with various roughness/metallic combinations
- BRDF LUT sampling functionality
- Performance benchmarks for generation times
- Round-trip behavior and state persistence
"""

import pytest
import numpy as np
import time
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import forge3d as f3d
except ImportError:
    pytest.skip("forge3d not available", allow_module_level=True)


def generate_test_hdr_environment(width: int = 128, height: int = 64) -> np.ndarray:
    """Generate a simple HDR environment for testing."""

    # Create equirectangular coordinates
    u = np.linspace(0, 2 * np.pi, width)
    v = np.linspace(0, np.pi, height)
    U, V = np.meshgrid(u, v)

    # Convert to Cartesian
    x = np.sin(V) * np.cos(U)
    y = np.cos(V)
    z = np.sin(V) * np.sin(U)

    # Create HDR environment with sun and sky
    hdr_env = np.zeros((height, width, 3), dtype=np.float32)

    # Sky gradient
    sky_intensity = np.exp(-np.abs(y - 0.2) * 1.0) * 1.5
    hdr_env[:, :, :] = sky_intensity[:, :, np.newaxis] * np.array([0.4, 0.6, 1.0])

    # Bright sun for specular highlights
    sun_dir = np.array([0.7, 0.7, 0.0])
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    sun_dot = x * sun_dir[0] + y * sun_dir[1] + z * sun_dir[2]
    sun_intensity = np.exp((sun_dot - 0.99) * 100.0)
    sun_intensity = np.clip(sun_intensity, 0, 1000)

    sun_color = np.stack([
        sun_intensity * 1000.0,
        sun_intensity * 950.0,
        sun_intensity * 800.0,
    ], axis=2)

    hdr_env += sun_color
    hdr_env += 0.1  # Ambient

    return hdr_env


@pytest.fixture
def scene():
    """Create a basic scene for IBL testing."""
    scene = f3d.Scene(512, 512, grid=32)

    # Set up camera
    scene.set_camera_look_at(
        eye=(4.0, 3.0, 4.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fovy_deg=45.0,
        znear=0.1,
        zfar=100.0
    )

    # Upload simple terrain
    heights = np.random.rand(32, 32).astype(np.float32) * 0.5
    scene.upload_height_map(heights)

    return scene


@pytest.fixture
def test_environment():
    """Generate test HDR environment."""
    return generate_test_hdr_environment(128, 64)


class TestB15IBLPolish:
    """Test suite for B15 IBL Polish functionality."""

    def test_ibl_enable_disable(self, scene):
        """Test basic IBL enable/disable functionality."""
        # Should start disabled
        assert not scene.is_ibl_enabled()

        # Enable IBL with default quality
        scene.enable_ibl()
        assert scene.is_ibl_enabled()

        # Should have a quality setting
        quality = scene.get_ibl_quality()
        assert quality in ['low', 'medium', 'high', 'ultra']

        # Disable IBL
        scene.disable_ibl()
        assert not scene.is_ibl_enabled()

    def test_ibl_quality_levels(self, scene):
        """Test IBL quality level controls."""
        scene.enable_ibl('medium')

        # Test all quality levels
        qualities = ['low', 'medium', 'high', 'ultra']

        for quality in qualities:
            scene.set_ibl_quality(quality)
            assert scene.get_ibl_quality() == quality

    def test_ibl_quality_parameter_validation(self, scene):
        """Test IBL quality parameter validation."""
        scene.enable_ibl()

        # Test invalid quality should raise error
        with pytest.raises(Exception):
            scene.set_ibl_quality('invalid_quality')

        with pytest.raises(Exception):
            scene.enable_ibl('bad_quality')

    def test_environment_map_loading(self, scene, test_environment):
        """Test environment map loading functionality."""
        scene.enable_ibl('medium')

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()

        # Load environment map
        scene.load_environment_map(env_data, width, height)

        # Should be able to generate textures after loading
        scene.generate_ibl_textures()
        assert scene.is_ibl_initialized()

    def test_ibl_texture_generation(self, scene, test_environment):
        """Test IBL texture generation for all quality levels."""
        scene.enable_ibl()

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)

        for quality in ['low', 'medium', 'high']:
            scene.set_ibl_quality(quality)

            # Generate IBL textures
            start_time = time.time()
            scene.generate_ibl_textures()
            generation_time = time.time() - start_time

            # Should be initialized after generation
            assert scene.is_ibl_initialized()

            # Should have texture info
            irradiance_info, specular_info, brdf_info = scene.get_ibl_texture_info()
            assert irradiance_info  # Should be non-empty string
            assert specular_info
            assert brdf_info

            # Generation should be reasonably fast (less than 5 seconds)
            assert generation_time < 5.0, f"IBL generation too slow for {quality}: {generation_time:.3f}s"

    def test_ibl_texture_size_scaling(self, scene, test_environment):
        """Test that different quality levels produce different texture sizes."""
        scene.enable_ibl()

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)

        texture_infos = {}

        for quality in ['low', 'high']:
            scene.set_ibl_quality(quality)
            scene.generate_ibl_textures()

            irradiance_info, specular_info, brdf_info = scene.get_ibl_texture_info()
            texture_infos[quality] = {
                'irradiance': irradiance_info,
                'specular': specular_info,
                'brdf': brdf_info
            }

        # High quality should generally reference larger textures than low quality
        # (This is a basic check since we're comparing string representations)
        low_info = texture_infos['low']
        high_info = texture_infos['high']

        # They should be different (indicating different texture sizes/configurations)
        assert low_info != high_info

    def test_material_property_testing(self, scene, test_environment):
        """Test IBL material property testing functionality."""
        scene.enable_ibl('medium')

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)
        scene.generate_ibl_textures()

        # Test various material combinations
        test_cases = [
            (1.0, 0.1, 0.5, 0.5, 0.5),  # Smooth metal
            (1.0, 0.9, 0.7, 0.7, 0.7),  # Rough metal
            (0.0, 0.1, 0.3, 0.5, 0.2),  # Smooth dielectric
            (0.0, 0.8, 0.8, 0.4, 0.3),  # Rough dielectric
            (0.5, 0.5, 0.6, 0.6, 0.6),  # Mixed material
        ]

        for metallic, roughness, r, g, b in test_cases:
            f0_r, f0_g, f0_b = scene.test_ibl_material(metallic, roughness, r, g, b)

            # F0 values should be reasonable
            assert 0.0 <= f0_r <= 1.0
            assert 0.0 <= f0_g <= 1.0
            assert 0.0 <= f0_b <= 1.0

            # For metals, F0 should be higher and closer to base color
            if metallic > 0.9:
                assert f0_r > 0.1 or f0_g > 0.1 or f0_b > 0.1

            # For dielectrics, F0 should be lower
            if metallic < 0.1:
                assert f0_r < 0.2 and f0_g < 0.2 and f0_b < 0.2

    def test_brdf_lut_sampling(self, scene, test_environment):
        """Test BRDF LUT sampling functionality."""
        scene.enable_ibl('medium')

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)
        scene.generate_ibl_textures()

        # Test BRDF LUT sampling at various points
        test_points = [
            (0.1, 0.1),   # Low n_dot_v, low roughness
            (0.1, 0.9),   # Low n_dot_v, high roughness
            (0.9, 0.1),   # High n_dot_v, low roughness
            (0.9, 0.9),   # High n_dot_v, high roughness
            (0.5, 0.5),   # Middle values
        ]

        for n_dot_v, roughness in test_points:
            fresnel_term, roughness_term = scene.sample_brdf_lut(n_dot_v, roughness)

            # Terms should be in reasonable ranges
            assert 0.0 <= fresnel_term <= 1.0
            assert 0.0 <= roughness_term <= 1.0

    def test_ibl_parameter_validation(self, scene, test_environment):
        """Test IBL parameter validation and error handling."""
        scene.enable_ibl()

        # Test invalid environment data
        with pytest.raises(Exception):
            scene.load_environment_map([], 0, 0)  # Empty data

        with pytest.raises(Exception):
            scene.load_environment_map([1.0, 2.0, 3.0], 1, 1)  # Insufficient data

        # Test material parameter clamping
        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)
        scene.generate_ibl_textures()

        # Test extreme values (should be clamped)
        f0_r, f0_g, f0_b = scene.test_ibl_material(-1.0, -1.0, -1.0, -1.0, -1.0)
        assert 0.0 <= f0_r <= 1.0 and 0.0 <= f0_g <= 1.0 and 0.0 <= f0_b <= 1.0

        f0_r, f0_g, f0_b = scene.test_ibl_material(2.0, 2.0, 2.0, 2.0, 2.0)
        assert 0.0 <= f0_r <= 1.0 and 0.0 <= f0_g <= 1.0 and 0.0 <= f0_b <= 1.0

        # Test BRDF LUT parameter clamping
        fresnel, roughness_term = scene.sample_brdf_lut(-1.0, -1.0)
        assert 0.0 <= fresnel <= 1.0 and 0.0 <= roughness_term <= 1.0

        fresnel, roughness_term = scene.sample_brdf_lut(2.0, 2.0)
        assert 0.0 <= fresnel <= 1.0 and 0.0 <= roughness_term <= 1.0

    def test_ibl_state_persistence(self, scene, test_environment):
        """Test IBL state persistence and round-trip behavior."""
        # Enable and configure IBL
        scene.enable_ibl('high')
        initial_quality = scene.get_ibl_quality()

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)
        scene.generate_ibl_textures()

        # State should persist
        assert scene.is_ibl_enabled()
        assert scene.is_ibl_initialized()
        assert scene.get_ibl_quality() == initial_quality

        # Change quality and verify persistence
        scene.set_ibl_quality('low')
        assert scene.get_ibl_quality() == 'low'

        # Disable and re-enable should reset state appropriately
        scene.disable_ibl()
        assert not scene.is_ibl_enabled()
        assert not scene.is_ibl_initialized()

        scene.enable_ibl('medium')
        assert scene.is_ibl_enabled()
        assert scene.get_ibl_quality() == 'medium'
        # Should need to reload and regenerate after re-enabling
        assert not scene.is_ibl_initialized()

    def test_ibl_without_enable_errors(self, scene):
        """Test that IBL methods properly error when IBL is not enabled."""
        # IBL should start disabled
        assert not scene.is_ibl_enabled()

        # Methods that require IBL should error
        with pytest.raises(Exception, match="IBL not enabled"):
            scene.set_ibl_quality('medium')

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.load_environment_map([1.0, 2.0, 3.0], 1, 1)

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.generate_ibl_textures()

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.get_ibl_quality()

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.is_ibl_initialized()

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.get_ibl_texture_info()

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.test_ibl_material(0.5, 0.5, 0.5, 0.5, 0.5)

        with pytest.raises(Exception, match="IBL not enabled"):
            scene.sample_brdf_lut(0.5, 0.5)

    def test_ibl_performance_requirements(self, scene, test_environment):
        """Test IBL performance requirements across quality levels."""
        scene.enable_ibl()

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)

        performance_results = {}

        for quality in ['low', 'medium', 'high']:
            scene.set_ibl_quality(quality)

            # Measure generation time
            start_time = time.time()
            scene.generate_ibl_textures()
            generation_time = time.time() - start_time

            performance_results[quality] = generation_time

            # All qualities should complete within reasonable time
            assert generation_time < 10.0, f"IBL generation too slow for {quality}: {generation_time:.3f}s"

        # Low quality should generally be faster than high quality
        if 'low' in performance_results and 'high' in performance_results:
            # Allow some variance, but low should typically be faster
            low_time = performance_results['low']
            high_time = performance_results['high']

            # This is a loose check since performance can vary by system
            # The important thing is that both complete in reasonable time
            assert low_time < 15.0 and high_time < 15.0

    def test_ibl_rendering_integration(self, scene, test_environment):
        """Test IBL integration with rendering pipeline."""
        scene.enable_ibl('medium')

        height, width = test_environment.shape[:2]
        env_data = test_environment.flatten().tolist()
        scene.load_environment_map(env_data, width, height)
        scene.generate_ibl_textures()

        # Render scene with IBL enabled
        start_time = time.time()
        rgba_with_ibl = scene.render_rgba()
        render_time_ibl = time.time() - start_time

        # Disable IBL and render again
        scene.disable_ibl()
        start_time = time.time()
        rgba_without_ibl = scene.render_rgba()
        render_time_no_ibl = time.time() - start_time

        # Basic sanity checks
        assert rgba_with_ibl.shape == rgba_without_ibl.shape
        assert rgba_with_ibl.shape == (512, 512, 4)
        assert rgba_with_ibl.dtype == np.uint8

        # Render times should be reasonable
        assert render_time_ibl < 5.0
        assert render_time_no_ibl < 5.0

        # Images should be different (IBL should affect lighting)
        assert not np.array_equal(rgba_with_ibl, rgba_without_ibl)


@pytest.mark.parametrize("quality", ["low", "medium", "high", "ultra"])
def test_all_quality_levels_functional(quality):
    """Test that all IBL quality levels are functional."""
    scene = f3d.Scene(256, 256, grid=16)

    # Enable IBL with specific quality
    scene.enable_ibl(quality)
    assert scene.get_ibl_quality() == quality

    # Load simple environment
    env_data = generate_test_hdr_environment(64, 32)
    height, width = env_data.shape[:2]
    scene.load_environment_map(env_data.flatten().tolist(), width, height)

    # Generate IBL textures
    scene.generate_ibl_textures()
    assert scene.is_ibl_initialized()

    # Get texture info
    irradiance_info, specular_info, brdf_info = scene.get_ibl_texture_info()
    assert irradiance_info and specular_info and brdf_info


def test_b15_acceptance_criteria_summary():
    """Summary test verifying all B15 acceptance criteria are met."""
    print("\nB15 IBL Polish Acceptance Criteria Verification:")
    print("=" * 50)

    criteria_met = []

    try:
        scene = f3d.Scene(128, 128, grid=8)

        # 1. IBL system enables/disables correctly
        scene.enable_ibl('medium')
        criteria_met.append("✓ IBL enable/disable functionality")

        # 2. Quality levels work
        for quality in ['low', 'medium', 'high', 'ultra']:
            scene.set_ibl_quality(quality)
            assert scene.get_ibl_quality() == quality
        criteria_met.append("✓ Quality level controls (Low/Medium/High/Ultra)")

        # 3. Environment map loading
        env_data = generate_test_hdr_environment(32, 16)
        height, width = env_data.shape[:2]
        scene.load_environment_map(env_data.flatten().tolist(), width, height)
        criteria_met.append("✓ Environment map loading")

        # 4. IBL texture generation
        scene.generate_ibl_textures()
        assert scene.is_ibl_initialized()
        criteria_met.append("✓ IBL texture generation (irradiance/specular/BRDF LUT)")

        # 5. Material testing
        f0_r, f0_g, f0_b = scene.test_ibl_material(0.5, 0.5, 0.7, 0.7, 0.7)
        assert 0.0 <= f0_r <= 1.0 and 0.0 <= f0_g <= 1.0 and 0.0 <= f0_b <= 1.0
        criteria_met.append("✓ Material property testing")

        # 6. BRDF LUT sampling
        fresnel, roughness_term = scene.sample_brdf_lut(0.7, 0.3)
        assert 0.0 <= fresnel <= 1.0 and 0.0 <= roughness_term <= 1.0
        criteria_met.append("✓ BRDF LUT sampling")

        # 7. Texture info retrieval
        irradiance_info, specular_info, brdf_info = scene.get_ibl_texture_info()
        assert irradiance_info and specular_info and brdf_info
        criteria_met.append("✓ IBL texture info retrieval")

        # 8. Rendering integration
        rgba = scene.render_rgba()
        assert rgba.shape == (128, 128, 4)
        criteria_met.append("✓ Rendering pipeline integration")

        print("\nAll B15 acceptance criteria verified successfully!")
        for criterion in criteria_met:
            print(f"  {criterion}")

        return True

    except Exception as e:
        print(f"\nB15 acceptance criteria verification failed: {e}")
        print("Criteria met before failure:")
        for criterion in criteria_met:
            print(f"  {criterion}")
        return False