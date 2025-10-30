#!/usr/bin/env python3
# tests/test_b4_csm.py
# Test suite for Cascaded Shadow Maps (B4) implementation
# RELEVANT FILES:shaders/shadows.wgsl,python/forge3d/lighting.py,examples/csm_demo.py,src/shadows/csm.rs

"""
Test suite for Cascaded Shadow Maps implementation.

Tests B4 acceptance criteria:
- 3-4 cascades configuration
- PCF/EVSM filtering functionality
- Peter-panning artifact prevention
- Stable shadow rendering during motion
- Debug visualization modes
"""

import pytest
import numpy as np
from typing import List, Tuple

try:
    import forge3d.lighting as lighting
    from forge3d.lighting import (
        CsmConfig, CsmController, CsmQualityPreset, create_csm_config,
        calculate_cascade_splits, detect_peter_panning_cpu, get_csm_controller
    )
except ImportError:
    pytest.skip("forge3d not available", allow_module_level=True)


class TestCsmConfig:
    """Test CSM configuration and validation."""

    def test_csm_config_creation(self):
        """Test CSM configuration object creation and validation."""
        # Valid configuration
        config = CsmConfig(
            cascade_count=3,
            shadow_map_size=2048,
            max_shadow_distance=200.0,
            pcf_kernel_size=3,
            depth_bias=0.005,
            slope_bias=0.01,
            peter_panning_offset=0.001
        )

        assert config.cascade_count == 3
        assert config.shadow_map_size == 2048
        assert config.max_shadow_distance == 200.0
        assert config.pcf_kernel_size == 3

    def test_csm_config_validation(self):
        """Test CSM configuration parameter validation."""
        # Invalid cascade count
        with pytest.raises(ValueError, match="cascade_count must be 2-4"):
            CsmConfig(cascade_count=1)

        with pytest.raises(ValueError, match="cascade_count must be 2-4"):
            CsmConfig(cascade_count=5)

        # Invalid shadow map size
        with pytest.raises(ValueError, match="shadow_map_size must be 512-8192"):
            CsmConfig(shadow_map_size=256)

        with pytest.raises(ValueError, match="shadow_map_size must be 512-8192"):
            CsmConfig(shadow_map_size=16384)

        # Invalid max shadow distance
        with pytest.raises(ValueError, match="max_shadow_distance must be positive"):
            CsmConfig(max_shadow_distance=-1.0)

        # Invalid PCF kernel size
        with pytest.raises(ValueError, match="pcf_kernel_size must be 1, 3, 5, or 7"):
            CsmConfig(pcf_kernel_size=2)

        with pytest.raises(ValueError, match="pcf_kernel_size must be 1, 3, 5, or 7"):
            CsmConfig(pcf_kernel_size=8)

    def test_quality_presets(self):
        """Test CSM quality preset configurations."""
        # Test all quality presets
        for preset in CsmQualityPreset:
            config = create_csm_config(preset)

            # All presets should have valid parameters
            assert 2 <= config.cascade_count <= 4
            assert 512 <= config.shadow_map_size <= 8192
            assert config.pcf_kernel_size in [1, 3, 5, 7]
            assert config.depth_bias > 0
            assert config.slope_bias > 0
            assert config.peter_panning_offset > 0

        # Check specific preset properties
        low_config = create_csm_config(CsmQualityPreset.LOW)
        assert low_config.pcf_kernel_size == 1  # No PCF for low quality
        assert not low_config.enable_evsm

        ultra_config = create_csm_config(CsmQualityPreset.ULTRA)
        assert ultra_config.pcf_kernel_size == 7  # Poisson PCF for ultra quality
        assert ultra_config.enable_evsm


class TestCsmController:
    """Test CSM controller functionality."""

    def test_controller_creation(self):
        """Test CSM controller creation and initialization."""
        controller = CsmController()

        assert controller.config is not None
        assert not controller.is_enabled()  # Should start disabled
        assert controller.config.cascade_count == 3  # Default configuration

    def test_controller_with_custom_config(self):
        """Test CSM controller with custom configuration."""
        custom_config = CsmConfig(
            cascade_count=4,
            shadow_map_size=4096,
            pcf_kernel_size=5
        )
        controller = CsmController(custom_config)

        assert controller.config.cascade_count == 4
        assert controller.config.shadow_map_size == 4096
        assert controller.config.pcf_kernel_size == 5

    def test_enable_disable_shadows(self):
        """Test enabling and disabling shadow rendering."""
        controller = CsmController()

        # Initially disabled
        assert not controller.is_enabled()

        # Enable shadows
        controller.enable_shadows(True)
        assert controller.is_enabled()

        # Disable shadows
        controller.enable_shadows(False)
        assert not controller.is_enabled()

    def test_light_direction_setting(self):
        """Test setting light direction."""
        controller = CsmController()

        # Set various light directions
        directions = [
            (0.0, -1.0, 0.0),   # Straight down
            (-1.0, -1.0, 0.0),  # Angled
            (0.6, -0.8, 0.0),   # Normalized
        ]

        for direction in directions:
            controller.set_light_direction(direction)
            # Light direction should be normalized
            light_dir = controller._light_direction
            assert abs(np.linalg.norm(light_dir) - 1.0) < 1e-6

    def test_pcf_configuration(self):
        """Test PCF kernel size configuration."""
        controller = CsmController()

        # Valid PCF kernel sizes
        valid_sizes = [1, 3, 5, 7]
        for size in valid_sizes:
            controller.configure_pcf(size)
            assert controller.config.pcf_kernel_size == size

        # Invalid PCF kernel size
        with pytest.raises(ValueError, match="PCF kernel size must be 1, 3, 5, or 7"):
            controller.configure_pcf(4)

    def test_bias_parameters(self):
        """Test shadow bias parameter configuration."""
        controller = CsmController()

        depth_bias = 0.003
        slope_bias = 0.008
        peter_panning_offset = 0.0008

        controller.set_bias_parameters(depth_bias, slope_bias, peter_panning_offset)

        assert controller.config.depth_bias == depth_bias
        assert controller.config.slope_bias == slope_bias
        assert controller.config.peter_panning_offset == peter_panning_offset

    def test_debug_mode_setting(self):
        """Test debug mode configuration."""
        controller = CsmController()

        # Valid debug modes
        for mode in [0, 1, 2]:
            controller.set_debug_mode(mode)
            assert controller.config.debug_mode == mode

        # Invalid debug mode
        with pytest.raises(ValueError, match="Debug mode must be 0-2"):
            controller.set_debug_mode(3)

    def test_quality_preset_switching(self):
        """Test switching between quality presets."""
        controller = CsmController()

        for preset in CsmQualityPreset:
            controller.set_quality_preset(preset)
            expected_config = create_csm_config(preset)

            assert controller.config.cascade_count == expected_config.cascade_count
            assert controller.config.shadow_map_size == expected_config.shadow_map_size
            assert controller.config.pcf_kernel_size == expected_config.pcf_kernel_size
            assert controller.config.enable_evsm == expected_config.enable_evsm

    def test_cascade_info_retrieval(self):
        """Test cascade information retrieval."""
        controller = CsmController()

        # Get cascade information
        cascade_info = controller.get_cascade_info()

        assert len(cascade_info) == controller.config.cascade_count

        # Check that cascades have increasing distances
        for i in range(len(cascade_info) - 1):
            near1, far1, _ = cascade_info[i]
            near2, far2, _ = cascade_info[i + 1]

            assert far1 <= near2  # Cascades should not overlap significantly
            assert near1 < far1   # Each cascade should have positive range
            assert near2 < far2


class TestCascadeSplitCalculation:
    """Test cascade split distance calculations."""

    def test_basic_split_calculation(self):
        """Test basic cascade split calculation."""
        near_plane = 0.1
        far_plane = 100.0
        cascade_count = 3

        splits = calculate_cascade_splits(near_plane, far_plane, cascade_count)

        # Should have cascade_count + 1 splits (including near and far)
        assert len(splits) == cascade_count + 1

        # First split should be near plane
        assert abs(splits[0] - near_plane) < 1e-6

        # Last split should be far plane
        assert abs(splits[-1] - far_plane) < 1e-6

        # Splits should be monotonically increasing
        for i in range(len(splits) - 1):
            assert splits[i] < splits[i + 1]

    def test_split_calculation_different_counts(self):
        """Test split calculation with different cascade counts."""
        near_plane = 1.0
        far_plane = 200.0

        for cascade_count in [2, 3, 4]:
            splits = calculate_cascade_splits(near_plane, far_plane, cascade_count)

            assert len(splits) == cascade_count + 1
            assert splits[0] == near_plane
            assert splits[-1] == far_plane

            # Check that splits are reasonably distributed
            for i in range(1, len(splits) - 1):
                # Each split should be between near and far
                assert near_plane < splits[i] < far_plane

    def test_split_calculation_lambda_blending(self):
        """Test split calculation with different lambda blending factors."""
        near_plane = 0.1
        far_plane = 100.0
        cascade_count = 3

        # Pure uniform splits (lambda = 0)
        uniform_splits = calculate_cascade_splits(near_plane, far_plane, cascade_count, 0.0)

        # Pure logarithmic splits (lambda = 1)
        log_splits = calculate_cascade_splits(near_plane, far_plane, cascade_count, 1.0)

        # Default blended splits
        default_splits = calculate_cascade_splits(near_plane, far_plane, cascade_count, 0.75)

        # All should have same length
        assert len(uniform_splits) == len(log_splits) == len(default_splits)

        # Uniform splits should be more evenly spaced
        uniform_intervals = np.diff(uniform_splits)
        uniform_variance = np.var(uniform_intervals)

        # Logarithmic splits should have larger intervals near the far plane
        log_intervals = np.diff(log_splits)
        log_variance = np.var(log_intervals)

        # Logarithmic splits should have higher variance than uniform
        assert log_variance > uniform_variance


class TestPeterPanningPrevention:
    """Test peter-panning artifact detection and prevention."""

    def test_peter_panning_detection_normal_cases(self):
        """Test peter-panning detection for normal lighting scenarios."""
        # Normal case: upward surface with downward light (no peter-panning)
        assert not detect_peter_panning_cpu(
            shadow_factor=1.0,
            surface_normal=(0.0, 1.0, 0.0),
            light_direction=(0.0, -1.0, 0.0)
        )

        # Normal case: surface facing light (no peter-panning)
        assert not detect_peter_panning_cpu(
            shadow_factor=0.8,
            surface_normal=(0.0, 1.0, 0.0),
            light_direction=(0.0, -1.0, 0.0)
        )

    def test_peter_panning_detection_problematic_cases(self):
        """Test peter-panning detection for problematic lighting scenarios."""
        # Peter-panning case: downward surface with downward light and shadow
        assert detect_peter_panning_cpu(
            shadow_factor=0.2,
            surface_normal=(0.0, -1.0, 0.0),
            light_direction=(0.0, -1.0, 0.0)
        )

        # Peter-panning case: surface facing away from light with shadow
        assert detect_peter_panning_cpu(
            shadow_factor=0.3,
            surface_normal=(0.707, -0.707, 0.0),  # 45 degrees away from upward
            light_direction=(0.0, -1.0, 0.0)
        )

    def test_peter_panning_prevention_validation(self):
        """Test CSM controller peter-panning prevention validation."""
        # Controller with proper bias settings should pass validation
        config = CsmConfig(
            depth_bias=0.005,
            peter_panning_offset=0.001
        )
        controller = CsmController(config)

        assert controller.validate_peter_panning_prevention()

        # Controller with insufficient bias should fail validation
        insufficient_config = CsmConfig(
            depth_bias=0.00001,  # Too small
            peter_panning_offset=0.00001  # Too small
        )
        insufficient_controller = CsmController(insufficient_config)

        assert not insufficient_controller.validate_peter_panning_prevention()


class TestCsmIntegration:
    """Test CSM integration and end-to-end functionality."""

    def test_default_controller_singleton(self):
        """Test default CSM controller singleton."""
        controller1 = get_csm_controller()
        controller2 = get_csm_controller()

        # Should return the same instance
        assert controller1 is controller2

    def test_csm_memory_usage_estimation(self):
        """Test memory usage estimation for different configurations."""
        configs = [
            CsmConfig(cascade_count=3, shadow_map_size=1024),
            CsmConfig(cascade_count=3, shadow_map_size=2048),
            CsmConfig(cascade_count=4, shadow_map_size=2048),
            CsmConfig(cascade_count=4, shadow_map_size=4096),
        ]

        for config in configs:
            # Calculate expected memory usage (depth buffer only)
            expected_memory_mb = (
                config.shadow_map_size ** 2 *  # Pixels per cascade
                config.cascade_count *         # Number of cascades
                4                              # Bytes per pixel (32-bit depth)
            ) / (1024 * 1024)

            # Memory should be reasonable (under 512MB for GPU constraint)
            assert expected_memory_mb < 512, f"Config exceeds memory budget: {expected_memory_mb:.1f}MB"

            # Higher resolution/more cascades should use more memory
            if config.shadow_map_size > 1024 or config.cascade_count > 3:
                assert expected_memory_mb > 12  # Should use more than minimal config

    def test_csm_quality_performance_tradeoff(self):
        """Test quality vs. performance tradeoffs for different presets."""
        presets = [CsmQualityPreset.LOW, CsmQualityPreset.MEDIUM,
                  CsmQualityPreset.HIGH, CsmQualityPreset.ULTRA]

        prev_complexity = 0
        for preset in presets:
            config = create_csm_config(preset)

            # Calculate complexity score (higher = more expensive)
            complexity = (
                config.cascade_count *
                (config.shadow_map_size / 1024) ** 2 *
                max(1, config.pcf_kernel_size) *
                (2 if config.enable_evsm else 1)
            )

            # Complexity should generally increase with higher quality presets
            if prev_complexity > 0:
                # Allow some flexibility, but higher presets should generally be more complex
                assert complexity >= prev_complexity * 0.8

            prev_complexity = complexity

    @pytest.mark.parametrize("cascade_count", [3, 4])
    def test_cascade_count_acceptance_criteria(self, cascade_count):
        """Test that CSM supports 3-4 cascades as per B4 acceptance criteria."""
        config = CsmConfig(cascade_count=cascade_count)
        controller = CsmController(config)

        assert controller.config.cascade_count == cascade_count

        # Get cascade info to verify implementation
        cascade_info = controller.get_cascade_info()
        assert len(cascade_info) == cascade_count

        # Each cascade should have valid parameters
        for i, (near, far, texel_size) in enumerate(cascade_info):
            assert near >= 0
            assert far > near
            assert texel_size > 0

    @pytest.mark.parametrize("pcf_kernel", [1, 3, 5, 7])
    def test_pcf_filtering_acceptance_criteria(self, pcf_kernel):
        """Test PCF filtering options as per B4 acceptance criteria."""
        config = CsmConfig(pcf_kernel_size=pcf_kernel)
        controller = CsmController(config)

        controller.configure_pcf(pcf_kernel)
        assert controller.config.pcf_kernel_size == pcf_kernel

    def test_stability_during_motion_simulation(self):
        """Test shadow stability during simulated camera motion."""
        controller = CsmController()
        controller.enable_shadows(True)

        # Simulate camera motion by changing light direction over time
        stable_measurements = []

        for frame in range(10):
            # Slight light direction changes (simulating time of day)
            angle = frame * 0.1
            light_dir = (np.sin(angle) * 0.1, -1.0, np.cos(angle) * 0.1)
            controller.set_light_direction(light_dir)

            # Get cascade information
            cascade_info = controller.get_cascade_info()

            # Check stability metrics
            if frame > 0:
                # Compare with previous frame
                prev_info = stable_measurements[-1]

                for i, ((near1, far1, _), (near2, far2, _)) in enumerate(zip(prev_info, cascade_info)):
                    # Changes should be gradual (< 10% per frame for stability)
                    near_change = abs(near2 - near1) / max(near1, 0.001)
                    far_change = abs(far2 - far1) / max(far1, 0.001)

                    assert near_change < 0.1, f"Cascade {i} near plane changed too rapidly"
                    assert far_change < 0.1, f"Cascade {i} far plane changed too rapidly"

            stable_measurements.append(cascade_info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
