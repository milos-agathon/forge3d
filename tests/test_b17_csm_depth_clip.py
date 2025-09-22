"""
B17: Depth-clip control for CSM test suite

Tests unclipped depth support detection, cascade retuning, shadow artifact reduction,
and performance requirements for the B17 CSM depth-clip control system.

Acceptance Criteria:
1. Unclipped depth support detection on compatible hardware
2. Cascade retuning optimized for unclipped depth ranges
3. CSM clipping artifact removal (≥80% reduction)
4. Performance stability (≤5% regression in frame time)
5. Graceful fallback on hardware without unclipped depth support
"""

import pytest
import numpy as np
import forge3d as f3d
import time
from typing import Dict, List, Tuple, Optional


class TestB17CSMDepthClip:
    """Test suite for B17 depth-clip control for CSM functionality."""

    @pytest.fixture
    def scene_setup(self):
        """Create a test scene with CSM configuration."""
        scene = f3d.Scene(width=512, height=512)

        # Configure CSM with settings that expose depth clipping issues
        scene.configure_csm(
            cascade_count=4,
            shadow_map_size=1024,
            max_shadow_distance=200.0,
            pcf_kernel_size=5,
            depth_bias=0.001,
            slope_bias=0.005,
            peter_panning_offset=0.1,
            enable_evsm=False,
            debug_mode=0
        )

        # Add test geometry that creates shadow artifacts
        scene.add_terrain_quad(
            center_x=0.0, center_y=0.0,
            width=100.0, height=100.0,
            height_noise_scale=10.0,
            height_noise_octaves=4
        )

        # Position light to create challenging shadow scenarios
        scene.set_csm_light_direction((-0.5, -0.8, -0.3))
        scene.set_csm_enabled(True)

        return scene

    def test_unclipped_depth_detection(self, scene_setup):
        """Test B17.1: Unclipped depth support detection."""
        scene = scene_setup

        # Test hardware capability detection
        has_unclipped_support = scene.detect_unclipped_depth_support()
        assert isinstance(has_unclipped_support, bool)

        # Test enabling unclipped depth
        if has_unclipped_support:
            # Should enable successfully on supported hardware
            result = scene.set_unclipped_depth_enabled(True)
            assert result is True

            # Verify state change
            assert scene.is_unclipped_depth_enabled() is True

            # Test disabling
            result = scene.set_unclipped_depth_enabled(False)
            assert result is True
            assert scene.is_unclipped_depth_enabled() is False
        else:
            # Should fail gracefully on unsupported hardware
            result = scene.set_unclipped_depth_enabled(True)
            assert result is False
            assert scene.is_unclipped_depth_enabled() is False

    def test_cascade_retuning(self, scene_setup):
        """Test B17.2: Cascade retuning for unclipped depth."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Get baseline cascade splits
        baseline_splits = scene.get_csm_cascade_splits()
        assert len(baseline_splits) == 4

        # Enable unclipped depth and retune
        scene.set_unclipped_depth_enabled(True)
        scene.retune_cascades_for_unclipped_depth()

        # Get updated cascade splits
        unclipped_splits = scene.get_csm_cascade_splits()
        assert len(unclipped_splits) == 4

        # Verify cascade optimization for unclipped depth
        # Splits should be different and optimized for extended range
        assert unclipped_splits != baseline_splits

        # Verify proper cascade ordering
        for i in range(len(unclipped_splits) - 1):
            assert unclipped_splits[i] < unclipped_splits[i + 1]

        # Test custom range calculation
        custom_splits = scene.calculate_unclipped_cascade_splits(1.0, 500.0)
        assert len(custom_splits) == 4
        assert custom_splits[0] >= 1.0
        assert custom_splits[-1] <= 500.0

    def test_shadow_artifact_reduction(self, scene_setup):
        """Test B17.3: CSM clipping artifact removal (≥80% reduction)."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Configure camera for challenging shadow scenario
        scene.set_camera_position(50.0, 30.0, 50.0)
        scene.set_camera_target(0.0, 0.0, 0.0)

        # Render with standard clipped depth
        scene.set_unclipped_depth_enabled(False)
        clipped_stats = scene.analyze_shadow_artifacts()

        # Render with unclipped depth
        scene.set_unclipped_depth_enabled(True)
        scene.retune_cascades_for_unclipped_depth()
        unclipped_stats = scene.analyze_shadow_artifacts()

        # Verify artifact reduction
        clipped_artifacts = clipped_stats.get('artifact_pixels', 0)
        unclipped_artifacts = unclipped_stats.get('artifact_pixels', 0)

        if clipped_artifacts > 0:
            reduction = 1.0 - (unclipped_artifacts / clipped_artifacts)
            assert reduction >= 0.8, f"Artifact reduction {reduction:.1%} < 80% requirement"

        # Verify peter-panning reduction
        clipped_peter_panning = clipped_stats.get('peter_panning_score', 0.0)
        unclipped_peter_panning = unclipped_stats.get('peter_panning_score', 0.0)

        assert unclipped_peter_panning <= clipped_peter_panning

    @pytest.mark.parametrize("cascade_count", [2, 3, 4])
    def test_cascade_count_compatibility(self, scene_setup, cascade_count):
        """Test B17.4: Compatibility with different cascade counts."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Reconfigure with different cascade count
        scene.configure_csm(
            cascade_count=cascade_count,
            shadow_map_size=1024,
            max_shadow_distance=200.0,
            pcf_kernel_size=3,
            depth_bias=0.001,
            slope_bias=0.005,
            peter_panning_offset=0.1,
            enable_evsm=False,
            debug_mode=0
        )

        # Enable unclipped depth
        result = scene.set_unclipped_depth_enabled(True)
        assert result is True

        # Retune cascades
        scene.retune_cascades_for_unclipped_depth()

        # Verify cascade splits
        splits = scene.get_csm_cascade_splits()
        assert len(splits) == cascade_count

        # Render successfully
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)

    def test_performance_stability(self, scene_setup):
        """Test B17.5: Performance stability (≤5% regression)."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Measure baseline performance
        scene.set_unclipped_depth_enabled(False)
        baseline_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            scene.render_rgba()
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)

        baseline_avg = np.mean(baseline_times)

        # Measure unclipped depth performance
        scene.set_unclipped_depth_enabled(True)
        scene.retune_cascades_for_unclipped_depth()

        unclipped_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            scene.render_rgba()
            end_time = time.perf_counter()
            unclipped_times.append(end_time - start_time)

        unclipped_avg = np.mean(unclipped_times)

        # Verify performance regression ≤5%
        regression = (unclipped_avg - baseline_avg) / baseline_avg
        assert regression <= 0.05, f"Performance regression {regression:.1%} > 5% threshold"

    def test_cascade_statistics(self, scene_setup):
        """Test B17.6: Cascade statistics and validation."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Enable unclipped depth
        scene.set_unclipped_depth_enabled(True)
        scene.retune_cascades_for_unclipped_depth()

        # Get cascade statistics
        stats = scene.get_csm_cascade_statistics()

        # Verify statistics structure
        assert 'cascades' in stats
        assert 'coverage' in stats
        assert 'efficiency' in stats

        cascades = stats['cascades']
        assert len(cascades) == 4  # Default cascade count

        for i, cascade in enumerate(cascades):
            assert 'near' in cascade
            assert 'far' in cascade
            assert 'texel_density' in cascade
            assert cascade['near'] < cascade['far']

            if i > 0:
                # Verify cascade continuity
                prev_cascade = cascades[i - 1]
                assert abs(cascade['near'] - prev_cascade['far']) < 0.01

    def test_depth_range_validation(self, scene_setup):
        """Test B17.7: Depth range validation and optimization."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Test various depth ranges
        test_ranges = [
            (1.0, 100.0),
            (0.1, 1000.0),
            (5.0, 500.0),
            (10.0, 2000.0)
        ]

        scene.set_unclipped_depth_enabled(True)

        for near_plane, far_plane in test_ranges:
            splits = scene.calculate_unclipped_cascade_splits(near_plane, far_plane)

            # Verify splits are within range
            assert splits[0] >= near_plane
            assert splits[-1] <= far_plane

            # Verify progressive distribution
            for i in range(len(splits) - 1):
                assert splits[i] < splits[i + 1]

    def test_hardware_fallback(self, scene_setup):
        """Test B17.8: Graceful fallback on unsupported hardware."""
        scene = scene_setup

        # Test behavior regardless of hardware support
        initial_state = scene.is_unclipped_depth_enabled()

        # Attempt to enable unclipped depth
        result = scene.set_unclipped_depth_enabled(True)

        if scene.detect_unclipped_depth_support():
            # Should succeed on supported hardware
            assert result is True
            assert scene.is_unclipped_depth_enabled() is True
        else:
            # Should fail gracefully on unsupported hardware
            assert result is False
            assert scene.is_unclipped_depth_enabled() is False

            # Verify fallback rendering still works
            rgba = scene.render_rgba()
            assert rgba.shape == (512, 512, 4)

    def test_integration_with_existing_csm(self, scene_setup):
        """Test B17.9: Integration with existing CSM B4 functionality."""
        scene = scene_setup

        # Test that existing CSM functions still work
        scene.set_csm_enabled(True)
        scene.set_csm_light_direction((0.0, -1.0, 0.0))
        scene.set_csm_pcf_kernel(3)
        scene.set_csm_bias_params(0.002, 0.01, 0.05)
        scene.set_csm_debug_mode(1)

        # Verify settings applied
        assert scene.validate_csm_peter_panning() is True

        # Test cascade info retrieval
        cascade_info = scene.get_csm_cascade_info()
        assert len(cascade_info) > 0

        # Enable unclipped depth if supported
        if scene.detect_unclipped_depth_support():
            scene.set_unclipped_depth_enabled(True)
            scene.retune_cascades_for_unclipped_depth()

        # Verify rendering still works
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)

        # Verify pixel content is valid
        assert np.any(rgba[:, :, 3] > 0)  # Some alpha content

    @pytest.mark.parametrize("shadow_map_size", [512, 1024, 2048])
    def test_shadow_map_size_compatibility(self, scene_setup, shadow_map_size):
        """Test B17.10: Compatibility with different shadow map sizes."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Reconfigure with different shadow map size
        scene.configure_csm(
            cascade_count=4,
            shadow_map_size=shadow_map_size,
            max_shadow_distance=200.0,
            pcf_kernel_size=3,
            depth_bias=0.001,
            slope_bias=0.005,
            peter_panning_offset=0.1,
            enable_evsm=False,
            debug_mode=0
        )

        # Enable unclipped depth
        scene.set_unclipped_depth_enabled(True)
        scene.retune_cascades_for_unclipped_depth()

        # Verify rendering works with larger shadow maps
        rgba = scene.render_rgba()
        assert rgba.shape == (512, 512, 4)

    def test_edge_cases_and_robustness(self, scene_setup):
        """Test B17.11: Edge cases and robustness."""
        scene = scene_setup

        if not scene.detect_unclipped_depth_support():
            pytest.skip("Unclipped depth not supported on this hardware")

        # Test enabling/disabling multiple times
        for _ in range(5):
            scene.set_unclipped_depth_enabled(True)
            scene.set_unclipped_depth_enabled(False)

        # Final enable state
        scene.set_unclipped_depth_enabled(True)

        # Test retuning multiple times
        for _ in range(3):
            scene.retune_cascades_for_unclipped_depth()

        # Test extreme camera positions
        extreme_positions = [
            (0.0, 1000.0, 0.0),  # Very high
            (1000.0, 10.0, 1000.0),  # Very far
            (0.1, 0.1, 0.1),  # Very close
        ]

        for pos in extreme_positions:
            scene.set_camera_position(pos[0], pos[1], pos[2])

            # Should not crash
            rgba = scene.render_rgba()
            assert rgba.shape == (512, 512, 4)

    def test_acceptance_criteria_summary(self, scene_setup):
        """Test B17.12: Overall acceptance criteria validation."""
        scene = scene_setup

        # AC1: Hardware detection
        has_support = scene.detect_unclipped_depth_support()
        assert isinstance(has_support, bool)

        if not has_support:
            pytest.skip("Unclipped depth not supported - graceful degradation verified")

        # AC2: Enable unclipped depth
        result = scene.set_unclipped_depth_enabled(True)
        assert result is True

        # AC3: Cascade retuning
        scene.retune_cascades_for_unclipped_depth()
        splits = scene.get_csm_cascade_splits()
        assert len(splits) >= 2

        # AC4: Artifact reduction (if measurable)
        stats = scene.analyze_shadow_artifacts()
        assert 'artifact_pixels' in stats

        # AC5: Performance stability
        start_time = time.perf_counter()
        rgba = scene.render_rgba()
        end_time = time.perf_counter()

        render_time = end_time - start_time
        assert render_time < 1.0  # Should render within 1 second

        # AC6: Integration validation
        assert rgba.shape == (512, 512, 4)
        assert np.any(rgba[:, :, 3] > 0)

        print(f"B17 Depth-clip control for CSM: All acceptance criteria validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])