# tests/test_restir.py
# Tests for ReSTIR DI implementation

import pytest
import numpy as np
from typing import Dict, Any

pytest_plugins = ["helpers_namespace"]

try:
    import forge3d
    from forge3d.lighting import RestirDI, RestirConfig, LightSample, LightType, create_test_scene
    HAS_FORGE3D = True
except ImportError:
    HAS_FORGE3D = False


class TestAliasTable:
    """Test alias table functionality."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_alias_table_uniform_weights(self):
        """Test alias table with uniform weights."""
        restir = RestirDI()

        # Add lights with uniform weights
        for i in range(4):
            restir.add_light(
                position=(i * 2.0, 1.0, 0.0),
                intensity=1.0,
                weight=1.0
            )

        # Sample multiple times and check distribution
        samples = []
        for _ in range(1000):
            result = restir.sample_light(np.random.random(), np.random.random())
            if result is not None:
                samples.append(result[0])

        # Check that all lights are sampled roughly equally
        unique, counts = np.unique(samples, return_counts=True)
        assert len(unique) == 4, f"Expected 4 unique samples, got {len(unique)}"

        # Each light should be sampled roughly 250 times (±50)
        for count in counts:
            assert 200 <= count <= 300, f"Sample count {count} outside expected range"

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_alias_table_weighted(self):
        """Test alias table with non-uniform weights."""
        restir = RestirDI()

        # Add lights with different weights (10:1 ratio)
        restir.add_light(position=(0.0, 1.0, 0.0), intensity=1.0, weight=10.0)
        restir.add_light(position=(2.0, 1.0, 0.0), intensity=1.0, weight=1.0)

        # Sample multiple times
        samples = []
        for _ in range(1000):
            result = restir.sample_light(np.random.random(), np.random.random())
            if result is not None:
                samples.append(result[0])

        unique, counts = np.unique(samples, return_counts=True)
        assert len(unique) == 2

        # First light (weight 10) should be sampled ~10x more than second (weight 1)
        ratio = counts[0] / counts[1] if counts[1] > 0 else float('inf')
        assert 8.0 <= ratio <= 12.0, f"Weight ratio {ratio} outside expected range [8,12]"


class TestReservoir:
    """Test reservoir sampling functionality."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_reservoir_basic_operations(self):
        """Test basic reservoir operations."""
        restir = RestirDI()

        # Add a single light
        restir.add_light(position=(0.0, 1.0, 0.0), intensity=1.0)

        # Basic sampling should work
        result = restir.sample_light(0.5, 0.5)
        assert result is not None
        assert result[0] == 0  # First (and only) light
        assert result[1] > 0.0  # Non-zero PDF

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_empty_reservoir(self):
        """Test behavior with no lights."""
        restir = RestirDI()

        # No lights added
        result = restir.sample_light(0.5, 0.5)
        assert result is None


class TestRestirConfig:
    """Test ReSTIR configuration."""

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config
        config = RestirConfig(
            initial_candidates=32,
            spatial_radius=16.0,
            depth_threshold=0.1,
            normal_threshold=0.9
        )
        assert config.initial_candidates == 32

        # Invalid candidates
        with pytest.raises(ValueError, match="initial_candidates must be positive"):
            RestirConfig(initial_candidates=0)

        # Invalid radius
        with pytest.raises(ValueError, match="spatial_radius must be positive"):
            RestirConfig(spatial_radius=0.0)

        # Invalid thresholds
        with pytest.raises(ValueError, match="depth_threshold must be in"):
            RestirConfig(depth_threshold=1.5)

        with pytest.raises(ValueError, match="normal_threshold must be in"):
            RestirConfig(normal_threshold=-0.1)

    def test_config_defaults(self):
        """Test default configuration values."""
        config = RestirConfig()
        assert config.initial_candidates == 32
        assert config.temporal_neighbors == 1
        assert config.spatial_neighbors == 4
        assert config.spatial_radius == 16.0
        assert config.max_temporal_age == 20
        assert config.bias_correction is True


class TestLightManagement:
    """Test light addition and management."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_add_single_light(self):
        """Test adding a single light."""
        restir = RestirDI()

        light_idx = restir.add_light(
            position=(1.0, 2.0, 3.0),
            intensity=1.5,
            light_type=LightType.POINT,
            weight=2.0
        )

        assert light_idx == 0
        assert restir.num_lights == 1

        lights = restir.lights
        assert len(lights) == 1
        assert lights[0].position == (1.0, 2.0, 3.0)
        assert lights[0].intensity == 1.5
        assert lights[0].light_type == LightType.POINT

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_add_multiple_lights(self):
        """Test adding multiple lights."""
        restir = RestirDI()

        positions = [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)]
        intensities = [1.0, 2.0, 3.0]

        for i, (pos, intensity) in enumerate(zip(positions, intensities)):
            light_idx = restir.add_light(position=pos, intensity=intensity)
            assert light_idx == i

        assert restir.num_lights == 3

        lights = restir.lights
        for i, light in enumerate(lights):
            assert light.position == positions[i]
            assert light.intensity == intensities[i]

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_set_lights_bulk(self):
        """Test setting lights in bulk."""
        restir = RestirDI()

        light_samples = [
            LightSample(position=(0.0, 1.0, 0.0), light_index=0, direction=(0.0, 0.0, 1.0),
                       intensity=1.0, light_type=LightType.POINT),
            LightSample(position=(1.0, 1.0, 0.0), light_index=1, direction=(0.0, -1.0, 0.0),
                       intensity=2.0, light_type=LightType.DIRECTIONAL),
        ]
        weights = [1.0, 3.0]

        restir.set_lights(light_samples, weights)

        assert restir.num_lights == 2
        lights = restir.lights
        assert lights[0].intensity == 1.0
        assert lights[1].intensity == 2.0
        assert lights[1].light_type == LightType.DIRECTIONAL

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_clear_lights(self):
        """Test clearing all lights."""
        restir = RestirDI()

        # Add some lights
        restir.add_light(position=(0.0, 1.0, 0.0), intensity=1.0)
        restir.add_light(position=(1.0, 1.0, 0.0), intensity=2.0)
        assert restir.num_lights == 2

        # Clear lights
        restir.clear_lights()
        assert restir.num_lights == 0
        assert len(restir.lights) == 0


class TestVarianceReduction:
    """Test variance reduction calculation."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_variance_reduction_calculation(self):
        """Test variance reduction calculation."""
        restir = RestirDI()

        # Create two test images with different variance
        high_var_image = np.random.normal(0.5, 0.3, (64, 64, 3)).astype(np.float32)
        low_var_image = np.random.normal(0.5, 0.1, (64, 64, 3)).astype(np.float32)

        # Clip to valid range
        high_var_image = np.clip(high_var_image, 0.0, 1.0)
        low_var_image = np.clip(low_var_image, 0.0, 1.0)

        # Calculate reduction (should be positive since low_var has lower variance)
        reduction = restir.calculate_variance_reduction(high_var_image, low_var_image)
        assert reduction > 0, f"Expected positive variance reduction, got {reduction}"

        # Test with same image (should be 0% reduction)
        reduction_same = restir.calculate_variance_reduction(high_var_image, high_var_image)
        assert abs(reduction_same) < 1e-6, f"Expected ~0% reduction for same image, got {reduction_same}"

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_variance_reduction_shape_mismatch(self):
        """Test variance reduction with mismatched shapes."""
        restir = RestirDI()

        image1 = np.zeros((64, 64, 3))
        image2 = np.zeros((32, 32, 3))

        with pytest.raises(ValueError, match="Images must have the same shape"):
            restir.calculate_variance_reduction(image1, image2)


class TestTestScene:
    """Test the test scene creation utility."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_create_test_scene(self):
        """Test creating a test scene."""
        restir = create_test_scene(num_lights=50, seed=42)

        assert restir.num_lights == 50

        lights = restir.lights
        assert len(lights) == 50

        # Check that lights are within expected bounds
        for light in lights:
            assert -5.0 <= light.position[0] <= 5.0  # within scene_bounds[0]/2
            assert 0.1 <= light.position[1] <= 5.0   # within height range
            assert -5.0 <= light.position[2] <= 5.0  # within scene_bounds[1]/2
            assert 0.1 <= light.intensity <= 2.0     # within intensity range

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_create_test_scene_reproducible(self):
        """Test that test scene creation is reproducible."""
        restir1 = create_test_scene(num_lights=10, seed=123)
        restir2 = create_test_scene(num_lights=10, seed=123)

        lights1 = restir1.lights
        lights2 = restir2.lights

        for l1, l2 in zip(lights1, lights2):
            assert l1.position == l2.position
            assert l1.intensity == l2.intensity
            assert l1.light_type == l2.light_type


class TestStatistics:
    """Test statistics and metrics."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_get_statistics(self):
        """Test getting ReSTIR statistics."""
        config = RestirConfig(initial_candidates=64, spatial_radius=20.0)
        restir = RestirDI(config)

        # Add some lights
        restir.add_light(position=(0.0, 1.0, 0.0), intensity=1.0)
        restir.add_light(position=(1.0, 1.0, 0.0), intensity=2.0)

        stats = restir.get_statistics()
        assert stats["num_lights"] == 2
        assert stats["config"]["initial_candidates"] == 64
        assert stats["config"]["spatial_radius"] == 20.0


@pytest.mark.gpu
class TestRestirGPU:
    """GPU-dependent tests for ReSTIR."""

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_gpu_availability(self):
        """Test if GPU ReSTIR implementation is available."""
        try:
            restir = RestirDI()
            # If we get here without exception, GPU implementation should be available
            assert restir is not None
        except RuntimeError as e:
            pytest.skip(f"GPU ReSTIR not available: {e}")

    @pytest.mark.skipif(not HAS_FORGE3D, reason="forge3d not available")
    def test_variance_target_achievement(self):
        """Test that ReSTIR achieves the variance reduction target (≤40% vs MIS-only)."""
        pytest.skip("Requires full GPU implementation and MIS reference")

        # This test would:
        # 1. Create a scene with many lights
        # 2. Render with MIS-only sampling
        # 3. Render with ReSTIR DI
        # 4. Calculate variance reduction
        # 5. Assert reduction >= 40%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])