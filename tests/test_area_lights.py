#!/usr/bin/env python3
"""Tests for A20: Soft Area Lights with Penumbra Control"""

import pytest
import numpy as np

try:
    from forge3d.lighting import (
        AreaLightType, AreaLight, AreaLightManager,
        create_area_light_test_scene
    )
    AREA_LIGHTS_AVAILABLE = True
except ImportError:
    AREA_LIGHTS_AVAILABLE = False


@pytest.mark.skipif(not AREA_LIGHTS_AVAILABLE, reason="Area lights module not available")
class TestAreaLight:
    """Test A20: Area light functionality."""

    def test_area_light_creation(self):
        """Test area light creation with default parameters."""
        light = AreaLight()
        assert light.position == (0.0, 5.0, 0.0)
        assert light.light_type == AreaLightType.DISC
        assert light.radius == 1.0
        assert light.intensity == 10.0
        assert hasattr(light, 'energy_factor')
        assert light.energy_factor > 0.0

    def test_area_light_validation(self):
        """Test area light parameter validation."""
        # Invalid radius
        with pytest.raises(ValueError, match="Radius must be positive"):
            AreaLight(radius=-1.0)

        # Invalid intensity
        with pytest.raises(ValueError, match="Intensity must be positive"):
            AreaLight(intensity=-5.0)

        # Invalid softness
        with pytest.raises(ValueError, match="Softness must be in \\[0.0, 1.0\\]"):
            AreaLight(softness=1.5)

    def test_area_light_direction_normalization(self):
        """Test that light direction is properly normalized."""
        light = AreaLight(direction=(3.0, 4.0, 0.0))
        direction = np.array(light.direction)
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-6

    def test_penumbra_radius_control(self):
        """Test penumbra radius control functionality."""
        light = AreaLight(radius=1.0)
        initial_energy = light.get_effective_energy()

        # Change radius and verify energy factor updates
        light.set_radius(2.0)
        assert light.radius == 2.0
        new_energy = light.get_effective_energy()

        # Energy should change due to normalization
        assert new_energy != initial_energy

    def test_energy_factor_computation(self):
        """Test energy factor computation for different light types."""
        # Rectangle light
        rect_light = AreaLight(
            light_type=AreaLightType.RECTANGLE,
            size=(2.0, 3.0),
            radius=1.0
        )
        assert rect_light.energy_factor > 0.0

        # Disc light
        disc_light = AreaLight(
            light_type=AreaLightType.DISC,
            size=(2.0, 2.0),
            radius=1.0
        )
        assert disc_light.energy_factor > 0.0

        # Different light types should have different energy factors
        assert rect_light.energy_factor != disc_light.energy_factor

    def test_effective_energy_calculation(self):
        """Test effective energy calculation."""
        light = AreaLight(intensity=20.0, radius=1.5)
        effective_energy = light.get_effective_energy()

        # Effective energy should be intensity * energy_factor
        expected = light.intensity * light.energy_factor
        assert abs(effective_energy - expected) < 1e-6

    def test_rectangle_factory_method(self):
        """Test rectangle light factory method."""
        light = AreaLight.rectangle(
            position=(1, 2, 3),
            direction=(0, -1, 0),
            width=4.0,
            height=2.0,
            intensity=15.0,
            penumbra_radius=0.8
        )

        assert light.light_type == AreaLightType.RECTANGLE
        assert light.position == (1, 2, 3)
        assert light.size == (4.0, 2.0)
        assert light.intensity == 15.0
        assert light.radius == 0.8

    def test_disc_factory_method(self):
        """Test disc light factory method."""
        light = AreaLight.disc(
            position=(0, 5, 0),
            direction=(0, -1, 0),
            disc_radius=3.0,
            intensity=12.0,
            penumbra_radius=1.2
        )

        assert light.light_type == AreaLightType.DISC
        assert light.size == (3.0, 3.0)
        assert light.intensity == 12.0
        assert light.radius == 1.2

    def test_radius_affects_penumbra(self):
        """Test that radius changes affect penumbra characteristics."""
        light1 = AreaLight(radius=0.5)
        light2 = AreaLight(radius=2.0)

        # Larger radius should result in different energy normalization
        assert light1.energy_factor != light2.energy_factor

        # The relationship should be that larger radius = larger penumbra
        # (verified through energy factor change)
        energy1 = light1.get_effective_energy()
        energy2 = light2.get_effective_energy()
        assert energy1 != energy2


@pytest.mark.skipif(not AREA_LIGHTS_AVAILABLE, reason="Area lights module not available")
class TestAreaLightManager:
    """Test area light manager functionality."""

    def test_manager_creation(self):
        """Test area light manager creation."""
        manager = AreaLightManager(max_lights=8)
        assert manager.max_lights == 8
        assert len(manager.lights) == 0

    def test_add_light(self):
        """Test adding lights to manager."""
        manager = AreaLightManager()
        light = AreaLight()

        index = manager.add_light(light)
        assert index == 0
        assert len(manager.lights) == 1

    def test_max_lights_limit(self):
        """Test maximum lights enforcement."""
        manager = AreaLightManager(max_lights=2)

        # Add maximum number of lights
        for i in range(2):
            light = AreaLight(intensity=float(i + 1))
            manager.add_light(light)

        # Try to add one more - should fail
        with pytest.raises(ValueError, match="Maximum lights"):
            manager.add_light(AreaLight())

    def test_energy_calculation(self):
        """Test total energy calculation."""
        manager = AreaLightManager()

        light1 = AreaLight(intensity=10.0)
        light2 = AreaLight(intensity=20.0)

        manager.add_light(light1)
        manager.add_light(light2)

        total_energy = manager.calculate_total_energy()
        expected = light1.get_effective_energy() + light2.get_effective_energy()

        assert abs(total_energy - expected) < 1e-6

    def test_energy_target_setting(self):
        """Test energy target setting and validation."""
        manager = AreaLightManager()

        # Valid target
        manager.set_energy_target(100.0)
        assert manager._energy_target == 100.0

        # Invalid target
        with pytest.raises(ValueError, match="Energy target must be positive"):
            manager.set_energy_target(-10.0)

    def test_energy_normalization(self):
        """Test energy normalization functionality."""
        manager = AreaLightManager()

        # Add lights
        light1 = AreaLight(intensity=10.0)
        light2 = AreaLight(intensity=20.0)
        manager.add_light(light1)
        manager.add_light(light2)

        # Set energy target
        initial_energy = manager.calculate_total_energy()
        target_energy = initial_energy * 0.5  # Half the energy
        manager.set_energy_target(target_energy)

        # Normalize
        error = manager.normalize_energy()

        # Check that energy is now close to target
        new_energy = manager.calculate_total_energy()
        assert abs(new_energy - target_energy) / target_energy < 0.01

        # Error should be small
        assert error < 0.01

    def test_energy_conservation_requirement(self):
        """Test A20 energy conservation requirement (within 2%)."""
        manager = AreaLightManager()

        # Add test lights
        light1 = AreaLight.disc(
            position=(0, 5, 0),
            direction=(0, -1, 0),
            disc_radius=2.0,
            intensity=15.0,
            penumbra_radius=1.0
        )
        light2 = AreaLight.rectangle(
            position=(3, 4, 0),
            direction=(-1, -1, 0),
            width=2.0,
            height=1.5,
            intensity=10.0,
            penumbra_radius=0.8
        )

        manager.add_light(light1)
        manager.add_light(light2)

        # Test energy conservation
        passes_test = manager.test_energy_conservation()
        assert passes_test, "Energy conservation should be within 2% (A20 requirement)"

    def test_energy_conservation_edge_cases(self):
        """Test energy conservation with edge cases."""
        manager = AreaLightManager()

        # Empty manager should pass conservation test
        assert manager.test_energy_conservation()

        # Single light should pass
        manager.add_light(AreaLight())
        assert manager.test_energy_conservation()

    def test_penumbra_radius_scaling(self):
        """Test that penumbra widens with radius as required by A20."""
        # Create two identical lights with different radii
        light_small = AreaLight(radius=0.5, intensity=10.0)
        light_large = AreaLight(radius=2.0, intensity=10.0)

        # With same intensity, the energy factors should be different
        # This demonstrates that radius affects penumbra characteristics
        assert light_small.energy_factor != light_large.energy_factor

        # The physical interpretation is that larger radius = softer/wider penumbra
        # which requires energy normalization (tested through energy factor difference)


@pytest.mark.skipif(not AREA_LIGHTS_AVAILABLE, reason="Area lights module not available")
class TestAreaLightIntegration:
    """Integration tests for area lighting system."""

    def test_test_scene_creation(self):
        """Test creation of area light test scene."""
        manager = create_area_light_test_scene()

        assert len(manager.lights) == 2  # Key + fill lights
        assert all(isinstance(light, AreaLight) for light in manager.lights)

        # Verify lights have different configurations
        lights = manager.lights
        assert lights[0].light_type != lights[1].light_type  # Different types
        assert lights[0].position != lights[1].position      # Different positions

    def test_multi_light_energy_conservation(self):
        """Test energy conservation with multiple different light types."""
        manager = AreaLightManager()

        # Add variety of lights
        lights = [
            AreaLight.disc((0, 5, 0), (0, -1, 0), 2.0, 15.0, 1.0),
            AreaLight.rectangle((3, 4, 1), (-1, -1, 0), 2.0, 1.0, 12.0, 0.8),
            AreaLight(light_type=AreaLightType.SPHERE, size=(1.5, 1.5), intensity=8.0, radius=0.6)
        ]

        for light in lights:
            manager.add_light(light)

        # Test energy conservation across different light types
        initial_energy = manager.calculate_total_energy()
        manager.set_energy_target(initial_energy)

        # Modify all lights' radii
        for light in manager.lights:
            light.set_radius(light.radius * 1.3)

        # Normalize and check conservation
        error = manager.normalize_energy()
        assert error < 0.02, f"Multi-light energy conservation failed: {error:.3f} > 0.02"

    def test_radius_parameter_vectors(self):
        """Test A20 radius parameter with vector configurations."""
        # Test that different radius values produce different penumbra characteristics
        radii = [0.5, 1.0, 1.5, 2.0, 3.0]
        lights = []

        for radius in radii:
            light = AreaLight(radius=radius, intensity=10.0)
            lights.append(light)

        # All lights should have different energy factors due to radius differences
        energy_factors = [light.energy_factor for light in lights]
        assert len(set(energy_factors)) == len(radii), "Different radii should produce different energy factors"

        # Energy factors should follow a predictable pattern
        for i in range(len(energy_factors) - 1):
            assert energy_factors[i] != energy_factors[i + 1]

    def test_energy_within_2_percent_requirement(self):
        """Specific test for A20 requirement: energy within 2%."""
        manager = AreaLightManager()

        # Add lights with known energies
        light1 = AreaLight(intensity=20.0, radius=1.0)
        light2 = AreaLight(intensity=15.0, radius=0.8)

        manager.add_light(light1)
        manager.add_light(light2)

        initial_energy = manager.calculate_total_energy()
        manager.set_energy_target(initial_energy)

        # Modify radii significantly
        light1.set_radius(2.5)  # 2.5x increase
        light2.set_radius(0.3)  # Major decrease

        # Energy conservation should still work
        error = manager.normalize_energy()
        final_energy = manager.calculate_total_energy()

        # Check the 2% requirement
        energy_difference = abs(final_energy - initial_energy) / initial_energy
        assert energy_difference < 0.02, f"Energy conservation error {energy_difference:.4f} exceeds 2% requirement"

        # Also check the direct error measurement
        assert error < 0.02, f"Normalization error {error:.4f} exceeds 2% requirement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])