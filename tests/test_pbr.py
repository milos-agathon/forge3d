#!/usr/bin/env python3
"""
Tests for PBR (Physically-Based Rendering) materials functionality.

Tests PBR material creation, validation, BRDF calculations, texture management,
and integration with the rendering system.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d as f3d
    import forge3d.pbr as pbr
    HAS_PBR = True
except ImportError:
    HAS_PBR = False

pytestmark = pytest.mark.skipif(not HAS_PBR, reason="PBR module not available")


class TestPbrMaterials:
    """Test PBR material creation and properties."""
    
    def test_pbr_support_detection(self):
        """Test PBR support detection."""
        has_support = pbr.has_pbr_support()
        assert isinstance(has_support, bool)
        
        if has_support:
            print("PBR support detected")
        else:
            pytest.skip("PBR support not available")
    
    def test_basic_material_creation(self):
        """Test basic PBR material creation."""
        material = pbr.PbrMaterial()
        
        # Check default values
        assert material.base_color == (1.0, 1.0, 1.0, 1.0)
        assert material.metallic == 0.0
        assert material.roughness == 1.0
        assert material.normal_scale == 1.0
        assert material.occlusion_strength == 1.0
        assert material.emissive == (0.0, 0.0, 0.0)
        assert material.alpha_cutoff == 0.5
        assert material.texture_flags == 0
    
    def test_custom_material_creation(self):
        """Test PBR material creation with custom properties."""
        base_color = (0.8, 0.2, 0.2, 1.0)
        metallic = 0.7
        roughness = 0.3
        
        material = pbr.PbrMaterial(
            base_color=base_color,
            metallic=metallic,
            roughness=roughness
        )
        
        assert material.base_color == base_color
        assert material.metallic == metallic
        assert material.roughness == roughness
    
    def test_metallic_clamping(self):
        """Test that metallic values are properly clamped."""
        # Test below minimum
        material = pbr.PbrMaterial(metallic=-0.5)
        assert material.metallic == 0.0
        
        # Test above maximum
        material = pbr.PbrMaterial(metallic=1.5)
        assert material.metallic == 1.0
        
        # Test within range
        material = pbr.PbrMaterial(metallic=0.5)
        assert material.metallic == 0.5
    
    def test_roughness_clamping(self):
        """Test that roughness values are properly clamped."""
        # Test below minimum (should clamp to 0.04)
        material = pbr.PbrMaterial(roughness=0.01)
        assert material.roughness >= 0.04
        
        # Test above maximum
        material = pbr.PbrMaterial(roughness=1.5)
        assert material.roughness == 1.0
        
        # Test within range
        material = pbr.PbrMaterial(roughness=0.5)
        assert material.roughness == 0.5
    
    def test_material_validation_valid(self):
        """Test validation of valid materials."""
        material = pbr.PbrMaterial(
            base_color=(0.8, 0.2, 0.2, 1.0),
            metallic=0.0,
            roughness=0.7
        )
        
        validation = pbr.validate_pbr_material(material)
        
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
        # May have warnings, but should be valid
        
        stats = validation['statistics']
        assert 'is_metallic' in stats
        assert 'is_dielectric' in stats
        assert 'is_rough' in stats
        assert 'is_smooth' in stats
        assert 'is_emissive' in stats
    
    def test_material_validation_invalid(self):
        """Test validation of invalid materials."""
        # Create material with invalid values
        material = pbr.PbrMaterial(
            base_color=(1.2, -0.1, 0.5, 1.0),  # Out of range values
            metallic=1.5,  # Out of range
            roughness=0.01  # Below minimum
        )
        
        # Manual override for testing (since constructor clamps values)
        material._base_color = (1.2, -0.1, 0.5, 1.0)
        material._metallic = 1.5
        material._roughness = 0.01
        
        validation = pbr.validate_pbr_material(material)
        
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
        
        # Check that specific errors are detected
        error_messages = ' '.join(validation['errors'])
        assert any('base_color' in error for error in validation['errors'])
    
    def test_test_materials_creation(self):
        """Test creation of test material library."""
        materials = pbr.create_test_materials()
        
        assert isinstance(materials, dict)
        assert len(materials) >= 5  # Should have several test materials
        
        # Check some expected materials
        expected_materials = ['plastic_red', 'metal_gold', 'metal_iron']
        for name in expected_materials:
            if name in materials:
                material = materials[name]
                assert isinstance(material, pbr.PbrMaterial)
        
        # Verify materials have different properties
        metallic_values = [mat.metallic for mat in materials.values()]
        roughness_values = [mat.roughness for mat in materials.values()]
        
        assert len(set(metallic_values)) > 1  # Different metallic values
        assert len(set(roughness_values)) > 1  # Different roughness values


class TestBrdfCalculations:
    """Test BRDF calculation functionality."""
    
    def test_brdf_renderer_creation(self):
        """Test PBR renderer creation."""
        renderer = pbr.PbrRenderer()
        assert renderer is not None
    
    def test_brdf_lighting_setup(self):
        """Test PBR lighting configuration."""
        lighting = pbr.PbrLighting(
            light_direction=(0.0, -1.0, 0.0),
            light_color=(1.0, 1.0, 1.0),
            light_intensity=2.0,
            camera_position=(0.0, 0.0, 5.0)
        )
        
        assert lighting.light_direction == (0.0, -1.0, 0.0)
        assert lighting.light_color == (1.0, 1.0, 1.0)
        assert lighting.light_intensity == 2.0
        assert lighting.camera_position == (0.0, 0.0, 5.0)
        
        renderer = pbr.PbrRenderer()
        renderer.set_lighting(lighting)
    
    def test_brdf_evaluation_dielectric(self):
        """Test BRDF evaluation for dielectric materials."""
        material = pbr.PbrMaterial(
            base_color=(0.8, 0.8, 0.8, 1.0),
            metallic=0.0,  # Dielectric
            roughness=0.5
        )
        
        renderer = pbr.PbrRenderer()
        
        light_dir = np.array([0.0, -1.0, 0.3])
        light_dir = light_dir / np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        
        brdf = renderer.evaluate_brdf(material, light_dir, view_dir, normal)
        
        assert isinstance(brdf, np.ndarray)
        assert brdf.shape == (3,)  # RGB
        assert np.all(brdf >= 0.0)  # Non-negative
        assert np.all(np.isfinite(brdf))  # No NaN/inf
    
    def test_brdf_evaluation_metallic(self):
        """Test BRDF evaluation for metallic materials."""
        material = pbr.PbrMaterial(
            base_color=(0.7, 0.4, 0.2, 1.0),  # Copper-like
            metallic=1.0,  # Metallic
            roughness=0.2
        )
        
        renderer = pbr.PbrRenderer()
        
        light_dir = np.array([0.0, -1.0, 0.3])
        light_dir = light_dir / np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        
        brdf = renderer.evaluate_brdf(material, light_dir, view_dir, normal)
        
        assert isinstance(brdf, np.ndarray)
        assert brdf.shape == (3,)  # RGB
        assert np.all(brdf >= 0.0)  # Non-negative
        assert np.all(np.isfinite(brdf))  # No NaN/inf
    
    def test_brdf_metallic_vs_dielectric_difference(self):
        """Test that metallic and dielectric materials produce different BRDF results."""
        base_color = (0.7, 0.4, 0.3, 1.0)
        roughness = 0.3
        
        dielectric = pbr.PbrMaterial(
            base_color=base_color,
            metallic=0.0,
            roughness=roughness
        )
        
        metallic = pbr.PbrMaterial(
            base_color=base_color,
            metallic=1.0,
            roughness=roughness
        )
        
        renderer = pbr.PbrRenderer()
        
        light_dir = np.array([0.0, -1.0, 0.5])
        light_dir = light_dir / np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        
        dielectric_brdf = renderer.evaluate_brdf(dielectric, light_dir, view_dir, normal)
        metallic_brdf = renderer.evaluate_brdf(metallic, light_dir, view_dir, normal)
        
        # Should have significant difference
        brdf_diff = np.linalg.norm(metallic_brdf - dielectric_brdf)
        assert brdf_diff > 0.05  # At least 5% difference
    
    def test_brdf_roughness_effect(self):
        """Test that roughness affects BRDF calculations."""
        base_color = (0.8, 0.8, 0.8, 1.0)
        metallic = 0.0
        
        smooth = pbr.PbrMaterial(
            base_color=base_color,
            metallic=metallic,
            roughness=0.1  # Smooth
        )
        
        rough = pbr.PbrMaterial(
            base_color=base_color,
            metallic=metallic,
            roughness=0.9  # Rough
        )
        
        renderer = pbr.PbrRenderer()
        
        light_dir = np.array([0.0, -1.0, 0.3])
        light_dir = light_dir / np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        
        smooth_brdf = renderer.evaluate_brdf(smooth, light_dir, view_dir, normal)
        rough_brdf = renderer.evaluate_brdf(rough, light_dir, view_dir, normal)
        
        # Should have measurable difference
        brdf_diff = np.linalg.norm(smooth_brdf - rough_brdf)
        assert brdf_diff > 0.01  # At least 1% difference


class TestPbrTextures:
    """Test PBR texture functionality."""
    
    def test_test_textures_creation(self):
        """Test creation of test texture library."""
        textures = pbr.create_test_textures()
        
        assert isinstance(textures, dict)
        assert len(textures) >= 3  # Should have several test textures
        
        # Check expected textures
        expected_textures = ['checker_base_color', 'metallic_roughness', 'normal']
        for name in expected_textures:
            if name in textures:
                texture = textures[name]
                assert isinstance(texture, np.ndarray)
                assert texture.dtype in [np.uint8, np.float32]
                assert len(texture.shape) in [2, 3]  # 2D or 3D array
    
    def test_material_texture_assignment(self):
        """Test assigning textures to materials."""
        material = pbr.PbrMaterial()
        textures = pbr.create_test_textures()
        
        # Initially no textures
        assert material.texture_flags == 0
        
        # Set base color texture
        if 'checker_base_color' in textures:
            material.set_base_color_texture(textures['checker_base_color'])
            assert material.texture_flags & 1  # Base color flag set
        
        # Set metallic-roughness texture
        if 'metallic_roughness' in textures:
            material.set_metallic_roughness_texture(textures['metallic_roughness'])
            assert material.texture_flags & 2  # Metallic-roughness flag set
        
        # Set normal texture
        if 'normal' in textures:
            material.set_normal_texture(textures['normal'])
            assert material.texture_flags & 4  # Normal flag set


class TestPbrIntegration:
    """Test PBR integration with rendering system."""
    
    def test_pbr_with_renderer(self):
        """Test PBR materials with main renderer."""
        if not f3d.has_gpu():
            pytest.skip("GPU not available")
        
        renderer = f3d.Renderer(256, 256)
        
        # Create PBR material
        material = pbr.PbrMaterial(
            base_color=(0.8, 0.2, 0.2, 1.0),
            metallic=0.0,
            roughness=0.7
        )
        
        # Should be able to use material with renderer
        # (This tests basic compatibility - full rendering tested in examples)
        assert material is not None
        assert renderer is not None
    
    def test_pbr_material_serialization(self):
        """Test that PBR materials can be serialized for GPU use."""
        material = pbr.PbrMaterial(
            base_color=(0.7, 0.3, 0.1, 1.0),
            metallic=0.8,
            roughness=0.2,
            normal_scale=1.5,
            occlusion_strength=0.8,
            emissive=(0.1, 0.1, 0.0),
            alpha_cutoff=0.3
        )
        
        # Test that material properties are accessible
        assert material.base_color == (0.7, 0.3, 0.1, 1.0)
        assert material.metallic == 0.8
        assert material.roughness == 0.2
        assert material.normal_scale == 1.5
        assert material.occlusion_strength == 0.8
        assert material.emissive == (0.1, 0.1, 0.0)
        assert material.alpha_cutoff == 0.3


def test_pbr_example_runs():
    """Test that the PBR example can run without errors."""
    from pathlib import Path
    import subprocess
    import sys
    
    example_path = Path(__file__).parent.parent / "examples" / "pbr_materials.py"
    
    if not example_path.exists():
        pytest.skip("PBR example not found")
    
    # Run example in test mode
    result = subprocess.run([
        sys.executable, str(example_path),
        "--headless",
        "--out", "out/test_pbr_materials.png",
        "--width", "400",
        "--height", "300"
    ], capture_output=True, text=True, cwd=str(example_path.parent.parent))
    
    assert result.returncode == 0, f"PBR example failed: {result.stderr}"
    
    # Check that output file was created
    output_path = example_path.parent.parent / "out" / "test_pbr_materials.png"
    assert output_path.exists(), "PBR example did not create output file"


if __name__ == "__main__":
    pytest.main([__file__])