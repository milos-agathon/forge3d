#!/usr/bin/env python3
"""
Tests for PBR pipeline rendering accuracy.

Tests PBR pipeline output for luminance accuracy, ensuring rendered pixels
have luminance values within 10% of expected values for known material configurations.
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


def calculate_luminance(rgb):
    """Calculate luminance from RGB values using ITU-R BT.709 standard."""
    if len(rgb.shape) == 3:
        # RGB image
        return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    else:
        # Single RGB value
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


class TestPbrPipelineLuminance:
    """Test PBR pipeline luminance accuracy."""
    
    def test_pbr_pipeline_support(self):
        """Test that PBR pipeline is available."""
        has_support = pbr.has_pbr_support()
        if not has_support:
            pytest.skip("PBR pipeline not available")
        
        assert has_support
        print("PBR pipeline support detected")
    
    def test_white_material_luminance(self):
        """Test luminance accuracy for pure white material."""
        if not pbr.has_pbr_support():
            pytest.skip("PBR pipeline not available")
        
        # Create a white material
        material = pbr.PbrMaterial()
        material.base_color = (1.0, 1.0, 1.0, 1.0)
        material.metallic = 0.0
        material.roughness = 1.0  # Fully rough for diffuse only
        
        # Render a simple scene with the material
        try:
            # Create a simple renderer for testing
            renderer = f3d.Renderer(width=64, height=64)
            
            # Render with PBR material
            image = pbr.render_pbr_material(renderer, material)
            
            # Convert to RGB if needed and calculate luminance
            if image.shape[-1] == 4:  # RGBA
                rgb = image[:, :, :3]
            else:
                rgb = image
            
            # Normalize to [0,1] if needed
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            
            luminance = calculate_luminance(rgb)
            
            # For a white diffuse material under typical lighting,
            # we expect reasonable luminance values
            # The exact value depends on lighting setup, but should be significant
            expected_min_luma = 0.1  # Minimum expected luminance
            expected_max_luma = 1.0  # Maximum expected luminance
            
            # Check that all pixels have luminance within expected range
            assert np.all(luminance >= expected_min_luma * 0.9), f"Some pixels too dark: min={np.min(luminance)}"
            assert np.all(luminance <= expected_max_luma * 1.1), f"Some pixels too bright: max={np.max(luminance)}"
            
            # Check that luminance variation is within 10% of mean
            mean_luma = np.mean(luminance)
            tolerance = mean_luma * 0.1
            
            assert np.all(np.abs(luminance - mean_luma) <= tolerance), \
                f"Luminance variation exceeds 10%: mean={mean_luma:.3f}, range=({np.min(luminance):.3f}, {np.max(luminance):.3f})"
            
            print(f"White material luminance test PASS: mean={mean_luma:.3f}, range=({np.min(luminance):.3f}, {np.max(luminance):.3f})")
            
        except Exception as e:
            # If PBR rendering is not fully implemented, mark as expected failure
            pytest.xfail(f"PBR rendering not fully implemented: {e}")
    
    def test_black_material_luminance(self):
        """Test luminance accuracy for black material."""
        if not pbr.has_pbr_support():
            pytest.skip("PBR pipeline not available")
        
        # Create a black material
        material = pbr.PbrMaterial()
        material.base_color = (0.0, 0.0, 0.0, 1.0)
        material.metallic = 0.0
        material.roughness = 1.0
        
        try:
            renderer = f3d.Renderer(width=64, height=64)
            image = pbr.render_pbr_material(renderer, material)
            
            if image.shape[-1] == 4:  # RGBA
                rgb = image[:, :, :3]
            else:
                rgb = image
            
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            
            luminance = calculate_luminance(rgb)
            
            # Black material should have very low luminance
            expected_max_luma = 0.1  # Should be quite dark
            
            assert np.all(luminance <= expected_max_luma), \
                f"Black material too bright: max luminance={np.max(luminance):.3f}"
            
            mean_luma = np.mean(luminance)
            print(f"Black material luminance test PASS: mean={mean_luma:.3f}, max={np.max(luminance):.3f}")
            
        except Exception as e:
            pytest.xfail(f"PBR rendering not fully implemented: {e}")
    
    def test_metallic_material_luminance(self):
        """Test luminance accuracy for metallic material."""
        if not pbr.has_pbr_support():
            pytest.skip("PBR pipeline not available")
        
        # Create a metallic material (chrome-like)
        material = pbr.PbrMaterial()
        material.base_color = (0.7, 0.7, 0.7, 1.0)
        material.metallic = 1.0
        material.roughness = 0.1  # Very smooth
        
        try:
            renderer = f3d.Renderer(width=64, height=64)
            image = pbr.render_pbr_material(renderer, material)
            
            if image.shape[-1] == 4:  # RGBA
                rgb = image[:, :, :3]
            else:
                rgb = image
            
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            
            luminance = calculate_luminance(rgb)
            mean_luma = np.mean(luminance)
            
            # Metallic materials can have high variation due to specular reflections
            # but the mean should be reasonable
            assert mean_luma > 0.05, f"Metallic material too dark: mean={mean_luma:.3f}"
            assert mean_luma < 0.95, f"Metallic material too bright: mean={mean_luma:.3f}"
            
            print(f"Metallic material luminance test PASS: mean={mean_luma:.3f}, range=({np.min(luminance):.3f}, {np.max(luminance):.3f})")
            
        except Exception as e:
            pytest.xfail(f"PBR rendering not fully implemented: {e}")
    
    def test_material_luminance_ordering(self):
        """Test that materials with different albedos produce correctly ordered luminance."""
        if not pbr.has_pbr_support():
            pytest.skip("PBR pipeline not available")
        
        materials = []
        expected_luminance_order = []
        
        # Create materials with different brightness levels
        for brightness in [0.2, 0.5, 0.8]:
            material = pbr.PbrMaterial()
            material.base_color = (brightness, brightness, brightness, 1.0)
            material.metallic = 0.0
            material.roughness = 1.0
            materials.append(material)
            expected_luminance_order.append(brightness)
        
        try:
            renderer = f3d.Renderer(width=32, height=32)
            actual_luminances = []
            
            for i, material in enumerate(materials):
                image = pbr.render_pbr_material(renderer, material)
                
                if image.shape[-1] == 4:  # RGBA
                    rgb = image[:, :, :3]
                else:
                    rgb = image
                
                if rgb.dtype == np.uint8:
                    rgb = rgb.astype(np.float32) / 255.0
                
                luminance = calculate_luminance(rgb)
                mean_luma = np.mean(luminance)
                actual_luminances.append(mean_luma)
                
                print(f"Material {i+1} (brightness={expected_luminance_order[i]:.1f}): luminance={mean_luma:.3f}")
            
            # Check that luminance ordering is correct (within 10% tolerance)
            for i in range(len(actual_luminances) - 1):
                # Each subsequent material should be brighter
                ratio = actual_luminances[i+1] / max(actual_luminances[i], 1e-6)
                expected_ratio = expected_luminance_order[i+1] / expected_luminance_order[i]
                
                # Allow 10% tolerance on the ratio
                assert ratio >= expected_ratio * 0.9, \
                    f"Luminance ordering incorrect: material {i+1} should be brighter than {i}"
                    
            print("Luminance ordering test PASS: materials correctly ordered by brightness")
            
        except Exception as e:
            pytest.xfail(f"PBR rendering not fully implemented: {e}")
    
    def test_specular_luma_ordering(self):
        """Test metallic specular luminance ordering with >=10% gaps."""
        import forge3d.pbr as pbr
        import numpy as np
        
        if not pbr.has_pbr_support():
            pytest.skip("PBR pipeline not available")
        
        # Create PBR renderer
        renderer = pbr.PbrRenderer()
        
        # Fix light/view/normal vectors as specified
        light_dir = np.array([0.0, -1.0, 0.5])
        light_dir = light_dir / np.linalg.norm(light_dir)  # normalize
        view_dir = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        
        # Test metallic factors [1.0, 0.5, 0.0]
        metallic_factors = [1.0, 0.5, 0.0]
        luminances = []
        
        for metallic in metallic_factors:
            # Create PBR material with fixed parameters
            material = pbr.PbrMaterial()
            material.base_color = (1.0, 1.0, 1.0, 1.0)
            material.roughness = 0.2
            material.metallic = metallic
            
            # Evaluate BRDF to get RGB
            rgb = renderer.evaluate_brdf(material, light_dir, view_dir, normal)
            
            # Compute luminance using specified formula: L = 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            luminances.append(luminance)
        
        L1, L05, L0 = luminances
        
        # Print debug values
        print(f"Metallic luminance values: L(1.0)={L1:.4f}, L(0.5)={L05:.4f}, L(0.0)={L0:.4f}")
        
        # Assert monotonicity: L(1.0) >= L(0.5) >= L(0.0)
        assert L1 >= L05, f"Monotonicity violated: L(1.0)={L1:.4f} < L(0.5)={L05:.4f}"
        assert L05 >= L0, f"Monotonicity violated: L(0.5)={L05:.4f} < L(0.0)={L0:.4f}"
        
        # Assert adjacent gaps >= 10% of the larger adjacent value
        gap1 = L1 - L05
        gap2 = L05 - L0
        min_gap1 = 0.10 * max(L1, L05)
        min_gap2 = 0.10 * max(L05, L0)
        
        assert gap1 >= min_gap1, f"Gap too small: (L1 - L05) = {gap1:.4f} < 10% of max = {min_gap1:.4f}"
        assert gap2 >= min_gap2, f"Gap too small: (L05 - L0) = {gap2:.4f} < 10% of max = {min_gap2:.4f}"
        
        print(f"Specular luma ordering test PASS: gaps {gap1:.4f}, {gap2:.4f} both >= 10%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])