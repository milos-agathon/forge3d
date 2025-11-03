"""
P2-10: Python unit tests for BRDF override precedence

Tests that RendererConfig.brdf_override takes precedence over per-material
(shading.brdf) settings. These tests verify the configuration logic and can
run on CPU-only CI without requiring GPU rendering.

Exit criteria: Passing tests on CPU-only CI (use stubs/mocks if needed).
"""

import copy
import sys
from forge3d.config import RendererConfig, ShadingParams, LightingParams


class TestBrdfOverridePrecedence:
    """Test BRDF override precedence in RendererConfig"""

    def test_no_override_uses_material_brdf(self):
        """When brdf_override is None, shading.brdf should be used"""
        config = RendererConfig()
        config.shading.brdf = "lambert"
        config.brdf_override = None
        
        # Verify configuration
        assert config.shading.brdf == "lambert"
        assert config.brdf_override is None
        
        # Serialize and verify
        data = config.to_dict()
        assert data["shading"]["brdf"] == "lambert"
        assert "brdf_override" not in data  # None values are omitted

    def test_override_set_lambert(self):
        """When brdf_override='lambert', it should override material brdf"""
        config = RendererConfig()
        config.shading.brdf = "cooktorrance-ggx"  # Material setting
        config.brdf_override = "lambert"  # Global override
        
        # Verify configuration
        assert config.shading.brdf == "cooktorrance-ggx"  # Material unchanged
        assert config.brdf_override == "lambert"  # Override set
        
        # Serialize and verify both are present
        data = config.to_dict()
        assert data["shading"]["brdf"] == "cooktorrance-ggx"
        assert data["brdf_override"] == "lambert"

    def test_override_set_ggx(self):
        """Test override with Cook-Torrance GGX model"""
        config = RendererConfig()
        config.shading.brdf = "lambert"
        config.brdf_override = "cooktorrance-ggx"
        
        assert config.shading.brdf == "lambert"
        assert config.brdf_override == "cooktorrance-ggx"
        
        data = config.to_dict()
        assert data["brdf_override"] == "cooktorrance-ggx"

    def test_override_set_disney(self):
        """Test override with Disney Principled BRDF"""
        config = RendererConfig()
        config.shading.brdf = "phong"
        config.brdf_override = "disney-principled"
        
        assert config.shading.brdf == "phong"
        assert config.brdf_override == "disney-principled"
        
        data = config.to_dict()
        assert data["brdf_override"] == "disney-principled"

    def test_override_all_brdf_models(self):
        """Test override with all supported BRDF models"""
        brdf_models = [
            "lambert",
            "phong",
            "blinn-phong",
            "oren-nayar",
            "cooktorrance-ggx",
            "disney-principled",
            "toon",
            "minnaert",
            "ward",
            "ashikhmin-shirley",
        ]
        
        for brdf in brdf_models:
            config = RendererConfig()
            config.shading.brdf = "lambert"  # Default material
            config.brdf_override = brdf
            
            assert config.brdf_override == brdf, f"Failed to set override to {brdf}"
            
            data = config.to_dict()
            assert data["brdf_override"] == brdf

    def test_from_mapping_with_override(self):
        """Test loading config from dict with brdf_override"""
        data = {
            "shading": {"brdf": "lambert"},
            "brdf_override": "cooktorrance-ggx",
        }
        
        config = RendererConfig.from_mapping(data)
        
        assert config.shading.brdf == "lambert"
        assert config.brdf_override == "cooktorrance-ggx"

    def test_from_mapping_without_override(self):
        """Test loading config from dict without brdf_override"""
        data = {
            "shading": {"brdf": "disney-principled"},
        }
        
        config = RendererConfig.from_mapping(data)
        
        assert config.shading.brdf == "disney-principled"
        assert config.brdf_override is None

    def test_from_mapping_override_null(self):
        """Test loading config with explicit null override"""
        data = {
            "shading": {"brdf": "phong"},
            "brdf_override": None,
        }
        
        config = RendererConfig.from_mapping(data)
        
        assert config.shading.brdf == "phong"
        assert config.brdf_override is None

    def test_override_normalization_case_insensitive(self):
        """Test that BRDF override values are case-normalized"""
        data = {
            "brdf_override": "LAMBERT",  # Uppercase
        }
        
        config = RendererConfig.from_mapping(data)
        
        # Should be normalized to lowercase with hyphens
        assert config.brdf_override == "lambert"

    def test_override_normalization_underscores_to_hyphens(self):
        """Test that underscores are converted to hyphens"""
        data = {
            "brdf_override": "cook_torrance_ggx",
        }
        
        config = RendererConfig.from_mapping(data)
        
        assert config.brdf_override == "cooktorrance-ggx"

    def test_override_with_default_base(self):
        """Test from_mapping with override using default base"""
        base_config = RendererConfig()
        base_config.shading.brdf = "phong"
        
        data = {
            "brdf_override": "lambert",
        }
        
        config = RendererConfig.from_mapping(data, default=base_config)
        
        # Base shading.brdf should be preserved
        assert config.shading.brdf == "phong"
        # Override should be set
        assert config.brdf_override == "lambert"

    def test_override_replaces_base_override(self):
        """Test that new override replaces base config override"""
        base_config = RendererConfig()
        base_config.brdf_override = "lambert"
        
        data = {
            "brdf_override": "disney-principled",
        }
        
        config = RendererConfig.from_mapping(data, default=base_config)
        
        # Override should be replaced
        assert config.brdf_override == "disney-principled"

    def test_copy_preserves_override(self):
        """Test that copying config preserves brdf_override"""
        config1 = RendererConfig()
        config1.shading.brdf = "lambert"
        config1.brdf_override = "cooktorrance-ggx"
        
        config2 = config1.copy()
        
        assert config2.shading.brdf == "lambert"
        assert config2.brdf_override == "cooktorrance-ggx"
        
        # Verify deep copy (modifications don't affect original)
        config2.brdf_override = "disney-principled"
        assert config1.brdf_override == "cooktorrance-ggx"

    def test_invalid_brdf_override_raises_error(self):
        """Test that invalid BRDF override value raises ValueError"""
        data = {
            "brdf_override": "invalid-brdf-model",
        }
        
        try:
            RendererConfig.from_mapping(data)
            assert False, "Expected ValueError but none was raised"
        except ValueError as e:
            assert "BRDF model" in str(e)

    def test_precedence_multiple_materials(self):
        """Test that override applies globally across different material settings"""
        # Scenario: Different materials might specify different BRDFs,
        # but brdf_override should force a single model globally
        
        config = RendererConfig()
        config.shading.brdf = "lambert"  # Material 1 setting
        config.brdf_override = "cooktorrance-ggx"  # Global override
        
        # In a real renderer, even if we had multiple materials with different
        # shading.brdf values, the brdf_override should apply to all
        
        assert config.brdf_override == "cooktorrance-ggx"
        assert config.shading.brdf == "lambert"  # Original setting preserved
        
        # The renderer implementation should respect brdf_override first


class TestBrdfOverrideSerialization:
    """Test serialization and deserialization of BRDF override"""

    def test_roundtrip_with_override(self):
        """Test serialize -> deserialize preserves override"""
        config1 = RendererConfig()
        config1.shading.brdf = "phong"
        config1.brdf_override = "disney-principled"
        
        # Serialize
        data = config1.to_dict()
        
        # Deserialize
        config2 = RendererConfig.from_mapping(data)
        
        assert config2.shading.brdf == "phong"
        assert config2.brdf_override == "disney-principled"

    def test_roundtrip_without_override(self):
        """Test serialize -> deserialize without override"""
        config1 = RendererConfig()
        config1.shading.brdf = "cooktorrance-ggx"
        config1.brdf_override = None
        
        data = config1.to_dict()
        config2 = RendererConfig.from_mapping(data)
        
        assert config2.shading.brdf == "cooktorrance-ggx"
        assert config2.brdf_override is None

    def test_json_compatible(self):
        """Test that serialized config with override is JSON-compatible"""
        import json
        
        config = RendererConfig()
        config.brdf_override = "lambert"
        
        data = config.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        
        assert loaded["brdf_override"] == "lambert"


class TestBrdfOverrideValidation:
    """Test validation behavior with BRDF override"""

    def test_validate_with_override(self):
        """Test that validate() works with brdf_override set"""
        config = RendererConfig()
        config.brdf_override = "lambert"
        
        # Should not raise
        config.validate()

    def test_validate_without_override(self):
        """Test that validate() works without brdf_override"""
        config = RendererConfig()
        config.brdf_override = None
        
        # Should not raise
        config.validate()


class TestBrdfOverrideEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_override_empty_string_invalid(self):
        """Test that empty string override is rejected"""
        data = {
            "brdf_override": "",
        }
        
        try:
            RendererConfig.from_mapping(data)
            assert False, "Expected ValueError but none was raised"
        except ValueError:
            pass  # Expected

    def test_override_none_vs_not_set(self):
        """Test distinction between None and not setting override"""
        # Explicitly set to None
        config1 = RendererConfig()
        config1.brdf_override = None
        
        # Default (not set, but defaults to None)
        config2 = RendererConfig()
        
        assert config1.brdf_override is None
        assert config2.brdf_override is None
        
        # Both should serialize the same way (omit brdf_override)
        data1 = config1.to_dict()
        data2 = config2.to_dict()
        
        assert "brdf_override" not in data1
        assert "brdf_override" not in data2

    def test_changing_override_at_runtime(self):
        """Test changing override value at runtime"""
        config = RendererConfig()
        
        # Start with no override
        assert config.brdf_override is None
        
        # Set override
        config.brdf_override = "lambert"
        assert config.brdf_override == "lambert"
        
        # Change override
        config.brdf_override = "cooktorrance-ggx"
        assert config.brdf_override == "cooktorrance-ggx"
        
        # Clear override
        config.brdf_override = None
        assert config.brdf_override is None

    def test_override_with_all_config_sections(self):
        """Test override in a fully configured renderer"""
        data = {
            "lighting": {
                "exposure": 1.5,
                "lights": [{"type": "sun", "intensity": 5.0}],
            },
            "shading": {
                "brdf": "phong",
                "normal_maps": True,
            },
            "shadows": {
                "enabled": True,
                "technique": "pcf",
            },
            "gi": {
                "modes": ["ibl"],
            },
            "brdf_override": "disney-principled",
        }
        
        config = RendererConfig.from_mapping(data)
        
        # Verify all sections loaded correctly
        assert config.lighting.exposure == 1.5
        assert config.shading.brdf == "phong"
        assert config.shadows.enabled is True
        assert "ibl" in config.gi.modes
        
        # Verify override
        assert config.brdf_override == "disney-principled"


class TestBrdfOverrideDocumentation:
    """Test that documents the expected behavior for developers"""

    def test_override_precedence_example(self):
        """
        Example demonstrating BRDF override precedence:
        
        When rendering, the precedence should be:
        1. RendererConfig.brdf_override (if set) - HIGHEST PRIORITY
        2. Material/shading.brdf (if no override) - FALLBACK
        
        This allows users to:
        - Set per-material BRDFs via shading.brdf
        - Override ALL materials globally via brdf_override
        - Easily switch BRDF models for testing/comparison
        """
        # Scene with material using GGX
        config = RendererConfig()
        config.shading.brdf = "cooktorrance-ggx"
        
        # User wants to test with Lambert instead
        config.brdf_override = "lambert"
        
        # Renderer should use Lambert (override), not GGX (material)
        assert config.brdf_override == "lambert"  # What renderer should use
        assert config.shading.brdf == "cooktorrance-ggx"  # Original preserved
        
        # User can remove override to go back to material setting
        config.brdf_override = None
        
        # Now renderer should use GGX (material)
        assert config.brdf_override is None
        assert config.shading.brdf == "cooktorrance-ggx"  # Material setting used

    def test_use_case_brdf_comparison(self):
        """
        Use case: Comparing BRDF models side-by-side
        
        Users can render the same scene with different BRDFs by changing
        only the override, without modifying material settings.
        """
        base_config = RendererConfig()
        base_config.shading.brdf = "cooktorrance-ggx"
        
        # Render with Lambert
        lambert_config = base_config.copy()
        lambert_config.brdf_override = "lambert"
        assert lambert_config.brdf_override == "lambert"
        
        # Render with Disney
        disney_config = base_config.copy()
        disney_config.brdf_override = "disney-principled"
        assert disney_config.brdf_override == "disney-principled"
        
        # Render with material setting (no override)
        material_config = base_config.copy()
        material_config.brdf_override = None
        assert material_config.brdf_override is None
        assert material_config.shading.brdf == "cooktorrance-ggx"


# Simple test runner
def run_tests():
    """Run all tests and report results"""
    test_classes = [
        TestBrdfOverridePrecedence,
        TestBrdfOverrideSerialization,
        TestBrdfOverrideValidation,
        TestBrdfOverrideEdgeCases,
        TestBrdfOverrideDocumentation,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__name__
        instance = test_class()
        
        # Get all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            test_method = getattr(instance, method_name)
            
            try:
                test_method()
                passed_tests += 1
                print(f"✓ {class_name}.{method_name}")
            except Exception as e:
                failed_tests.append((class_name, method_name, str(e)))
                print(f"✗ {class_name}.{method_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}")
            print(f"    {error}")
        sys.exit(1)
    else:
        print("\nAll tests passed! ✓")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
