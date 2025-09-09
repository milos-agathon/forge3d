#!/usr/bin/env python3
"""
Tests for input validation across forge3d APIs.

Tests comprehensive input validation including dtype/shape/contiguity checks,
parameter validation, and error handling with meaningful error messages.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d._validate as validate
    HAS_VALIDATE = True
except ImportError:
    HAS_VALIDATE = False

pytestmark = pytest.mark.skipif(not HAS_VALIDATE, reason="Validation module not available")


class TestArrayValidation:
    """Test comprehensive array validation."""
    
    def test_validate_array_basic(self):
        """Test basic array validation."""
        # Valid array
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = validate.validate_array(arr, "test_array")
        assert np.array_equal(result, arr)
    
    def test_validate_array_type_error(self):
        """Test array type validation."""
        with pytest.raises(TypeError, match="test_array must be numpy.ndarray"):
            validate.validate_array([1, 2, 3], "test_array")
    
    def test_validate_array_contiguity(self):
        """Test contiguity validation."""
        # Create non-contiguous array
        arr = np.arange(12).reshape(3, 4)
        non_contiguous = arr[:, ::2]  # Every other column
        assert not non_contiguous.flags['C_CONTIGUOUS']
        
        with pytest.raises(ValueError, match="must be C-contiguous"):
            validate.validate_array(non_contiguous, "test_array", require_contiguous=True)
        
        # Should pass with contiguity disabled
        result = validate.validate_array(non_contiguous, "test_array", require_contiguous=False)
        assert np.array_equal(result, non_contiguous)
    
    def test_validate_array_size_limits(self):
        """Test array size validation."""
        arr = np.array([1, 2, 3])
        
        # Too small
        with pytest.raises(ValueError, match="array too small"):
            validate.validate_array(arr, "test_array", min_size=5)
        
        # Too large
        with pytest.raises(ValueError, match="array too large"):
            validate.validate_array(arr, "test_array", max_size=2)
        
        # Just right
        result = validate.validate_array(arr, "test_array", min_size=3, max_size=3)
        assert np.array_equal(result, arr)
    
    def test_validate_array_dtype(self):
        """Test dtype validation."""
        arr_float32 = np.array([1.0, 2.0], dtype=np.float32)
        arr_int32 = np.array([1, 2], dtype=np.int32)
        arr_bool = np.array([True, False], dtype=bool)
        
        # Valid dtype
        result = validate.validate_array(arr_float32, "test_array", dtype=[np.float32])
        assert result.dtype == np.float32
        
        # Invalid dtype
        with pytest.raises(ValueError, match="has invalid dtype"):
            validate.validate_array(arr_int32, "test_array", dtype=[np.float32])
        
        # Non-numeric dtype (when no dtype specified)
        with pytest.raises(ValueError, match="must have numeric dtype"):
            validate.validate_array(arr_bool, "test_array")


class TestShapePatternValidation:
    """Test shape pattern validation."""
    
    def test_shape_2d_valid(self):
        """Test valid 2D shape validation."""
        arr = np.random.rand(100, 200).astype(np.float32)
        result = validate.validate_array(arr, "height_field", shape=validate.SHAPE_2D)
        assert result.shape == (100, 200)
    
    def test_shape_2d_invalid(self):
        """Test invalid 2D shape validation."""
        # Wrong dimensions
        arr_1d = np.random.rand(100).astype(np.float32)
        with pytest.raises(ValueError, match="must be 2D array"):
            validate.validate_array(arr_1d, "height_field", shape=validate.SHAPE_2D)
        
        # Zero dimensions (will be caught by size validation first)
        arr_zero = np.zeros((0, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="array too small|dimensions must be > 0"):
            validate.validate_array(arr_zero, "height_field", shape=validate.SHAPE_2D)
        
        # Too large dimensions
        with pytest.raises(ValueError, match="dimensions exceed limit"):
            validate.validate_array(
                np.zeros((validate.MAX_ARRAY_DIM + 1, 10), dtype=np.float32),
                "height_field", 
                shape=validate.SHAPE_2D
            )
    
    def test_shape_3d_rgb_valid(self):
        """Test valid RGB shape validation."""
        arr = np.random.rand(100, 200, 3).astype(np.uint8)
        result = validate.validate_array(arr, "rgb_image", shape=validate.SHAPE_3D_RGB)
        assert result.shape == (100, 200, 3)
    
    def test_shape_3d_rgb_invalid(self):
        """Test invalid RGB shape validation."""
        # Wrong channel count
        arr_rgba = np.random.rand(100, 200, 4).astype(np.uint8)
        with pytest.raises(ValueError, match="must be 3D RGB array"):
            validate.validate_array(arr_rgba, "rgb_image", shape=validate.SHAPE_3D_RGB)
        
        # Wrong dimensions
        arr_2d = np.random.rand(100, 200).astype(np.uint8)
        with pytest.raises(ValueError, match="must be 3D RGB array"):
            validate.validate_array(arr_2d, "rgb_image", shape=validate.SHAPE_3D_RGB)
    
    def test_shape_3d_rgba_valid(self):
        """Test valid RGBA shape validation."""
        arr = np.random.rand(100, 200, 4).astype(np.uint8)
        result = validate.validate_array(arr, "rgba_image", shape=validate.SHAPE_3D_RGBA)
        assert result.shape == (100, 200, 4)
    
    def test_shape_3d_rgba_invalid(self):
        """Test invalid RGBA shape validation."""
        # Wrong channel count
        arr_rgb = np.random.rand(100, 200, 3).astype(np.uint8)
        with pytest.raises(ValueError, match="must be 3D RGBA array"):
            validate.validate_array(arr_rgb, "rgba_image", shape=validate.SHAPE_3D_RGBA)
    
    def test_shape_height_field_valid(self):
        """Test valid height field validation."""
        arr = np.random.rand(256, 512).astype(np.float32)
        result = validate.validate_array(arr, "dem_data", shape=validate.SHAPE_HEIGHT_FIELD)
        assert result.shape == (256, 512)
    
    def test_shape_height_field_invalid(self):
        """Test invalid height field validation."""
        # Too small
        arr_small = np.random.rand(1, 1).astype(np.float32)
        with pytest.raises(ValueError, match="height field too small"):
            validate.validate_array(arr_small, "dem_data", shape=validate.SHAPE_HEIGHT_FIELD)
        
        # Wrong dimensions
        arr_3d = np.random.rand(10, 10, 3).astype(np.float32)
        with pytest.raises(ValueError, match="height field must be 2D array"):
            validate.validate_array(arr_3d, "dem_data", shape=validate.SHAPE_HEIGHT_FIELD)


class TestNumericParameterValidation:
    """Test numeric parameter validation."""
    
    def test_validate_numeric_float_valid(self):
        """Test valid float parameter validation."""
        result = validate.validate_numeric_parameter(3.14, "test_param", expected_type=float)
        assert result == 3.14
        assert isinstance(result, float)
        
        # String convertible to float
        result = validate.validate_numeric_parameter("2.71", "test_param", expected_type=float)
        assert result == 2.71
        assert isinstance(result, float)
    
    def test_validate_numeric_int_valid(self):
        """Test valid int parameter validation."""
        result = validate.validate_numeric_parameter(42, "test_param", expected_type=int)
        assert result == 42
        assert isinstance(result, int)
        
        # Float convertible to int
        result = validate.validate_numeric_parameter(3.0, "test_param", expected_type=int)
        assert result == 3
        assert isinstance(result, int)
    
    def test_validate_numeric_type_error(self):
        """Test numeric parameter type errors."""
        with pytest.raises(TypeError, match="must be convertible to float"):
            validate.validate_numeric_parameter("not_a_number", "test_param", expected_type=float)
        
        with pytest.raises(TypeError, match="must be convertible to int"):
            validate.validate_numeric_parameter([1, 2, 3], "test_param", expected_type=int)
    
    def test_validate_numeric_range_validation(self):
        """Test numeric parameter range validation."""
        # Valid range
        result = validate.validate_numeric_parameter(5.0, "test_param", min_val=0.0, max_val=10.0)
        assert result == 5.0
        
        # Below minimum
        with pytest.raises(ValueError, match="too small"):
            validate.validate_numeric_parameter(-1.0, "test_param", min_val=0.0, max_val=10.0)
        
        # Above maximum
        with pytest.raises(ValueError, match="too large"):
            validate.validate_numeric_parameter(15.0, "test_param", min_val=0.0, max_val=10.0)


class TestColorTupleValidation:
    """Test color tuple validation."""
    
    def test_validate_color_rgb_valid(self):
        """Test valid RGB color validation."""
        color = (0.5, 0.7, 0.9)
        result = validate.validate_color_tuple(color, "test_color", num_channels=3)
        assert result == (0.5, 0.7, 0.9)
        assert len(result) == 3
        
        # List input
        color_list = [0.1, 0.2, 0.3]
        result = validate.validate_color_tuple(color_list, "test_color", num_channels=3)
        assert result == (0.1, 0.2, 0.3)
    
    def test_validate_color_rgba_valid(self):
        """Test valid RGBA color validation."""
        color = (1.0, 0.5, 0.0, 0.8)
        result = validate.validate_color_tuple(color, "test_color", num_channels=4)
        assert result == (1.0, 0.5, 0.0, 0.8)
        assert len(result) == 4
    
    def test_validate_color_type_error(self):
        """Test color tuple type errors."""
        with pytest.raises(TypeError, match="must be tuple, list, or array"):
            validate.validate_color_tuple("red", "test_color")
    
    def test_validate_color_length_error(self):
        """Test color tuple length errors."""
        # Wrong number of channels
        with pytest.raises(ValueError, match="must have 3 channels"):
            validate.validate_color_tuple((1.0, 0.5), "test_color", num_channels=3)
        
        with pytest.raises(ValueError, match="must have 4 channels"):
            validate.validate_color_tuple((1.0, 0.5, 0.2), "test_color", num_channels=4)
    
    def test_validate_color_range_error(self):
        """Test color tuple range validation."""
        # Out of range values
        with pytest.raises(ValueError, match=r"out of range.*not in \[0\.0, 1\.0\]"):
            validate.validate_color_tuple((1.5, 0.5, 0.2), "test_color")
        
        with pytest.raises(ValueError, match=r"out of range.*not in \[0\.0, 1\.0\]"):
            validate.validate_color_tuple((-0.1, 0.5, 0.2), "test_color")
    
    def test_validate_color_custom_range(self):
        """Test color tuple with custom value range."""
        color = (128, 64, 192)
        result = validate.validate_color_tuple(
            color, "test_color", num_channels=3, value_range=(0, 255)
        )
        assert result == (128.0, 64.0, 192.0)
        
        # Out of custom range
        with pytest.raises(ValueError, match=r"out of range.*not in \[0, 255\]"):
            validate.validate_color_tuple((300, 64, 192), "test_color", value_range=(0, 255))


class TestStringChoiceValidation:
    """Test string choice validation."""
    
    def test_validate_string_choice_valid(self):
        """Test valid string choice validation."""
        choices = ["low", "medium", "high", "ultra"]
        result = validate.validate_string_choice("medium", "quality", choices)
        assert result == "medium"
    
    def test_validate_string_choice_invalid(self):
        """Test invalid string choice validation."""
        choices = ["low", "medium", "high", "ultra"]
        
        # Invalid choice
        with pytest.raises(ValueError, match="invalid choice.*'extreme'"):
            validate.validate_string_choice("extreme", "quality", choices)
        
        # Wrong type
        with pytest.raises(TypeError, match="must be string"):
            validate.validate_string_choice(123, "quality", choices)
    
    def test_validate_string_choice_case_insensitive(self):
        """Test case-insensitive string choice validation."""
        choices = ["Low", "Medium", "High", "Ultra"]
        
        # Case insensitive should work
        result = validate.validate_string_choice(
            "medium", "quality", choices, case_sensitive=False
        )
        assert result == "medium"  # Original case preserved
        
        # Case sensitive should fail
        with pytest.raises(ValueError, match="invalid choice"):
            validate.validate_string_choice("medium", "quality", choices, case_sensitive=True)


class TestContextualErrorMessages:
    """Test that error messages include helpful context."""
    
    def test_context_in_error_messages(self):
        """Test that context appears in error messages."""
        arr = np.array([1, 2, 3], dtype=bool)
        
        with pytest.raises(ValueError, match="PBR material texture validation"):
            validate.validate_array(
                arr, "texture_data", 
                context="PBR material texture validation"
            )
    
    def test_detailed_error_information(self):
        """Test that errors contain detailed diagnostic information."""
        arr = np.array([[1, 2, 3, 4]], dtype=np.float32)  # Wrong shape for RGB
        
        with pytest.raises(ValueError) as exc_info:
            validate.validate_array(arr, "rgb_data", shape=validate.SHAPE_3D_RGB)
        
        error_msg = str(exc_info.value)
        assert "rgb_data" in error_msg
        assert "3D RGB array" in error_msg
        assert "(H, W, 3)" in error_msg
        assert str(arr.shape) in error_msg


class TestIntegrationWithAPIs:
    """Test integration of validation with existing APIs."""
    
    def test_dem_normalize_validation(self):
        """Test that dem_normalize uses validation (integration test)."""
        try:
            import forge3d
        except ImportError:
            pytest.skip("forge3d module not available")
        
        # Valid input
        valid_dem = np.random.rand(100, 200).astype(np.float32)
        result = forge3d.dem_normalize(valid_dem)
        assert result.shape == (100, 200)
        
        # Invalid input (should be caught by existing validation or enhanced)
        # This test documents current behavior and can be enhanced
        invalid_dem = "not_an_array"
        with pytest.raises((TypeError, ValueError)):
            forge3d.dem_normalize(invalid_dem)


def test_validation_module_import():
    """Test that validation module imports correctly."""
    import forge3d._validate as validate
    
    # Test that key functions are available
    assert hasattr(validate, 'validate_array')
    assert hasattr(validate, 'validate_numeric_parameter') 
    assert hasattr(validate, 'validate_color_tuple')
    assert hasattr(validate, 'validate_string_choice')
    
    # Test that constants are defined
    assert hasattr(validate, 'SHAPE_2D')
    assert hasattr(validate, 'SHAPE_3D_RGB')
    assert hasattr(validate, 'SHAPE_3D_RGBA')
    assert hasattr(validate, 'SHAPE_HEIGHT_FIELD')


if __name__ == "__main__":
    pytest.main([__file__])