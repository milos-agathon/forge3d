"""
Test zero-copy NumPy interop for RGBA output and height input.

This test suite validates that zero-copy pathways are working correctly
by comparing memory buffer pointers between NumPy arrays and their
underlying Rust-owned backing stores.
"""
import pytest
import numpy as np
import ctypes

# Import forge3d and validation helpers
import forge3d as f3d


def get_numpy_data_ptr(arr):
    """Get the underlying data pointer from a NumPy array."""
    return arr.ctypes.data


class TestZeroCopyRGBAOutput:
    """Test zero-copy pathways for RGBA output from render operations."""
    
    def test_render_triangle_rgba_zero_copy_via_renderer_method(self):
        """Test that Renderer.render_triangle_rgba() returns zero-copy NumPy array."""
        renderer = f3d.Renderer(32, 32)
        
        # Use the test hook to get both array and pointer
        rgba_array, rust_ptr = renderer.render_triangle_rgba_with_ptr()
        
        # Validate output shape and dtype
        assert rgba_array.shape == (32, 32, 4)
        assert rgba_array.dtype == np.uint8
        assert rgba_array.flags['C_CONTIGUOUS']
        
        # Get NumPy array pointer
        numpy_ptr = get_numpy_data_ptr(rgba_array)
        
        # Assert zero-copy: NumPy array data pointer should match Rust pointer
        assert numpy_ptr == rust_ptr, f"Zero-copy validation failed: NumPy ptr=0x{numpy_ptr:x}, Rust ptr=0x{rust_ptr:x}"
        assert numpy_ptr != 0, "NumPy array should have valid data pointer"
        
    def test_module_level_render_triangle_rgba_zero_copy(self):
        """Test that module-level render_triangle_rgba function provides zero-copy output."""
        rgba_array = f3d.render_triangle_rgba(16, 16)
        
        # Validate output shape and dtype
        assert rgba_array.shape == (16, 16, 4)
        assert rgba_array.dtype == np.uint8
        assert rgba_array.flags['C_CONTIGUOUS']
        
        # Verify we have a valid data pointer
        ptr = get_numpy_data_ptr(rgba_array)
        assert ptr != 0, "NumPy array should have valid data pointer"
        
    def test_rgba_output_buffer_reuse_across_sizes(self):
        """Test that RGBA output buffer reuse works correctly for different sizes."""
        renderer = f3d.Renderer(64, 64)
        
        # First render at initial size
        rgba1 = renderer.render_triangle_rgba()
        ptr1 = get_numpy_data_ptr(rgba1)
        
        # Second render at same size - buffer should be reusable
        rgba2 = renderer.render_triangle_rgba()
        ptr2 = get_numpy_data_ptr(rgba2)
        
        assert rgba1.shape == rgba2.shape
        assert ptr1 != 0 and ptr2 != 0
        
        # Content should be deterministic (same triangle)
        np.testing.assert_array_equal(rgba1, rgba2)


class TestZeroCopyHeightInput:
    """Test zero-copy pathways for height input to terrain operations."""
    
    def test_height_input_float32_zero_copy_validation(self):
        """Test that float32 heightmap input uses zero-copy pathway."""
        renderer = f3d.Renderer(32, 32)
        
        # Create a C-contiguous float32 heightmap
        height, width = 8, 8
        heightmap = np.arange(height * width, dtype=np.float32).reshape((height, width))
        assert heightmap.flags['C_CONTIGUOUS'], "Test heightmap should be C-contiguous"
        
        # Store original pointer for comparison
        input_ptr = get_numpy_data_ptr(heightmap)
        assert input_ptr != 0, "Input heightmap should have valid data pointer"
        
        # Add terrain - this should capture the source pointer for zero-copy validation
        renderer.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
        
        # Check that debug hook captured the correct source pointer
        captured_ptr = renderer.debug_last_height_src_ptr()
        assert captured_ptr == input_ptr, f"Zero-copy validation failed: input ptr=0x{input_ptr:x}, captured ptr=0x{captured_ptr:x}"
        
        # Upload to GPU and read back to validate roundtrip
        renderer.upload_height_r32f()
        readback = renderer.read_full_height_texture()
        
        # Validate roundtrip accuracy
        assert readback.shape == (height, width)
        np.testing.assert_allclose(readback, heightmap, rtol=1e-6)
        
    def test_height_input_float64_conversion_path(self):
        """Test that float64 heightmap input goes through conversion (not zero-copy)."""
        renderer = f3d.Renderer(32, 32)
        
        # Create a C-contiguous float64 heightmap
        height, width = 6, 6  
        heightmap_f64 = np.random.rand(height, width).astype(np.float64)
        assert heightmap_f64.flags['C_CONTIGUOUS']
        assert heightmap_f64.dtype == np.float64
        
        input_ptr = get_numpy_data_ptr(heightmap_f64)
        assert input_ptr != 0
        
        # Add terrain - float64 requires conversion so not zero-copy
        renderer.add_terrain(heightmap_f64, spacing=(1.0, 1.0), exaggeration=1.0, colormap="terrain")
        
        # Upload and validate roundtrip (should convert to float32 internally)
        renderer.upload_height_r32f()
        readback = renderer.read_full_height_texture()
        
        # Validate shape and approximate equality (float64 -> float32 conversion)
        assert readback.shape == (height, width)
        np.testing.assert_allclose(readback, heightmap_f64, rtol=1e-6)
        
    def test_height_input_non_contiguous_error(self):
        """Test that non-contiguous heightmap input is rejected with clear error."""
        renderer = f3d.Renderer(16, 16)
        
        # Create a non-contiguous array (transpose creates view with different strides)
        base_array = np.arange(20).reshape((4, 5)).astype(np.float32)
        non_contiguous = base_array.T  # This creates a non-contiguous view
        
        assert not non_contiguous.flags['C_CONTIGUOUS'], "Test array should be non-contiguous"
        
        # Should raise error about array format (contiguity or type validation)
        with pytest.raises(RuntimeError, match="C-contiguous|dtype float32 or float64"):
            renderer.add_terrain(non_contiguous, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
            
    def test_height_roundtrip_zero_copy_validation(self):
        """Test that height data roundtrip maintains data integrity with zero-copy input."""
        renderer = f3d.Renderer(64, 64)
        
        # Create deterministic test data
        height, width = 10, 12
        original_heights = np.linspace(0, 100, height * width, dtype=np.float32).reshape((height, width))
        
        # Ensure C-contiguous for zero-copy path
        assert original_heights.flags['C_CONTIGUOUS']
        
        # Get pointer for reference
        input_ptr = get_numpy_data_ptr(original_heights)
        
        # Process through terrain system
        renderer.add_terrain(original_heights, spacing=(2.0, 2.0), exaggeration=1.0, colormap="magma")
        renderer.upload_height_r32f()
        
        # Read back and validate
        readback_heights = renderer.read_full_height_texture()
        readback_ptr = get_numpy_data_ptr(readback_heights)
        
        # Validate data integrity
        assert readback_heights.shape == original_heights.shape
        np.testing.assert_allclose(readback_heights, original_heights, rtol=1e-6, atol=1e-6)
        
        # Pointers should be different (input vs readback buffer)
        assert input_ptr != readback_ptr, "Input and readback should use different buffers"
        assert input_ptr != 0 and readback_ptr != 0, "Both pointers should be valid"


class TestZeroCopyEdgeCases:
    """Test edge cases and error conditions for zero-copy pathways."""
    
    def test_empty_heightmap_error(self):
        """Test that empty heightmaps are rejected."""
        renderer = f3d.Renderer(16, 16)
        
        empty_array = np.array([], dtype=np.float32).reshape((0, 0))
        
        with pytest.raises(RuntimeError, match="cannot be empty"):
            renderer.add_terrain(empty_array, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
            
    def test_invalid_dtype_heightmap_error(self):
        """Test that invalid dtype heightmaps are rejected with helpful error."""
        renderer = f3d.Renderer(16, 16)
        
        # Integer array should be rejected
        int_array = np.arange(16, dtype=np.int32).reshape((4, 4))
        
        with pytest.raises(RuntimeError, match="float32 or float64"):
            renderer.add_terrain(int_array, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
            
    def test_height_texture_read_before_upload(self):
        """Test that reading height texture before upload returns appropriate data."""
        renderer = f3d.Renderer(16, 16)
        
        # Create and add terrain but don't upload
        heightmap = np.ones((4, 4), dtype=np.float32)
        renderer.add_terrain(heightmap, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
        
        # According to test contract, should return zeros before upload
        with pytest.raises(RuntimeError, match="no height texture uploaded"):
            renderer.read_full_height_texture()
            
    def test_read_height_without_terrain_error(self):
        """Test that reading height texture without terrain raises appropriate error."""
        renderer = f3d.Renderer(16, 16)
        
        with pytest.raises(RuntimeError, match="no terrain uploaded"):
            renderer.read_full_height_texture()


class TestMemoryPointerValidation:
    """Test memory pointer validation and comparison utilities."""
    
    def test_numpy_pointer_extraction(self):
        """Test that we can reliably extract data pointers from NumPy arrays."""
        # Test with different array types
        arrays = [
            np.ones((4, 4), dtype=np.float32),
            np.zeros((3, 5, 4), dtype=np.uint8),
            np.arange(10, dtype=np.int32),
        ]
        
        for arr in arrays:
            ptr = get_numpy_data_ptr(arr)
            assert isinstance(ptr, int), "Pointer should be an integer"
            assert ptr != 0, "Valid arrays should have non-zero pointers"
            
    def test_pointer_stability_across_views(self):
        """Test pointer behavior with array views and slices."""
        base_array = np.arange(100, dtype=np.float32).reshape((10, 10))
        base_ptr = get_numpy_data_ptr(base_array)
        
        # Views should have different pointers (different starting positions)
        view = base_array[2:8, 2:8]
        view_ptr = get_numpy_data_ptr(view)
        
        # Both should be valid pointers
        assert base_ptr != 0 and view_ptr != 0
        
        # For contiguous views, pointer arithmetic should be predictable
        # (though exact relationship depends on strides and starting position)
        if view.flags['C_CONTIGUOUS']:
            # View starts at [2,2] = index 22, so offset should be 22*4 bytes
            expected_offset = 22 * 4  # 22 elements * 4 bytes per float32
            assert view_ptr == base_ptr + expected_offset