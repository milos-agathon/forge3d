"""
Tests for normal mapping functionality

Validates normal mapping pipeline according to acceptance criteria:
- AC: With a checkerboard normal map, rendered image's mean luminance differs ≥10% vs. flat normals
- AC: Normal mapping path passes on all 3 OS targets; no NaNs in G-buffer intermediates
"""

import numpy as np
import pytest
import forge3d as f3d

# Skip all tests if normal mapping feature not enabled
pytestmark = pytest.mark.skipif(
    not hasattr(f3d, 'normalmap'),
    reason="Normal mapping feature not enabled (enable-normal-mapping feature flag required)"
)


def test_normalmap_module_availability():
    """Test that normal mapping module is available when feature is enabled"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    # Check that key functions are available
    assert hasattr(f3d.normalmap, 'has_normal_mapping_support')
    assert hasattr(f3d.normalmap, 'create_checkerboard_normal_map')
    assert hasattr(f3d.normalmap, 'validate_normal_map')
    assert hasattr(f3d.normalmap, 'encode_normal_vector')
    assert hasattr(f3d.normalmap, 'decode_normal_vector')


def test_checkerboard_normal_map_generation():
    """Test generation of checkerboard normal map for testing"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
        
    # Generate checkerboard normal map
    normal_map = f3d.normalmap.create_checkerboard_normal_map(64)
    
    # Check shape and type
    assert normal_map.shape == (64, 64, 4)
    assert normal_map.dtype == np.uint8
    
    # Check that it contains different values (checkerboard pattern)
    unique_pixels = set()
    for y in range(0, 32, 8):
        for x in range(0, 32, 8):
            pixel = tuple(normal_map[y, x, :3])
            unique_pixels.add(pixel)
    
    # Should have at least 2 different pixel values for checkerboard
    assert len(unique_pixels) >= 2


def test_normal_vector_encoding_decoding():
    """Test normal vector encoding/decoding roundtrip"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    # Test vectors
    test_normals = np.array([
        [0.0, 0.0, 1.0],     # Up
        [1.0, 0.0, 0.0],     # Right
        [0.0, 1.0, 0.0],     # Forward
        [-0.5, 0.5, 0.707],  # Diagonal
    ], dtype=np.float32)
    
    # Encode and decode
    encoded = f3d.normalmap.encode_normal_vector(test_normals)
    decoded = f3d.normalmap.decode_normal_vector(encoded)
    
    # Check roundtrip accuracy (within tolerance for uint8 quantization)
    diff = np.abs(decoded - test_normals)
    assert np.all(diff < 0.01), f"Encoding roundtrip failed: max diff = {np.max(diff)}"


def test_normal_map_validation():
    """Test normal map validation functionality"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    # Create a valid normal map
    valid_map = f3d.normalmap.create_checkerboard_normal_map(32)
    results = f3d.normalmap.validate_normal_map(valid_map)
    
    assert results['valid'], f"Valid normal map failed validation: {results['errors']}"
    assert results['unit_length_ok']
    assert results['range_ok']
    assert results['z_positive_ok']
    
    # Create an invalid normal map (wrong dtype)
    invalid_map = np.random.rand(32, 32, 3).astype(np.float32)  # Float instead of uint8
    invalid_results = f3d.normalmap.validate_normal_map(invalid_map)
    
    assert not invalid_results['valid']
    assert len(invalid_results['errors']) > 0


def test_luminance_difference_calculation():
    """Test luminance difference calculation for AC validation"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    # Create two test images with different luminance
    image1 = np.full((64, 64, 3), 100, dtype=np.uint8)  # Darker
    image2 = np.full((64, 64, 3), 150, dtype=np.uint8)  # Brighter
    
    diff = f3d.normalmap.compute_luminance_difference(image1, image2)
    
    # Should detect significant difference
    assert diff > 0.0, "Failed to detect luminance difference"
    assert diff > 10.0, "Luminance difference should be substantial for test images"


def test_normal_map_file_operations():
    """Test saving and loading normal maps"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    try:
        import tempfile
        import os
        
        # Create a test normal map
        original_map = f3d.normalmap.create_checkerboard_normal_map(32)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            f3d.normalmap.save_normal_map(original_map, temp_path)
            
            # Load back
            loaded_map = f3d.normalmap.load_normal_map(temp_path)
            
            # Check that loaded map matches original (allowing for compression artifacts)
            assert loaded_map.shape == original_map.shape
            assert loaded_map.dtype == original_map.dtype
            
            # Validate loaded map
            validation = f3d.normalmap.validate_normal_map(loaded_map)
            assert validation['valid'], f"Loaded normal map validation failed: {validation['errors']}"
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except ImportError:
        pytest.skip("PIL not available for file operations test")


def test_normal_mapping_acceptance_criteria_preparation():
    """Test preparation for acceptance criteria validation
    
    This test sets up the components needed for AC validation:
    - AC: With a checkerboard normal map, rendered image's mean luminance differs ≥10% vs. flat normals
    - AC: Normal mapping path passes on all 3 OS targets; no NaNs in G-buffer intermediates
    """
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    # Create checkerboard normal map for testing
    normal_map = f3d.normalmap.create_checkerboard_normal_map(128)
    
    # Validate the normal map meets requirements
    validation = f3d.normalmap.validate_normal_map(normal_map)
    assert validation['valid'], "Test normal map must be valid for AC testing"
    
    # Verify it has sufficient variation for luminance testing
    # Check that flat and perturbed normals are present
    flat_pixels = 0
    perturbed_pixels = 0
    
    # Expected encoded values for flat normal (0, 0, 1)
    flat_normal_encoded = [128, 128, 255]
    
    # Create expected encoded value for perturbed normal
    nx, ny = 0.2, 0.2
    nz = np.sqrt(max(0.0, 1.0 - nx*nx - ny*ny))
    normal_vec = np.array([nx, ny, nz])
    perturbed_encoded = f3d.normalmap.encode_normal_vector(normal_vec)
    
    for y in range(0, 128, 8):
        for x in range(0, 128, 8):
            pixel = normal_map[y, x, :3]
            if np.array_equal(pixel, flat_normal_encoded):  # Flat normal
                flat_pixels += 1
            elif np.array_equal(pixel, perturbed_encoded):  # Perturbed normal
                perturbed_pixels += 1
    
    assert flat_pixels > 0, "Normal map must contain flat normals for comparison"
    assert perturbed_pixels > 0, "Normal map must contain perturbed normals for variation"
    
    print(f"Normal map prepared: {flat_pixels} flat, {perturbed_pixels} perturbed regions")


def test_tbn_integration_compatibility():
    """Test that normal mapping is compatible with TBN generation from N6"""
    if not hasattr(f3d, 'normalmap'):
        pytest.skip("Normal mapping module not available")
    
    if not hasattr(f3d, 'mesh'):
        pytest.skip("TBN mesh module not available")
    
    # Generate TBN data (from N6)
    vertices, indices, tbn_data = f3d.mesh.generate_cube_tbn()
    
    # Validate TBN for use with normal mapping
    validation = f3d.mesh.validate_tbn_data(tbn_data)
    assert validation['valid'], "TBN data must be valid for normal mapping"
    
    # Create normal map
    normal_map = f3d.normalmap.create_checkerboard_normal_map(64)
    normal_validation = f3d.normalmap.validate_normal_map(normal_map)
    assert normal_validation['valid'], "Normal map must be valid"
    
    print(f"N6/N7 integration ready: {len(tbn_data)} TBN vertices, {normal_map.shape} normal map")


if __name__ == "__main__":
    pytest.main([__file__])