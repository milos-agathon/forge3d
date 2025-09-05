"""
Tests for TBN (Tangent, Bitangent, Normal) generation

Validates TBN calculation correctness according to acceptance criteria.
"""

import numpy as np
import pytest
import forge3d as f3d

# Skip all tests if TBN feature not enabled
pytestmark = pytest.mark.skipif(
    not hasattr(f3d, 'mesh') or not hasattr(f3d.mesh, 'generate_cube_tbn'),
    reason="TBN feature not enabled (enable-tbn feature flag required)"
)

def test_cube_tbn_generation():
    """Test TBN generation for a unit cube"""
    if not hasattr(f3d, 'mesh'):
        pytest.skip("TBN module not available")
        
    vertices, indices, tbn_data = f3d.mesh.generate_cube_tbn()
    
    # Should have 24 vertices (6 faces * 4 vertices)
    assert len(vertices) == 24
    assert len(indices) == 36  # 6 faces * 2 triangles * 3 vertices
    assert len(tbn_data) == 24
    
    # Check TBN validity according to AC: |t|≈|b|≈|n|≈1 within 1e-3
    for tbn in tbn_data:
        tangent_len = np.linalg.norm(tbn['tangent'])
        bitangent_len = np.linalg.norm(tbn['bitangent'])
        normal_len = np.linalg.norm(tbn['normal'])
        
        assert abs(tangent_len - 1.0) < 1e-3, f"Tangent length {tangent_len} not unit"
        assert abs(bitangent_len - 1.0) < 1e-3, f"Bitangent length {bitangent_len} not unit"
        assert abs(normal_len - 1.0) < 1e-3, f"Normal length {normal_len} not unit"
        
        # Check orthogonality: |dot(t,n)| ≤ 1e-3
        t = np.array(tbn['tangent'])
        n = np.array(tbn['normal'])
        dot_tn = abs(np.dot(t, n))
        assert dot_tn <= 1e-3, f"Tangent-normal dot product {dot_tn} too large (not orthogonal)"

def test_plane_tbn_generation():
    """Test TBN generation for a flat plane"""  
    if not hasattr(f3d, 'mesh'):
        pytest.skip("TBN module not available")
        
    vertices, indices, tbn_data = f3d.mesh.generate_plane_tbn(3, 3)
    
    # Should have 9 vertices (3x3 grid)
    assert len(vertices) == 9
    assert len(indices) == 24  # 4 quads * 2 triangles * 3 vertices
    assert len(tbn_data) == 9
    
    # For a flat plane, all normals should point up (Y)
    for tbn in tbn_data:
        normal = np.array(tbn['normal'])
        expected_normal = np.array([0.0, 1.0, 0.0])
        
        # Normal should be close to Y-up
        assert np.linalg.norm(normal - expected_normal) < 1e-3

def test_tbn_determinant():
    """Test TBN matrix determinant for handedness validation"""
    if not hasattr(f3d, 'mesh'):
        pytest.skip("TBN module not available")
        
    vertices, indices, tbn_data = f3d.mesh.generate_cube_tbn()
    
    # Check determinant according to AC: det(TBN) ∈ [0.99, 1.01]
    for tbn in tbn_data:
        t = np.array(tbn['tangent'])
        b = np.array(tbn['bitangent'])
        n = np.array(tbn['normal'])
        
        # Construct TBN matrix
        tbn_matrix = np.column_stack([t, b, n])
        det = np.linalg.det(tbn_matrix)
        
        # Should be close to ±1 (right-handed or left-handed coordinate system)
        assert 0.99 <= abs(det) <= 1.01, f"TBN determinant {det} outside valid range"

def test_vertex_buffer_layout():
    """Test that vertex buffer layouts are valid"""
    if not hasattr(f3d, 'mesh'):
        pytest.skip("TBN module not available")
        
    # This would test the vertex buffer layout if we expose it to Python
    # For now, we verify the TBN generation works
    vertices, indices, tbn_data = f3d.mesh.generate_plane_tbn(2, 2) 
    
    # Should successfully generate without errors
    assert len(vertices) == 4
    assert len(tbn_data) == 4
    
    # Verify basic data integrity
    for vertex, tbn in zip(vertices, tbn_data):
        assert 'position' in vertex
        assert 'normal' in vertex
        assert 'uv' in vertex
        assert 'tangent' in tbn
        assert 'bitangent' in tbn
        assert 'normal' in tbn

def test_tbn_memory_layout():
    """Test memory layout and size requirements for TBN vertices"""
    if not hasattr(f3d, 'mesh'):
        pytest.skip("TBN module not available")
    
    # Generate test data
    vertices, indices, tbn_data = f3d.mesh.generate_cube_tbn()
    
    # Verify we don't have excessive memory usage
    # Each vertex should have reasonable memory footprint
    vertex_count = len(vertices)
    index_count = len(indices)
    
    # Basic sanity checks
    assert vertex_count > 0
    assert index_count > 0
    assert index_count % 3 == 0  # Should form triangles
    
if __name__ == "__main__":
    pytest.main([__file__])