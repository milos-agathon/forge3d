"""Tests for MaterialSet class (Task 2.1)."""
import pytest
import forge3d as f3d


def test_material_set_creation():
    """Test basic MaterialSet creation with default parameters."""
    materials = f3d.MaterialSet.terrain_default()
    assert materials is not None
    assert materials.material_count >= 3  # At least rock, grass, snow


def test_material_set_custom_params():
    """Test MaterialSet with custom parameters."""
    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=10.0,
        normal_strength=2.0,
        blend_sharpness=8.0
    )
    assert materials is not None
    assert materials.triplanar_scale == 10.0
    assert materials.normal_strength == 2.0
    assert materials.blend_sharpness == 8.0


def test_material_set_validation_scale():
    """Test that invalid triplanar_scale raises ValueError."""
    with pytest.raises(ValueError, match="triplanar_scale must be > 0"):
        f3d.MaterialSet.terrain_default(triplanar_scale=0.0)

    with pytest.raises(ValueError, match="triplanar_scale must be > 0"):
        f3d.MaterialSet.terrain_default(triplanar_scale=-1.0)


def test_material_set_validation_normal_strength():
    """Test that invalid normal_strength raises ValueError."""
    with pytest.raises(ValueError, match="normal_strength must be >= 0"):
        f3d.MaterialSet.terrain_default(normal_strength=-0.5)


def test_material_set_validation_blend_sharpness():
    """Test that invalid blend_sharpness raises ValueError."""
    with pytest.raises(ValueError, match="blend_sharpness must be > 0"):
        f3d.MaterialSet.terrain_default(blend_sharpness=0.0)


def test_material_set_properties():
    """Test MaterialSet property getters."""
    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=5.5,
        normal_strength=1.5,
        blend_sharpness=3.5
    )

    assert materials.material_count > 0
    assert materials.triplanar_scale == 5.5
    assert materials.normal_strength == 1.5
    assert materials.blend_sharpness == 3.5


def test_material_set_repr():
    """Test MaterialSet string representation."""
    materials = f3d.MaterialSet.terrain_default()
    repr_str = repr(materials)

    assert "MaterialSet" in repr_str
    assert "materials=" in repr_str
    assert "triplanar_scale=" in repr_str
