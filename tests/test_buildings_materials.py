"""
P4.2: Tests for building material inference and presets.

Tests material differentiation based on OSM tags and material presets.
"""

import pytest


def test_material_from_name_brick():
    """Test brick material preset."""
    from forge3d.buildings import material_from_name

    mat = material_from_name("brick")

    # Brick should be reddish-brown
    assert mat.albedo[0] > mat.albedo[1]  # Red > Green
    assert mat.albedo[0] > mat.albedo[2]  # Red > Blue
    assert mat.roughness > 0.5  # Rough surface
    assert mat.metallic < 0.1  # Non-metallic


def test_material_from_name_glass():
    """Test glass material preset."""
    from forge3d.buildings import material_from_name

    mat = material_from_name("glass")

    # Glass should be dark (low albedo), smooth
    assert mat.albedo[0] < 0.1
    assert mat.roughness < 0.2  # Very smooth
    assert mat.metallic < 0.1  # Non-metallic


def test_material_from_name_steel():
    """Test steel material preset."""
    from forge3d.buildings import material_from_name

    mat = material_from_name("steel")

    # Steel should be metallic
    assert mat.metallic > 0.5
    assert mat.roughness < 0.5  # Relatively smooth


def test_material_from_name_concrete():
    """Test concrete material preset."""
    from forge3d.buildings import material_from_name

    mat = material_from_name("concrete")

    # Concrete should be gray, rough
    assert 0.4 < mat.albedo[0] < 0.8
    assert abs(mat.albedo[0] - mat.albedo[1]) < 0.1  # Gray (R ~ G ~ B)
    assert mat.roughness > 0.5
    assert mat.metallic < 0.1


def test_material_from_name_wood():
    """Test wood material preset."""
    from forge3d.buildings import material_from_name

    mat = material_from_name("wood")

    # Wood should be brown-ish
    assert mat.albedo[0] > mat.albedo[2]  # More red than blue
    assert mat.roughness > 0.5
    assert mat.metallic < 0.1


def test_material_from_name_unknown():
    """Test unknown material returns default."""
    from forge3d.buildings import material_from_name, BuildingMaterial

    mat = material_from_name("unknown_material_xyz")
    default = BuildingMaterial()

    # Should return default material (use approx comparison for f32 precision)
    assert abs(mat.albedo[0] - default.albedo[0]) < 0.01
    assert abs(mat.albedo[1] - default.albedo[1]) < 0.01
    assert abs(mat.albedo[2] - default.albedo[2]) < 0.01
    assert abs(mat.roughness - default.roughness) < 0.01


def test_material_from_tags_explicit():
    """Test material inference from explicit tag."""
    from forge3d.buildings import material_from_tags

    tags = {"building:material": "brick"}
    mat = material_from_tags(tags)

    # Should return brick material
    assert mat.albedo[0] > mat.albedo[2]  # Reddish


def test_material_from_tags_building_type():
    """Test material inference from building type."""
    from forge3d.buildings import material_from_tags

    # Warehouse should be concrete
    tags = {"building": "warehouse"}
    mat = material_from_tags(tags)
    assert mat.roughness > 0.5

    # Office should be glass
    tags = {"building": "office"}
    mat = material_from_tags(tags)
    assert mat.roughness < 0.3


def test_building_material_dataclass():
    """Test BuildingMaterial dataclass."""
    from forge3d.buildings import BuildingMaterial

    mat = BuildingMaterial(
        albedo=(0.5, 0.3, 0.2),
        roughness=0.7,
        metallic=0.1,
    )

    assert mat.albedo == (0.5, 0.3, 0.2)
    assert mat.roughness == 0.7
    assert mat.metallic == 0.1
    assert mat.ior == 1.5  # Default
    assert mat.emissive == 0.0  # Default


def test_building_material_from_dict():
    """Test BuildingMaterial.from_dict()."""
    from forge3d.buildings import BuildingMaterial

    d = {
        "albedo": (0.8, 0.2, 0.1),
        "roughness": 0.9,
        "metallic": 0.0,
        "ior": 1.6,
        "emissive": 0.5,
    }
    mat = BuildingMaterial.from_dict(d)

    assert mat.albedo == (0.8, 0.2, 0.1)
    assert mat.roughness == 0.9
    assert mat.emissive == 0.5


def test_building_material_to_dict():
    """Test BuildingMaterial.to_dict()."""
    from forge3d.buildings import BuildingMaterial

    mat = BuildingMaterial(
        albedo=(0.6, 0.5, 0.4),
        roughness=0.65,
        metallic=0.2,
    )
    d = mat.to_dict()

    assert d["albedo"] == (0.6, 0.5, 0.4)
    assert d["roughness"] == 0.65
    assert d["metallic"] == 0.2


def test_material_differentiation():
    """Test that different materials are distinguishable."""
    from forge3d.buildings import material_from_name

    brick = material_from_name("brick")
    glass = material_from_name("glass")
    steel = material_from_name("steel")
    wood = material_from_name("wood")

    # All should be different
    materials = [brick, glass, steel, wood]
    for i, m1 in enumerate(materials):
        for j, m2 in enumerate(materials):
            if i != j:
                # At least one property should differ significantly
                albedo_diff = sum(abs(a - b) for a, b in zip(m1.albedo, m2.albedo))
                roughness_diff = abs(m1.roughness - m2.roughness)
                metallic_diff = abs(m1.metallic - m2.metallic)
                total_diff = albedo_diff + roughness_diff + metallic_diff
                assert total_diff > 0.1, f"Materials {i} and {j} are too similar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
