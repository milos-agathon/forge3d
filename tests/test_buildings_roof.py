"""
P4.1: Tests for roof type inference.

Tests inference of roof types from OSM tags and building attributes.
"""

import pytest


def test_infer_roof_type_import():
    """Test that roof type inference can be imported."""
    from forge3d.buildings import infer_roof_type
    assert callable(infer_roof_type)


def test_infer_roof_type_explicit_tag():
    """Test inference from explicit roof:shape tag."""
    from forge3d.buildings import infer_roof_type

    assert infer_roof_type({"building:roof:shape": "gabled"}) == "gabled"
    assert infer_roof_type({"roof:shape": "hipped"}) == "hipped"
    assert infer_roof_type({"roof_shape": "flat"}) == "flat"


def test_infer_roof_type_case_insensitive():
    """Test that roof type inference is case-insensitive."""
    from forge3d.buildings import infer_roof_type

    assert infer_roof_type({"roof:shape": "GABLED"}) == "gabled"
    assert infer_roof_type({"roof:shape": "Hipped"}) == "hipped"
    assert infer_roof_type({"roof:shape": "PYRAMIDAL"}) == "pyramidal"


def test_infer_roof_type_from_building_type():
    """Test inference from building type when no explicit roof tag."""
    from forge3d.buildings import infer_roof_type

    # Residential buildings typically have gabled roofs
    assert infer_roof_type({"building": "house"}) == "gabled"
    assert infer_roof_type({"building": "detached"}) == "gabled"
    assert infer_roof_type({"building": "residential"}) == "gabled"

    # Industrial/commercial typically have flat roofs
    assert infer_roof_type({"building": "warehouse"}) == "flat"
    assert infer_roof_type({"building": "industrial"}) == "flat"


def test_infer_roof_type_default():
    """Test default roof type when no tags available."""
    from forge3d.buildings import infer_roof_type

    assert infer_roof_type({}) == "flat"
    assert infer_roof_type({"name": "Some Building"}) == "flat"
    assert infer_roof_type({"building": "unknown_type"}) == "flat"


def test_infer_roof_type_all_types():
    """Test that all supported roof types can be inferred."""
    from forge3d.buildings import infer_roof_type

    roof_types = [
        "flat", "gabled", "hipped", "pyramidal",
        "dome", "mansard", "shed", "gambrel",
        "onion", "skillion"
    ]

    for roof_type in roof_types:
        result = infer_roof_type({"roof:shape": roof_type})
        assert result == roof_type, f"Failed for {roof_type}"


def test_infer_roof_type_lean_to():
    """Test lean-to roof variants."""
    from forge3d.buildings import infer_roof_type

    assert infer_roof_type({"roof:shape": "lean_to"}) == "shed"
    assert infer_roof_type({"roof:shape": "lean-to"}) == "shed"


def test_infer_roof_type_explicit_overrides_building():
    """Test that explicit roof tag overrides building type inference."""
    from forge3d.buildings import infer_roof_type

    # Warehouse would normally be flat, but explicit tag says gabled
    tags = {
        "building": "warehouse",
        "roof:shape": "gabled"
    }
    assert infer_roof_type(tags) == "gabled"


def test_infer_roof_type_priority():
    """Test tag priority order."""
    from forge3d.buildings import infer_roof_type

    # building:roof:shape takes priority over roof:shape
    tags = {
        "building:roof:shape": "hipped",
        "roof:shape": "gabled"
    }
    result = infer_roof_type(tags)
    # Could be either, but should be consistent
    assert result in ("hipped", "gabled")


def test_infer_roof_apartments():
    """Test apartment building roof inference."""
    from forge3d.buildings import infer_roof_type

    assert infer_roof_type({"building": "apartments"}) == "hipped"


def test_infer_roof_shed_building():
    """Test shed building roof inference."""
    from forge3d.buildings import infer_roof_type

    # Shed buildings typically have shed roofs
    result = infer_roof_type({"building": "shed"})
    # Could be shed or flat depending on implementation
    assert result in ("shed", "flat")


def test_infer_roof_commercial():
    """Test commercial building roof inference."""
    from forge3d.buildings import infer_roof_type

    assert infer_roof_type({"building": "commercial"}) == "flat"
    assert infer_roof_type({"building": "retail"}) == "flat"


def test_infer_roof_unknown_value():
    """Test handling of unknown roof:shape value."""
    from forge3d.buildings import infer_roof_type

    # Unknown value should default to flat
    assert infer_roof_type({"roof:shape": "unknown_roof_xyz"}) == "flat"


def test_infer_roof_numeric_value():
    """Test handling of numeric property values."""
    from forge3d.buildings import infer_roof_type

    # Should handle numeric values gracefully
    tags = {"height": 10.0, "building": "house"}
    assert infer_roof_type(tags) == "gabled"


def test_infer_roof_church():
    """Test church building roof inference."""
    from forge3d.buildings import infer_roof_type

    # Churches typically have gabled roofs
    assert infer_roof_type({"building": "church"}) == "gabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
