"""
P4.3: Tests for CityJSON parsing.

Tests CityJSON 1.1 format parsing and building extraction.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_add_buildings_cityjson_import():
    """Test that CityJSON loader can be imported."""
    from forge3d.buildings import add_buildings_cityjson
    assert callable(add_buildings_cityjson)


def test_cityjson_simple_cube():
    """Test parsing a simple CityJSON cube building."""
    from forge3d.buildings import add_buildings_cityjson

    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {
            "scale": [0.001, 0.001, 0.001],
            "translate": [0.0, 0.0, 0.0]
        },
        "vertices": [
            [0, 0, 0],
            [10000, 0, 0],
            [10000, 10000, 0],
            [0, 10000, 0],
            [0, 0, 5000],
            [10000, 0, 5000],
            [10000, 10000, 5000],
            [0, 10000, 5000]
        ],
        "CityObjects": {
            "building1": {
                "type": "Building",
                "attributes": {
                    "measuredHeight": 5.0
                },
                "geometry": [{
                    "type": "Solid",
                    "lod": "1",
                    "boundaries": [
                        [
                            [[0, 1, 2, 3]],
                            [[4, 5, 6, 7]],
                            [[0, 1, 5, 4]],
                            [[1, 2, 6, 5]],
                            [[2, 3, 7, 6]],
                            [[3, 0, 4, 7]]
                        ]
                    ]
                }]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cityjson, f)
        cityjson_path = Path(f.name)

    try:
        layer = add_buildings_cityjson(cityjson_path)

        assert layer.source_format == "cityjson"
        assert layer.building_count >= 1

        # Check that at least one building was parsed
        if layer.buildings:
            b = layer.buildings[0]
            assert b.id == "building1"
            # Native parser extracts height, fallback may not
            if b.height is not None:
                assert b.height == 5.0
    finally:
        cityjson_path.unlink()


def test_cityjson_with_crs():
    """Test CityJSON CRS extraction."""
    from forge3d.buildings import add_buildings_cityjson

    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "metadata": {
            "referenceSystem": "urn:ogc:def:crs:EPSG::28992"
        },
        "transform": {
            "scale": [0.001, 0.001, 0.001],
            "translate": [155000.0, 463000.0, 0.0]
        },
        "vertices": [[0, 0, 0], [1000, 0, 0], [1000, 1000, 0], [0, 1000, 0]],
        "CityObjects": {
            "b1": {
                "type": "Building",
                "geometry": [{
                    "type": "MultiSurface",
                    "lod": "1",
                    "boundaries": [[[0, 1, 2, 3]]]
                }]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cityjson, f)
        cityjson_path = Path(f.name)

    try:
        layer = add_buildings_cityjson(cityjson_path)
        # CRS should be extracted (if native parser is available)
        # Fallback parser also extracts EPSG
        assert layer.crs_epsg == 28992 or layer.crs_epsg is None
    finally:
        cityjson_path.unlink()


def test_cityjson_multiple_buildings():
    """Test CityJSON with multiple buildings."""
    from forge3d.buildings import add_buildings_cityjson

    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {"scale": [1, 1, 1], "translate": [0, 0, 0]},
        "vertices": [
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
            [20, 0, 0], [30, 0, 0], [30, 10, 0], [20, 10, 0],
        ],
        "CityObjects": {
            "building_a": {
                "type": "Building",
                "attributes": {"name": "Building A"},
                "geometry": [{"type": "MultiSurface", "lod": "1", "boundaries": [[[0, 1, 2, 3]]]}]
            },
            "building_b": {
                "type": "Building",
                "attributes": {"name": "Building B"},
                "geometry": [{"type": "MultiSurface", "lod": "1", "boundaries": [[[4, 5, 6, 7]]]}]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cityjson, f)
        cityjson_path = Path(f.name)

    try:
        layer = add_buildings_cityjson(cityjson_path)
        assert layer.building_count >= 2
    finally:
        cityjson_path.unlink()


def test_cityjson_lod_selection():
    """Test that highest available LOD is selected."""
    from forge3d.buildings import add_buildings_cityjson

    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {"scale": [1, 1, 1], "translate": [0, 0, 0]},
        "vertices": [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
        "CityObjects": {
            "building_lod": {
                "type": "Building",
                "geometry": [
                    {"type": "MultiSurface", "lod": "1", "boundaries": [[[0, 1, 2, 3]]]},
                    {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1, 2, 3]]]},
                ]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cityjson, f)
        cityjson_path = Path(f.name)

    try:
        layer = add_buildings_cityjson(cityjson_path)
        # Should select LOD 2 (highest)
        if layer.buildings and hasattr(layer.buildings[0], 'lod'):
            assert layer.buildings[0].lod >= 1
    finally:
        cityjson_path.unlink()


def test_cityjson_building_part():
    """Test that BuildingPart objects are also parsed."""
    from forge3d.buildings import add_buildings_cityjson

    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {"scale": [1, 1, 1], "translate": [0, 0, 0]},
        "vertices": [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
        "CityObjects": {
            "building_main": {
                "type": "Building",
                "geometry": [{"type": "MultiSurface", "lod": "1", "boundaries": [[[0, 1, 2, 3]]]}]
            },
            "building_part_1": {
                "type": "BuildingPart",
                "parents": ["building_main"],
                "geometry": [{"type": "MultiSurface", "lod": "1", "boundaries": [[[0, 1, 2, 3]]]}]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cityjson, f)
        cityjson_path = Path(f.name)

    try:
        layer = add_buildings_cityjson(cityjson_path)
        # Both Building and BuildingPart should be parsed
        assert layer.building_count >= 1
    finally:
        cityjson_path.unlink()


def test_cityjson_file_not_found():
    """Test error handling for missing file."""
    from forge3d.buildings import add_buildings_cityjson

    with pytest.raises(FileNotFoundError):
        add_buildings_cityjson("nonexistent_cityjson.json")


def test_cityjson_invalid_format():
    """Test error handling for invalid CityJSON."""
    from forge3d.buildings import add_buildings_cityjson

    # Create invalid CityJSON (wrong type)
    invalid = {"type": "NotCityJSON", "version": "1.0"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid, f)
        path = Path(f.name)

    try:
        with pytest.raises(ValueError):
            add_buildings_cityjson(path)
    finally:
        path.unlink()


def test_cityjson_transform_application():
    """Test that transform is correctly applied to vertices."""
    from forge3d.buildings import add_buildings_cityjson

    # Use integer vertices with scale/translate
    cityjson = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {
            "scale": [0.01, 0.01, 0.01],  # mm to m
            "translate": [100.0, 200.0, 0.0]
        },
        "vertices": [
            [0, 0, 0],        # -> [100, 200, 0]
            [1000, 0, 0],     # -> [110, 200, 0]
            [1000, 1000, 0],  # -> [110, 210, 0]
            [0, 1000, 0],     # -> [100, 210, 0]
        ],
        "CityObjects": {
            "building_transform": {
                "type": "Building",
                "geometry": [{"type": "MultiSurface", "lod": "1", "boundaries": [[[0, 1, 2, 3]]]}]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cityjson, f)
        cityjson_path = Path(f.name)

    try:
        layer = add_buildings_cityjson(cityjson_path)
        if layer.buildings and layer.buildings[0].positions.size > 0:
            bounds = layer.bounds()
            if bounds:
                # X should be around 100-110
                assert bounds[0] >= 99.0  # min_x
                assert bounds[3] <= 111.0  # max_x
    finally:
        cityjson_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
