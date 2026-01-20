# tests/test_mapbox_streets_fixture.py
"""Tests for Mapbox Streets v8 fixture parsing and application."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge3d.style import (
    load_style,
    parse_style,
    layer_to_vector_style,
    layer_to_label_style,
    VectorStyle,
    LabelStyle,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mapbox_streets_v8.json"


@pytest.fixture
def mapbox_streets():
    """Load Mapbox Streets v8 fixture."""
    return load_style(FIXTURE_PATH)


def test_mapbox_fixture_exists():
    """Verify Mapbox Streets v8 fixture file exists."""
    assert FIXTURE_PATH.exists(), f"Fixture missing: {FIXTURE_PATH}"


def test_mapbox_fixture_valid_json():
    """Fixture is valid JSON."""
    with open(FIXTURE_PATH) as f:
        data = json.load(f)
    assert "version" in data
    assert data["version"] == 8


def test_mapbox_streets_version(mapbox_streets):
    """Mapbox Streets has version 8."""
    assert mapbox_streets.version == 8


def test_mapbox_streets_has_name(mapbox_streets):
    """Mapbox Streets has a name."""
    assert mapbox_streets.name != ""
    assert "Streets" in mapbox_streets.name or "Mapbox" in mapbox_streets.name


def test_mapbox_streets_has_layers(mapbox_streets):
    """Mapbox Streets has multiple layers."""
    assert len(mapbox_streets.layers) > 5, "Should have multiple layers"


def test_mapbox_streets_has_fill_layers(mapbox_streets):
    """Mapbox Streets has fill layers (water, land, buildings)."""
    fill_layers = mapbox_streets.fill_layers()
    assert len(fill_layers) >= 1, "Should have at least one fill layer"
    
    # Check for common layer types
    layer_ids = [l.id for l in fill_layers]
    assert any("water" in lid for lid in layer_ids), "Should have water layer"


def test_mapbox_streets_has_line_layers(mapbox_streets):
    """Mapbox Streets has line layers (roads)."""
    line_layers = mapbox_streets.line_layers()
    assert len(line_layers) >= 1, "Should have at least one line layer"
    
    # Check for road layers
    layer_ids = [l.id for l in line_layers]
    assert any("road" in lid for lid in layer_ids), "Should have road layer"


def test_mapbox_streets_has_symbol_layers(mapbox_streets):
    """Mapbox Streets has symbol layers (labels)."""
    symbol_layers = mapbox_streets.symbol_layers()
    assert len(symbol_layers) >= 1, "Should have at least one symbol layer"


def test_mapbox_streets_water_style(mapbox_streets):
    """Water layer produces blue-ish style."""
    water_layer = mapbox_streets.layer_by_id("water")
    assert water_layer is not None, "Should have 'water' layer"
    
    style = layer_to_vector_style(water_layer)
    
    # Water should be blue (B channel > R and G)
    r, g, b, a = style.fill_color
    assert b > r, "Water should have more blue than red"
    assert b > 0.4, "Water should have significant blue component"


def test_mapbox_streets_road_style(mapbox_streets):
    """Road layers produce visible line styles."""
    # Find a road layer
    road_layer = None
    for layer in mapbox_streets.line_layers():
        if "road" in layer.id:
            road_layer = layer
            break
    
    assert road_layer is not None, "Should have a road layer"
    style = layer_to_vector_style(road_layer)
    
    # Road should have positive stroke width
    assert style.stroke_width > 0, "Road should have visible stroke width"


def test_mapbox_streets_label_style(mapbox_streets):
    """Symbol layers produce label styles."""
    # Find a label layer
    label_layer = None
    for layer in mapbox_streets.symbol_layers():
        if "label" in layer.id.lower() or "place" in layer.id.lower():
            label_layer = layer
            break
    
    if label_layer is None and mapbox_streets.symbol_layers():
        label_layer = mapbox_streets.symbol_layers()[0]
    
    assert label_layer is not None, "Should have a symbol layer"
    style = layer_to_label_style(label_layer)
    
    # Label should have positive size
    assert style.size > 0, "Label should have positive size"


def test_mapbox_streets_has_source(mapbox_streets):
    """Mapbox Streets has sources defined."""
    assert len(mapbox_streets.sources) >= 1, "Should have at least one source"


def test_mapbox_streets_has_sprite(mapbox_streets):
    """Mapbox Streets has sprite URL."""
    assert mapbox_streets.sprite is not None, "Should have sprite URL"
    assert "sprite" in mapbox_streets.sprite or "mapbox" in mapbox_streets.sprite


def test_mapbox_streets_has_glyphs(mapbox_streets):
    """Mapbox Streets has glyphs URL."""
    assert mapbox_streets.glyphs is not None, "Should have glyphs URL"
    assert "font" in mapbox_streets.glyphs or "glyph" in mapbox_streets.glyphs


def test_mapbox_streets_all_layers_valid(mapbox_streets):
    """All layers can be parsed without errors."""
    for layer in mapbox_streets.layers:
        assert layer.id != "", f"Layer should have ID"
        assert layer.layer_type in ("background", "fill", "line", "symbol", "circle", "raster", "fill-extrusion"), \
            f"Unknown layer type: {layer.layer_type}"
