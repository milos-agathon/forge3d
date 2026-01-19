# tests/test_style_parser.py
"""Tests for Mapbox Style Spec parsing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forge3d.style import (
    load_style,
    parse_style,
    parse_color,
    paint_to_vector_style,
    layout_to_label_style,
    apply_style,
    StyleSpec,
    StyleLayer,
    PaintProps,
    LayoutProps,
)


MINIMAL_STYLE = {
    "version": 8,
    "name": "Test Style",
    "sources": {},
    "layers": [
        {
            "id": "background",
            "type": "background",
            "paint": {"background-color": "#f0f0f0"},
        },
        {
            "id": "water",
            "type": "fill",
            "source": "composite",
            "source-layer": "water",
            "paint": {"fill-color": "#0066ff", "fill-opacity": 0.8},
        },
        {
            "id": "roads",
            "type": "line",
            "source": "composite",
            "source-layer": "road",
            "paint": {"line-color": "#ffffff", "line-width": 2},
            "filter": ["==", "class", "motorway"],
        },
        {
            "id": "labels",
            "type": "symbol",
            "source": "composite",
            "source-layer": "place_label",
            "layout": {"text-field": "{name}", "text-size": 14},
            "paint": {
                "text-color": "#333333",
                "text-halo-color": "#ffffff",
                "text-halo-width": 1.5,
            },
        },
        {
            "id": "hidden-layer",
            "type": "fill",
            "source": "composite",
            "source-layer": "landuse",
            "layout": {"visibility": "none"},
        },
    ],
}


def test_parse_minimal_style():
    """Parse a minimal style spec."""
    spec = parse_style(MINIMAL_STYLE)
    assert spec.version == 8
    assert spec.name == "Test Style"
    assert len(spec.layers) == 5


def test_parse_style_from_file():
    """Load style from JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINIMAL_STYLE, f)
        f.flush()
        spec = load_style(Path(f.name))
        assert len(spec.layers) >= 5


def test_layers_by_type():
    """Get layers filtered by type."""
    spec = parse_style(MINIMAL_STYLE)
    assert len(spec.fill_layers()) == 2
    assert len(spec.line_layers()) == 1
    assert len(spec.symbol_layers()) == 1


def test_layer_by_id():
    """Find layer by ID."""
    spec = parse_style(MINIMAL_STYLE)
    water = spec.layer_by_id("water")
    assert water is not None
    assert water.layer_type == "fill"
    
    missing = spec.layer_by_id("nonexistent")
    assert missing is None


def test_layer_visibility():
    """Check layer visibility flag."""
    spec = parse_style(MINIMAL_STYLE)
    water = spec.layer_by_id("water")
    hidden = spec.layer_by_id("hidden-layer")
    
    assert water.is_visible()
    assert not hidden.is_visible()


def test_fill_paint_props():
    """Extract fill paint properties."""
    spec = parse_style(MINIMAL_STYLE)
    water = spec.layer_by_id("water")
    
    assert water.paint.fill_color == "#0066ff"
    assert water.paint.fill_opacity == 0.8


def test_line_paint_props():
    """Extract line paint properties."""
    spec = parse_style(MINIMAL_STYLE)
    roads = spec.layer_by_id("roads")
    
    assert roads.paint.line_color == "#ffffff"
    assert roads.paint.line_width == 2.0


def test_symbol_layout_props():
    """Extract symbol layout properties."""
    spec = parse_style(MINIMAL_STYLE)
    labels = spec.layer_by_id("labels")
    
    assert labels.layout.text_field == "{name}"
    assert labels.layout.text_size == 14.0


def test_filter_extraction():
    """Extract filter expression."""
    spec = parse_style(MINIMAL_STYLE)
    roads = spec.layer_by_id("roads")
    
    assert roads.filter is not None
    assert roads.filter == ["==", "class", "motorway"]


def test_invalid_version():
    """Reject unsupported style version."""
    invalid = {"version": 7, "layers": []}
    with pytest.raises(ValueError, match="Unsupported style version"):
        parse_style(invalid)


def test_parse_hex_colors():
    """Parse hex color strings."""
    # 3-digit hex
    assert parse_color("#fff") == (1.0, 1.0, 1.0, 1.0)
    assert parse_color("#000") == (0.0, 0.0, 0.0, 1.0)
    
    # 6-digit hex
    rgba = parse_color("#ff0000")
    assert abs(rgba[0] - 1.0) < 0.01
    assert abs(rgba[1] - 0.0) < 0.01
    assert abs(rgba[2] - 0.0) < 0.01
    
    # 8-digit hex with alpha
    rgba = parse_color("#00ff0080")
    assert abs(rgba[1] - 1.0) < 0.01
    assert abs(rgba[3] - 0.502) < 0.01


def test_parse_rgb_colors():
    """Parse rgb() color strings."""
    rgba = parse_color("rgb(255, 0, 0)")
    assert abs(rgba[0] - 1.0) < 0.01
    assert abs(rgba[1] - 0.0) < 0.01
    
    rgba = parse_color("rgba(0, 255, 0, 0.5)")
    assert abs(rgba[1] - 1.0) < 0.01
    assert abs(rgba[3] - 0.5) < 0.01


def test_parse_hsl_colors():
    """Parse hsl() color strings."""
    # Red at HSL(0, 100%, 50%)
    rgba = parse_color("hsl(0, 100%, 50%)")
    assert abs(rgba[0] - 1.0) < 0.01
    assert abs(rgba[1] - 0.0) < 0.01
    assert abs(rgba[2] - 0.0) < 0.01


def test_parse_named_colors():
    """Parse named color strings."""
    assert parse_color("black") == (0.0, 0.0, 0.0, 1.0)
    assert parse_color("white") == (1.0, 1.0, 1.0, 1.0)
    assert parse_color("red") == (1.0, 0.0, 0.0, 1.0)
    assert parse_color("transparent") == (0.0, 0.0, 0.0, 0.0)


def test_paint_to_vector_style_fill():
    """Convert fill paint to VectorStyle."""
    paint = PaintProps(fill_color="#ff0000", fill_opacity=0.5)
    style = paint_to_vector_style(paint)
    
    assert abs(style.fill_color[0] - 1.0) < 0.01
    assert abs(style.fill_color[3] - 0.5) < 0.01


def test_paint_to_vector_style_line():
    """Convert line paint to VectorStyle."""
    paint = PaintProps(line_color="#00ff00", line_width=3.0)
    style = paint_to_vector_style(paint)
    
    assert abs(style.stroke_color[1] - 1.0) < 0.01
    assert abs(style.stroke_width - 3.0) < 0.01


def test_layout_to_label_style():
    """Convert layout/paint to LabelStyle."""
    layout = LayoutProps(text_size=16.0)
    paint = PaintProps(
        text_color="#333333",
        text_halo_color="#ffffff",
        text_halo_width=2.0,
    )
    style = layout_to_label_style(layout, paint)
    
    assert abs(style.size - 16.0) < 0.01
    assert abs(style.halo_width - 2.0) < 0.01
    assert abs(style.halo_color[0] - 1.0) < 0.01


def test_filter_evaluation_equality():
    """Evaluate equality filter."""
    spec = parse_style(MINIMAL_STYLE)
    roads = spec.layer_by_id("roads")
    
    # Should match motorway
    props_match = {"class": "motorway", "name": "Highway 1"}
    assert roads.matches_filter(props_match)
    
    # Should not match other classes
    props_nomatch = {"class": "residential", "name": "Main St"}
    assert not roads.matches_filter(props_nomatch)


def test_filter_evaluation_all():
    """Evaluate 'all' filter combinator."""
    layer = StyleLayer(
        id="test",
        layer_type="fill",
        filter=["all", ["==", "class", "road"], ["==", "level", 1]],
    )
    
    assert layer.matches_filter({"class": "road", "level": 1})
    assert not layer.matches_filter({"class": "road", "level": 2})
    assert not layer.matches_filter({"class": "path", "level": 1})


def test_filter_evaluation_any():
    """Evaluate 'any' filter combinator."""
    layer = StyleLayer(
        id="test",
        layer_type="fill",
        filter=["any", ["==", "class", "highway"], ["==", "class", "motorway"]],
    )
    
    assert layer.matches_filter({"class": "highway"})
    assert layer.matches_filter({"class": "motorway"})
    assert not layer.matches_filter({"class": "residential"})


def test_filter_evaluation_has():
    """Evaluate 'has' filter."""
    layer = StyleLayer(
        id="test",
        layer_type="symbol",
        filter=["has", "name"],
    )
    
    assert layer.matches_filter({"name": "Test"})
    assert not layer.matches_filter({"class": "road"})


def test_apply_style_to_features():
    """Apply style spec to GeoJSON features."""
    spec = parse_style(MINIMAL_STYLE)
    
    features = [
        {"type": "Feature", "properties": {"class": "motorway"}, "geometry": {}},
        {"type": "Feature", "properties": {"class": "residential"}, "geometry": {}},
    ]
    
    result = apply_style(spec, features, source_layer="road")
    assert len(result) == 2
    
    # First feature should match roads layer (white)
    _, style1 = result[0]
    assert abs(style1.stroke_color[0] - 1.0) < 0.01  # white


def test_layers_for_source_layer():
    """Get layers for specific source-layer."""
    spec = parse_style(MINIMAL_STYLE)
    
    water_layers = spec.layers_for_source_layer("water")
    assert len(water_layers) == 1
    assert water_layers[0].id == "water"
    
    road_layers = spec.layers_for_source_layer("road")
    assert len(road_layers) == 1
    assert road_layers[0].id == "roads"


def test_zoom_range_filter():
    """Filter layers by zoom range."""
    layer = StyleLayer(
        id="test",
        layer_type="fill",
        minzoom=5.0,
        maxzoom=15.0,
    )
    
    assert layer.in_zoom_range(10.0)
    assert not layer.in_zoom_range(3.0)
    assert not layer.in_zoom_range(20.0)


def test_style_spec_layer_count():
    """Verify layer count matches expected."""
    spec = parse_style(MINIMAL_STYLE)
    assert len(spec.layers) >= 5  # At least 5 layers
