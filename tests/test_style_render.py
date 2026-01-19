# tests/test_style_render.py
"""Tests for style-driven rendering integration."""

from __future__ import annotations

import pytest

from forge3d.style import (
    parse_style,
    layer_to_vector_style,
    layer_to_label_style,
    apply_style,
    VectorStyle,
)


STREETS_STYLE = {
    "version": 8,
    "name": "Streets Style",
    "sources": {},
    "layers": [
        {
            "id": "background",
            "type": "background",
            "paint": {"background-color": "#f8f4f0"},
        },
        {
            "id": "water",
            "type": "fill",
            "source": "composite",
            "source-layer": "water",
            "paint": {"fill-color": "#a0cfdf", "fill-opacity": 1.0},
        },
        {
            "id": "landuse-park",
            "type": "fill",
            "source": "composite",
            "source-layer": "landuse",
            "filter": ["==", "class", "park"],
            "paint": {"fill-color": "#c8df9f", "fill-opacity": 0.8},
        },
        {
            "id": "landuse-industrial",
            "type": "fill",
            "source": "composite",
            "source-layer": "landuse",
            "filter": ["==", "class", "industrial"],
            "paint": {"fill-color": "#d4d0c8"},
        },
        {
            "id": "road-primary",
            "type": "line",
            "source": "composite",
            "source-layer": "road",
            "filter": ["==", "class", "primary"],
            "paint": {"line-color": "#ffc107", "line-width": 4},
        },
        {
            "id": "road-secondary",
            "type": "line",
            "source": "composite",
            "source-layer": "road",
            "filter": ["==", "class", "secondary"],
            "paint": {"line-color": "#ffffff", "line-width": 2},
        },
        {
            "id": "place-labels",
            "type": "symbol",
            "source": "composite",
            "source-layer": "place_label",
            "layout": {"text-field": "{name}", "text-size": 12},
            "paint": {"text-color": "#333333", "text-halo-color": "#ffffff", "text-halo-width": 1.0},
        },
    ],
}


def test_style_has_multiple_layers():
    """Style spec has at least 5 layers as required."""
    spec = parse_style(STREETS_STYLE)
    assert len(spec.layers) >= 5


def test_water_layer_produces_non_default_style():
    """Water layer produces non-default fill color."""
    spec = parse_style(STREETS_STYLE)
    water = spec.layer_by_id("water")
    style = layer_to_vector_style(water)
    
    default = VectorStyle()
    assert style.fill_color != default.fill_color
    assert style.fill_color[2] > 0.5


def test_road_layer_produces_non_default_style():
    """Road layers produce non-default line styles."""
    spec = parse_style(STREETS_STYLE)
    
    primary = spec.layer_by_id("road-primary")
    style = layer_to_vector_style(primary)
    assert style.stroke_width == 4.0
    
    secondary = spec.layer_by_id("road-secondary")
    style2 = layer_to_vector_style(secondary)
    assert style2.stroke_width == 2.0


def test_label_layer_produces_non_default_style():
    """Symbol layers produce non-default label styles."""
    spec = parse_style(STREETS_STYLE)
    places = spec.layer_by_id("place-labels")
    style = layer_to_label_style(places)
    
    assert style.size == 12.0
    assert style.halo_width == 1.0


def test_apply_style_water_features():
    """Apply style to water features."""
    spec = parse_style(STREETS_STYLE)
    
    features = [
        {"type": "Feature", "properties": {"name": "Lake"}, "geometry": {}},
        {"type": "Feature", "properties": {"name": "River"}, "geometry": {}},
    ]
    
    result = apply_style(spec, features, source_layer="water")
    assert len(result) == 2
    
    for _, style in result:
        assert style.fill_color[2] > 0.5


def test_apply_style_landuse_filtering():
    """Apply style with filter matching."""
    spec = parse_style(STREETS_STYLE)
    
    features = [
        {"type": "Feature", "properties": {"class": "park"}, "geometry": {}},
        {"type": "Feature", "properties": {"class": "industrial"}, "geometry": {}},
    ]
    
    result = apply_style(spec, features, source_layer="landuse")
    assert len(result) == 2
    
    _, park_style = result[0]
    assert park_style.fill_color[1] > 0.5


def test_apply_style_road_filtering():
    """Apply style to road features with class filters."""
    spec = parse_style(STREETS_STYLE)
    
    features = [
        {"type": "Feature", "properties": {"class": "primary"}, "geometry": {}},
        {"type": "Feature", "properties": {"class": "secondary"}, "geometry": {}},
    ]
    
    result = apply_style(spec, features, source_layer="road")
    assert len(result) == 2
    
    _, primary_style = result[0]
    assert primary_style.stroke_width == 4.0
    
    _, secondary_style = result[1]
    assert secondary_style.stroke_width == 2.0


def test_style_vs_default_produces_different_output():
    """Styled output differs from default style."""
    spec = parse_style(STREETS_STYLE)
    
    features = [
        {"type": "Feature", "properties": {"class": "primary"}, "geometry": {}},
    ]
    
    result = apply_style(spec, features, source_layer="road")
    _, styled = result[0]
    default = VectorStyle()
    
    assert styled.stroke_width != default.stroke_width
    assert styled.stroke_color != default.stroke_color
