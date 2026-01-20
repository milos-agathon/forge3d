# tests/test_style_pixel_diff.py
"""Tests for style integration with pixel diff verification.

Per docs/plan.md acceptance test requirement:
- Assert render with style vs default produces pixel diff > 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge3d.style import (
    parse_style,
    apply_style,
    layer_to_vector_style,
    VectorStyle,
    LabelStyle,
)


# Test fixture: Mapbox-like style spec
WATER_STYLE = {
    "version": 8,
    "name": "Water Test Style",
    "sources": {},
    "layers": [
        {
            "id": "water-fill",
            "type": "fill",
            "source": "test",
            "source-layer": "water",
            "paint": {
                "fill-color": "#0077be",  # Distinct blue
                "fill-opacity": 1.0,
            },
        },
        {
            "id": "land-fill",
            "type": "fill", 
            "source": "test",
            "source-layer": "land",
            "paint": {
                "fill-color": "#228b22",  # Forest green
                "fill-opacity": 0.9,
            },
        },
    ],
}


def rgba_to_color_hash(rgba: tuple[float, float, float, float]) -> int:
    """Convert RGBA tuple to integer hash for comparison."""
    r, g, b, a = rgba
    return int(r * 255) << 24 | int(g * 255) << 16 | int(b * 255) << 8 | int(a * 255)


def test_style_produces_different_colors():
    """Styled output produces different colors than default."""
    spec = parse_style(WATER_STYLE)
    water_layer = spec.layer_by_id("water-fill")
    land_layer = spec.layer_by_id("land-fill")
    
    water_style = layer_to_vector_style(water_layer)
    land_style = layer_to_vector_style(land_layer)
    default_style = VectorStyle()
    
    # Assert different fill colors
    water_hash = rgba_to_color_hash(water_style.fill_color)
    land_hash = rgba_to_color_hash(land_style.fill_color)
    default_hash = rgba_to_color_hash(default_style.fill_color)
    
    assert water_hash != default_hash, "Water style must differ from default"
    assert land_hash != default_hash, "Land style must differ from default"
    assert water_hash != land_hash, "Water and land must have different colors"


def test_apply_style_changes_feature_colors():
    """apply_style() produces different colors per source-layer."""
    spec = parse_style(WATER_STYLE)
    
    water_features = [
        {"type": "Feature", "properties": {}, "geometry": {}},
    ]
    land_features = [
        {"type": "Feature", "properties": {}, "geometry": {}},
    ]
    
    water_result = apply_style(spec, water_features, source_layer="water")
    land_result = apply_style(spec, land_features, source_layer="land")
    
    # Extract fill colors
    assert len(water_result) == 1
    assert len(land_result) == 1
    
    _, water_style = water_result[0]
    _, land_style = land_result[0]
    
    # Colors should differ
    assert water_style.fill_color != land_style.fill_color


def test_style_color_values_in_expected_range():
    """Style colors have values in [0, 1] range."""
    spec = parse_style(WATER_STYLE)
    water_layer = spec.layer_by_id("water-fill")
    style = layer_to_vector_style(water_layer)
    
    for channel in style.fill_color:
        assert 0.0 <= channel <= 1.0, f"Color channel {channel} out of range"
    
    for channel in style.stroke_color:
        assert 0.0 <= channel <= 1.0, f"Stroke channel {channel} out of range"


def test_pixel_diff_styled_vs_default():
    """Simulated pixel diff: styled colors differ from default by measurable amount.
    
    This test simulates what a full render test would verify:
    styled output produces pixels that differ from default output.
    """
    spec = parse_style(WATER_STYLE)
    water_layer = spec.layer_by_id("water-fill")
    styled = layer_to_vector_style(water_layer)
    default = VectorStyle()
    
    # Compute "pixel difference" as Euclidean distance in RGB space
    def rgb_distance(c1: tuple, c2: tuple) -> float:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])))
    
    diff = rgb_distance(styled.fill_color, default.fill_color)
    
    # Water blue (#0077be) vs default gray-blue should have significant distance
    # Threshold: 0.2 in normalized [0,1] RGB space ≈ 20% difference
    assert diff > 0.2, f"Styled vs default diff ({diff:.3f}) must be > 0.2"


def test_multiple_layers_produce_distinct_styles():
    """Multiple layers each produce distinct, non-default styles."""
    spec = parse_style(WATER_STYLE)
    
    styles = []
    default = VectorStyle()
    
    for layer in spec.fill_layers():
        style = layer_to_vector_style(layer)
        styles.append(style)
        assert style.fill_color != default.fill_color, f"Layer {layer.id} should differ from default"
    
    # All layers should have distinct colors
    colors = [s.fill_color for s in styles]
    unique_colors = set(colors)
    assert len(unique_colors) == len(colors), "All layers should have unique colors"


def test_style_integration_workflow():
    """End-to-end workflow: load style → apply to features → get styled colors."""
    # This tests the documented workflow for style integration:
    # 1. Load style spec
    # 2. Call apply_style() to match features to layers
    # 3. Use resulting VectorStyle colors in render_polygons()
    
    style_data = {
        "version": 8,
        "name": "Integration Test",
        "sources": {},
        "layers": [
            {
                "id": "parks",
                "type": "fill",
                "source": "test",
                "source-layer": "landuse",
                "filter": ["==", "class", "park"],
                "paint": {"fill-color": "#90ee90"},  # Light green
            },
            {
                "id": "water",
                "type": "fill",
                "source": "test", 
                "source-layer": "landuse",
                "filter": ["==", "class", "water"],
                "paint": {"fill-color": "#4169e1"},  # Royal blue
            },
        ],
    }
    
    spec = parse_style(style_data)
    
    features = [
        {"type": "Feature", "properties": {"class": "park"}, "geometry": {}},
        {"type": "Feature", "properties": {"class": "water"}, "geometry": {}},
    ]
    
    results = apply_style(spec, features, source_layer="landuse")
    
    assert len(results) == 2
    
    # Extract styled colors
    park_style = results[0][1]
    water_style = results[1][1]
    
    # Park should be greenish (high G channel)
    assert park_style.fill_color[1] > 0.8, "Park should be green"
    
    # Water should be bluish (high B channel)
    assert water_style.fill_color[2] > 0.5, "Water should be blue"


def test_render_pixel_diff_styled_vs_default():
    """REAL render-based pixel diff: styled render differs from default render.
    
    This test performs actual rendering with render_polygons() and verifies
    that styled output produces measurably different pixels than default.
    """
    import json
    import tempfile
    from pathlib import Path
    
    # Check if geopandas is available
    try:
        import geopandas as gpd  # noqa: F401
        HAS_GEOPANDAS = True
    except ImportError:
        HAS_GEOPANDAS = False
    
    if not HAS_GEOPANDAS:
        pytest.skip("Requires geopandas for render test")
    
    from forge3d.render import render_polygons
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create GeoJSON test file
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"class": "water"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]]
                    }
                }
            ]
        }
        
        geojson_path = Path(tmpdir) / "test.geojson"
        with open(geojson_path, "w") as f:
            json.dump(geojson, f)
        
        # Render with DEFAULT style
        img_default = render_polygons(
            geojson_path,
            size=(100, 100),
            fill_rgba=(0.5, 0.5, 0.5, 1.0),  # Gray default
        )
        
        # Create style that produces BLUE fill
        style_data = {
            "version": 8,
            "layers": [
                {
                    "id": "water",
                    "type": "fill",
                    "source": "test",
                    "source-layer": "water",
                    "paint": {"fill-color": "#0077ff"}  # Bright blue
                }
            ]
        }
        style = parse_style(style_data)
        
        # Render with STYLED rendering
        img_styled = render_polygons(
            geojson_path,
            size=(100, 100),
            style=style,
            style_layer="water",
        )
        
        # Compute pixel difference
        diff = np.abs(img_default.astype(float) - img_styled.astype(float))
        total_diff = np.sum(diff)
        
        # There MUST be a measurable difference
        assert total_diff > 0, (
            "Styled render must produce different pixels than default. "
            f"Total pixel diff: {total_diff}"
        )
        
        # Additionally verify it's a significant difference (not just alpha)
        mean_diff = np.mean(diff)
        assert mean_diff > 1.0, (
            f"Mean pixel difference ({mean_diff:.2f}) should be > 1.0 "
            "to confirm style was applied"
        )

