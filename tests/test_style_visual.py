# tests/test_style_visual.py
"""Visual diff tests for style-driven rendering.

These tests verify that style spec application produces visually
different output compared to default styling.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from forge3d.style import (
    parse_style,
    apply_style,
    layer_to_vector_style,
    VectorStyle,
    evaluate_color_expr,
    evaluate_number_expr,
)
from forge3d.style_expressions import evaluate, EvalContext


# Test style with data-driven expressions
EXPRESSION_STYLE = {
    "version": 8,
    "name": "Expression Test Style",
    "sources": {},
    "layers": [
        {
            "id": "roads-width",
            "type": "line",
            "source": "composite",
            "source-layer": "road",
            "paint": {
                "line-color": ["match", ["get", "class"],
                    "motorway", "#ff0000",
                    "primary", "#ffa500",
                    "secondary", "#ffff00",
                    "#888888"
                ],
                "line-width": ["interpolate", ["linear"], ["zoom"],
                    5, 1,
                    10, 2,
                    15, 4,
                    20, 8
                ]
            }
        },
        {
            "id": "buildings",
            "type": "fill",
            "source": "composite",
            "source-layer": "building",
            "paint": {
                "fill-color": ["step", ["get", "height"],
                    "#e0e0e0",
                    10, "#c0c0c0",
                    50, "#a0a0a0",
                    100, "#808080"
                ],
                "fill-opacity": 0.8
            }
        },
        {
            "id": "labels",
            "type": "symbol",
            "source": "composite",
            "source-layer": "place_label",
            "layout": {
                "text-field": "{name}",
                "text-size": ["interpolate", ["linear"], ["zoom"],
                    8, 10,
                    12, 14,
                    16, 18
                ]
            },
            "paint": {
                "text-color": ["case",
                    [">", ["get", "population"], 1000000], "#000000",
                    [">", ["get", "population"], 100000], "#333333",
                    "#666666"
                ]
            }
        }
    ]
}


class TestExpressionEvaluation:
    """Test expression evaluation functions."""

    def test_interpolate_linear_zoom(self):
        """Test linear zoom interpolation."""
        expr = ["interpolate", ["linear"], ["zoom"], 5, 1, 15, 10]
        
        ctx = EvalContext(properties={}, zoom=5.0)
        assert evaluate(expr, ctx) == 1
        
        ctx = EvalContext(properties={}, zoom=10.0)
        result = evaluate(expr, ctx)
        assert abs(result - 5.5) < 0.1
        
        ctx = EvalContext(properties={}, zoom=15.0)
        assert evaluate(expr, ctx) == 10

    def test_interpolate_exponential(self):
        """Test exponential interpolation."""
        expr = ["interpolate", ["exponential", 2], ["zoom"], 0, 1, 10, 100]
        
        ctx = EvalContext(properties={}, zoom=0.0)
        assert evaluate(expr, ctx) == 1
        
        ctx = EvalContext(properties={}, zoom=10.0)
        assert evaluate(expr, ctx) == 100

    def test_step_function(self):
        """Test step function."""
        expr = ["step", ["get", "height"], "small", 10, "medium", 50, "large"]
        
        ctx = EvalContext(properties={"height": 5}, zoom=10.0)
        assert evaluate(expr, ctx) == "small"
        
        ctx = EvalContext(properties={"height": 25}, zoom=10.0)
        assert evaluate(expr, ctx) == "medium"
        
        ctx = EvalContext(properties={"height": 100}, zoom=10.0)
        assert evaluate(expr, ctx) == "large"

    def test_match_expression(self):
        """Test match expression."""
        expr = ["match", ["get", "type"],
            "highway", "#ff0000",
            "street", "#00ff00",
            "#888888"
        ]
        
        ctx = EvalContext(properties={"type": "highway"}, zoom=10.0)
        assert evaluate(expr, ctx) == "#ff0000"
        
        ctx = EvalContext(properties={"type": "street"}, zoom=10.0)
        assert evaluate(expr, ctx) == "#00ff00"
        
        ctx = EvalContext(properties={"type": "path"}, zoom=10.0)
        assert evaluate(expr, ctx) == "#888888"

    def test_match_with_array_labels(self):
        """Test match with array of labels."""
        expr = ["match", ["get", "class"],
            ["motorway", "trunk"], "major",
            ["primary", "secondary"], "medium",
            "minor"
        ]
        
        ctx = EvalContext(properties={"class": "motorway"}, zoom=10.0)
        assert evaluate(expr, ctx) == "major"
        
        ctx = EvalContext(properties={"class": "trunk"}, zoom=10.0)
        assert evaluate(expr, ctx) == "major"
        
        ctx = EvalContext(properties={"class": "primary"}, zoom=10.0)
        assert evaluate(expr, ctx) == "medium"

    def test_case_expression(self):
        """Test case expression."""
        expr = ["case",
            [">", ["get", "population"], 1000000], "large",
            [">", ["get", "population"], 100000], "medium",
            "small"
        ]
        
        ctx = EvalContext(properties={"population": 5000000}, zoom=10.0)
        assert evaluate(expr, ctx) == "large"
        
        ctx = EvalContext(properties={"population": 500000}, zoom=10.0)
        assert evaluate(expr, ctx) == "medium"
        
        ctx = EvalContext(properties={"population": 10000}, zoom=10.0)
        assert evaluate(expr, ctx) == "small"

    def test_coalesce(self):
        """Test coalesce expression."""
        expr = ["coalesce", ["get", "name_en"], ["get", "name"], "Unknown"]
        
        ctx = EvalContext(properties={"name": "Test"}, zoom=10.0)
        assert evaluate(expr, ctx) == "Test"
        
        ctx = EvalContext(properties={"name_en": "English", "name": "Native"}, zoom=10.0)
        assert evaluate(expr, ctx) == "English"
        
        ctx = EvalContext(properties={}, zoom=10.0)
        assert evaluate(expr, ctx) == "Unknown"

    def test_math_expressions(self):
        """Test math expressions."""
        ctx = EvalContext(properties={}, zoom=10.0)
        
        assert evaluate(["+", 1, 2, 3], ctx) == 6
        assert evaluate(["*", 2, 3, 4], ctx) == 24
        assert evaluate(["-", 10, 3], ctx) == 7
        assert evaluate(["/", 20, 4], ctx) == 5
        assert evaluate(["^", 2, 3], ctx) == 8
        assert evaluate(["sqrt", 16], ctx) == 4

    def test_comparison_expressions(self):
        """Test comparison expressions."""
        ctx = EvalContext(properties={"value": 50}, zoom=10.0)
        
        assert evaluate(["==", ["get", "value"], 50], ctx) is True
        assert evaluate(["!=", ["get", "value"], 100], ctx) is True
        assert evaluate([">", ["get", "value"], 25], ctx) is True
        assert evaluate(["<", ["get", "value"], 75], ctx) is True
        assert evaluate([">=", ["get", "value"], 50], ctx) is True
        assert evaluate(["<=", ["get", "value"], 50], ctx) is True

    def test_logic_expressions(self):
        """Test logic expressions."""
        ctx = EvalContext(properties={"a": 1, "b": 2}, zoom=10.0)
        
        assert evaluate(["all", [">", ["get", "a"], 0], [">", ["get", "b"], 0]], ctx) is True
        assert evaluate(["any", [">", ["get", "a"], 10], [">", ["get", "b"], 0]], ctx) is True
        assert evaluate(["!", [">", ["get", "a"], 10]], ctx) is True

    def test_string_expressions(self):
        """Test string expressions."""
        ctx = EvalContext(properties={"name": "Test"}, zoom=10.0)
        
        assert evaluate(["concat", "Hello ", ["get", "name"]], ctx) == "Hello Test"
        assert evaluate(["upcase", ["get", "name"]], ctx) == "TEST"
        assert evaluate(["downcase", "HELLO"], ctx) == "hello"

    def test_color_rgb_rgba(self):
        """Test rgb/rgba color constructors."""
        ctx = EvalContext(properties={}, zoom=10.0)
        
        result = evaluate(["rgb", 255, 0, 0], ctx)
        assert result is not None
        assert abs(result[0] - 1.0) < 0.01
        assert abs(result[1] - 0.0) < 0.01
        
        result = evaluate(["rgba", 0, 255, 0, 0.5], ctx)
        assert result is not None
        assert abs(result[1] - 1.0) < 0.01
        assert abs(result[3] - 0.5) < 0.01


class TestDataDrivenStyling:
    """Test data-driven styling with expressions."""

    def test_road_color_by_class(self):
        """Road color varies by class property."""
        spec = parse_style(EXPRESSION_STYLE)
        roads = spec.layer_by_id("roads-width")
        
        # Get the raw expression
        color_expr = roads.paint.line_color
        
        # Motorway should be red
        rgba = evaluate_color_expr(color_expr, {"class": "motorway"})
        assert rgba is not None
        assert rgba[0] > 0.9  # Red

        # Primary should be orange
        rgba = evaluate_color_expr(color_expr, {"class": "primary"})
        assert rgba is not None
        assert rgba[0] > 0.9  # Has red
        assert rgba[1] > 0.5  # Has some green (orange)

        # Unknown class should be gray
        rgba = evaluate_color_expr(color_expr, {"class": "unknown"})
        assert rgba is not None
        assert abs(rgba[0] - rgba[1]) < 0.1  # Gray (equal R and G)

    def test_road_width_by_zoom(self):
        """Road width varies by zoom level."""
        spec = parse_style(EXPRESSION_STYLE)
        roads = spec.layer_by_id("roads-width")
        
        width_expr = roads.paint.line_width
        
        # Low zoom should be narrow
        width = evaluate_number_expr(width_expr, {}, zoom=5.0)
        assert width is not None
        assert width == 1.0
        
        # High zoom should be wider
        width = evaluate_number_expr(width_expr, {}, zoom=20.0)
        assert width is not None
        assert width == 8.0

    def test_building_color_by_height(self):
        """Building color varies by height property."""
        spec = parse_style(EXPRESSION_STYLE)
        buildings = spec.layer_by_id("buildings")
        
        color_expr = buildings.paint.fill_color
        
        # Short building should be light gray
        rgba = evaluate_color_expr(color_expr, {"height": 5})
        assert rgba is not None
        # #e0e0e0 = 224/255 ≈ 0.878
        assert rgba[0] > 0.8

        # Tall building should be darker
        rgba = evaluate_color_expr(color_expr, {"height": 150})
        assert rgba is not None
        # #808080 = 128/255 ≈ 0.502
        assert rgba[0] < 0.6


class TestVisualDiff:
    """Test that styled output differs from default."""

    def test_styled_vs_default_produces_different_colors(self):
        """Styled features have different colors than default."""
        spec = parse_style(EXPRESSION_STYLE)
        
        features = [
            {"properties": {"class": "motorway"}, "geometry": {}},
            {"properties": {"class": "primary"}, "geometry": {}},
            {"properties": {"class": "residential"}, "geometry": {}},
        ]
        
        default = VectorStyle()
        
        # Apply styles
        result = apply_style(spec, features, source_layer="road")
        
        # At least one styled feature should differ from default
        different_count = 0
        for _, style in result:
            if style.fill_color != default.fill_color:
                different_count += 1
            if style.stroke_color != default.stroke_color:
                different_count += 1
        
        assert different_count > 0, "Styled output should differ from default"

    def test_zoom_produces_different_widths(self):
        """Different zoom levels produce different widths."""
        spec = parse_style(EXPRESSION_STYLE)
        roads = spec.layer_by_id("roads-width")
        width_expr = roads.paint.line_width
        
        widths_at_zooms = []
        for zoom in [5, 10, 15, 20]:
            width = evaluate_number_expr(width_expr, {}, zoom=float(zoom))
            widths_at_zooms.append(width)
        
        # All widths should be different
        assert len(set(widths_at_zooms)) == 4, "Each zoom should produce different width"

    def test_property_produces_different_colors(self):
        """Different property values produce different colors."""
        spec = parse_style(EXPRESSION_STYLE)
        roads = spec.layer_by_id("roads-width")
        color_expr = roads.paint.line_color
        
        colors_by_class = {}
        for cls in ["motorway", "primary", "secondary", "unknown"]:
            rgba = evaluate_color_expr(color_expr, {"class": cls})
            # Convert to hashable tuple rounded to 2 decimals
            colors_by_class[cls] = tuple(round(c, 2) for c in rgba) if rgba else None
        
        # At least 3 different colors (motorway, primary, secondary are different)
        unique_colors = set(colors_by_class.values())
        assert len(unique_colors) >= 3, "Different classes should produce different colors"


class TestStyleHashDiff:
    """Test using hash comparison for visual diff."""

    def test_expression_style_hash_differs_from_default(self):
        """Hash of expression-styled output differs from default."""
        spec = parse_style(EXPRESSION_STYLE)
        
        # Generate "rendered" output for multiple features
        features = [
            {"properties": {"class": "motorway", "height": 100}},
            {"properties": {"class": "primary", "height": 50}},
            {"properties": {"class": "secondary", "height": 10}},
        ]
        
        # Styled output
        styled_colors = []
        for feature in features:
            roads = spec.layer_by_id("roads-width")
            color = evaluate_color_expr(
                roads.paint.line_color,
                feature["properties"]
            )
            styled_colors.append(color)
        
        styled_hash = hashlib.md5(str(styled_colors).encode()).hexdigest()
        
        # Default output
        default = VectorStyle()
        default_colors = [default.fill_color] * len(features)
        default_hash = hashlib.md5(str(default_colors).encode()).hexdigest()
        
        assert styled_hash != default_hash, "Styled hash should differ from default"

    def test_zoom_change_changes_hash(self):
        """Changing zoom level changes output hash."""
        spec = parse_style(EXPRESSION_STYLE)
        roads = spec.layer_by_id("roads-width")
        
        hashes = []
        for zoom in [5, 10, 15, 20]:
            width = evaluate_number_expr(roads.paint.line_width, {}, zoom=float(zoom))
            h = hashlib.md5(str(width).encode()).hexdigest()
            hashes.append(h)
        
        # All hashes should be different
        assert len(set(hashes)) == 4, "Each zoom should produce different hash"
