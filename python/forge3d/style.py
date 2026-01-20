# python/forge3d/style.py
"""Mapbox Style Spec import for vector/label styling.

This module provides parsing and conversion of Mapbox GL Style Spec JSON
files into forge3d's native vector and label styles.

Supported layer types (v1):
- fill: Polygon fill with color, opacity, outline
- line: Polyline with color, width, opacity  
- symbol: Text labels with size, color, halo
- background: Background color (informational only)

Example:
    >>> from forge3d.style import load_style, apply_style
    >>> spec = load_style("mapbox-streets.json")
    >>> print(f"Loaded {len(spec.layers)} layers")
    >>> # Apply style to vectors
    >>> apply_style(spec, vectors, source_layer="water")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class VectorStyle:
    """Vector feature style (fill/stroke colors and widths)."""
    fill_color: tuple[float, float, float, float] = (0.2, 0.4, 0.8, 1.0)
    stroke_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    stroke_width: float = 1.0
    point_size: float = 4.0


@dataclass
class LabelStyle:
    """Label text style."""
    size: float = 14.0
    color: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
    halo_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8)
    halo_width: float = 1.5
    offset: tuple[float, float] = (0.0, 0.0)


@dataclass
class PaintProps:
    """Paint properties from style spec (raw values including expressions)."""
    fill_color: Any = None
    fill_opacity: Any = None
    fill_outline_color: Any = None
    line_color: Any = None
    line_width: Any = None
    line_opacity: Any = None
    text_color: Any = None
    text_halo_color: Any = None
    text_halo_width: Any = None
    text_opacity: Any = None
    circle_color: Any = None
    circle_radius: Any = None
    circle_opacity: Any = None
    background_color: Any = None


@dataclass
class LayoutProps:
    """Layout properties from style spec (raw values including expressions)."""
    visibility: Optional[str] = None
    text_field: Any = None
    text_size: Any = None
    text_font: Optional[list[str]] = None
    text_anchor: Optional[str] = None
    text_offset: Any = None
    text_max_width: Any = None
    line_cap: Optional[str] = None
    line_join: Optional[str] = None


@dataclass
class StyleLayer:
    """A single style layer."""
    id: str
    layer_type: str
    source: Optional[str] = None
    source_layer: Optional[str] = None
    paint: PaintProps = field(default_factory=PaintProps)
    layout: LayoutProps = field(default_factory=LayoutProps)
    filter: Optional[list] = None
    minzoom: Optional[float] = None
    maxzoom: Optional[float] = None

    def is_visible(self) -> bool:
        """Check if layer is visible."""
        if self.layout.visibility is None:
            return True
        return self.layout.visibility != "none"

    def in_zoom_range(self, zoom: float) -> bool:
        """Check if layer is visible at given zoom level."""
        if self.minzoom is not None and zoom < self.minzoom:
            return False
        if self.maxzoom is not None and zoom > self.maxzoom:
            return False
        return True

    def matches_filter(self, properties: dict[str, Any]) -> bool:
        """Check if feature properties match the layer filter."""
        if self.filter is None:
            return True
        return _evaluate_filter(self.filter, properties)


@dataclass
class StyleSpec:
    """Complete Mapbox GL Style specification."""
    version: int = 8
    name: str = ""
    layers: list[StyleLayer] = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    sprite: Optional[str] = None
    glyphs: Optional[str] = None

    def fill_layers(self) -> list[StyleLayer]:
        """Get all fill layers."""
        return [l for l in self.layers if l.layer_type == "fill"]

    def line_layers(self) -> list[StyleLayer]:
        """Get all line layers."""
        return [l for l in self.layers if l.layer_type == "line"]

    def symbol_layers(self) -> list[StyleLayer]:
        """Get all symbol layers."""
        return [l for l in self.layers if l.layer_type == "symbol"]

    def layer_by_id(self, layer_id: str) -> Optional[StyleLayer]:
        """Find a layer by ID."""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def layers_for_source_layer(self, source_layer: str) -> list[StyleLayer]:
        """Get all layers for a specific source-layer."""
        return [l for l in self.layers if l.source_layer == source_layer]


def load_style(path: Path | str) -> StyleSpec:
    """Load a Mapbox GL Style Spec JSON file.

    Args:
        path: Path to style.json file.

    Returns:
        Parsed StyleSpec object.

    Raises:
        ValueError: If style version is not 8.
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If JSON is invalid.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return parse_style(data)


def parse_style(data: dict) -> StyleSpec:
    """Parse a Mapbox GL Style Spec from a dictionary.

    Args:
        data: Parsed JSON dictionary.

    Returns:
        Parsed StyleSpec object.

    Raises:
        ValueError: If style version is not 8.
    """
    version = data.get("version", 8)
    if version != 8:
        raise ValueError(f"Unsupported style version: {version} (expected 8)")

    layers = []
    for layer_data in data.get("layers", []):
        layer = _parse_layer(layer_data)
        layers.append(layer)

    return StyleSpec(
        version=version,
        name=data.get("name", ""),
        layers=layers,
        sources=data.get("sources", {}),
        sprite=data.get("sprite"),
        glyphs=data.get("glyphs"),
    )


def _parse_layer(data: dict) -> StyleLayer:
    """Parse a single layer from JSON, preserving raw expressions."""
    paint_data = data.get("paint", {})
    layout_data = data.get("layout", {})

    # Store raw values to preserve expressions for later evaluation
    paint = PaintProps(
        fill_color=paint_data.get("fill-color"),
        fill_opacity=paint_data.get("fill-opacity"),
        fill_outline_color=paint_data.get("fill-outline-color"),
        line_color=paint_data.get("line-color"),
        line_width=paint_data.get("line-width"),
        line_opacity=paint_data.get("line-opacity"),
        text_color=paint_data.get("text-color"),
        text_halo_color=paint_data.get("text-halo-color"),
        text_halo_width=paint_data.get("text-halo-width"),
        text_opacity=paint_data.get("text-opacity"),
        circle_color=paint_data.get("circle-color"),
        circle_radius=paint_data.get("circle-radius"),
        circle_opacity=paint_data.get("circle-opacity"),
        background_color=paint_data.get("background-color"),
    )

    layout = LayoutProps(
        visibility=layout_data.get("visibility"),
        text_field=layout_data.get("text-field"),
        text_size=layout_data.get("text-size"),
        text_font=layout_data.get("text-font"),
        text_anchor=layout_data.get("text-anchor"),
        text_offset=layout_data.get("text-offset"),
        text_max_width=layout_data.get("text-max-width"),
        line_cap=layout_data.get("line-cap"),
        line_join=layout_data.get("line-join"),
    )

    return StyleLayer(
        id=data.get("id", ""),
        layer_type=data.get("type", "unknown"),
        source=data.get("source"),
        source_layer=data.get("source-layer"),
        paint=paint,
        layout=layout,
        filter=data.get("filter"),
        minzoom=data.get("minzoom"),
        maxzoom=data.get("maxzoom"),
    )


def _get_color_value(value: Any) -> Optional[str]:
    """Extract color string from value (skip expressions)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # Skip expressions for now
    return None


def _get_number_value(value: Any) -> Optional[float]:
    """Extract number from value (skip expressions)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Expressions handled by evaluate_number_expr
    return None


def evaluate_color_expr(
    value: Any,
    properties: dict[str, Any],
    zoom: float = 10.0
) -> Optional[tuple[float, float, float, float]]:
    """Evaluate a color expression with feature context.
    
    Args:
        value: Color string or expression.
        properties: Feature properties for data-driven styling.
        zoom: Current zoom level.
    
    Returns:
        RGBA tuple or None if evaluation fails.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return parse_color(value)
    if isinstance(value, list):
        from forge3d.style_expressions import evaluate_color, EvalContext
        ctx = EvalContext(properties=properties, zoom=zoom)
        return evaluate_color(value, ctx)
    return None


def evaluate_number_expr(
    value: Any,
    properties: dict[str, Any],
    zoom: float = 10.0
) -> Optional[float]:
    """Evaluate a number expression with feature context.
    
    Args:
        value: Number literal or expression.
        properties: Feature properties for data-driven styling.
        zoom: Current zoom level.
    
    Returns:
        Float value or None if evaluation fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        from forge3d.style_expressions import evaluate_number, EvalContext
        ctx = EvalContext(properties=properties, zoom=zoom)
        return evaluate_number(value, ctx)
    return None


def _get_text_field(value: Any) -> Optional[str]:
    """Extract text field property name."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 2:
        if value[0] == "get":
            return f"{{{value[1]}}}"
    return None


def paint_to_vector_style(
    paint: PaintProps,
    properties: Optional[dict[str, Any]] = None,
    zoom: float = 10.0
) -> VectorStyle:
    """Convert paint properties to VectorStyle.

    Args:
        paint: Paint properties from style layer.
        properties: Feature properties for data-driven styling.
        zoom: Current zoom level.

    Returns:
        VectorStyle with converted colors and widths.
    """
    props = properties or {}
    style = VectorStyle()

    if paint.fill_color is not None:
        rgba = evaluate_color_expr(paint.fill_color, props, zoom)
        if rgba:
            style = VectorStyle(
                fill_color=rgba,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
                point_size=style.point_size,
            )

    if paint.fill_opacity is not None:
        opacity = evaluate_number_expr(paint.fill_opacity, props, zoom)
        if opacity is not None:
            r, g, b, a = style.fill_color
            style = VectorStyle(
                fill_color=(r, g, b, a * opacity),
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
                point_size=style.point_size,
            )

    if paint.fill_outline_color is not None:
        rgba = evaluate_color_expr(paint.fill_outline_color, props, zoom)
        if rgba:
            style = VectorStyle(
                fill_color=style.fill_color,
                stroke_color=rgba,
                stroke_width=style.stroke_width,
                point_size=style.point_size,
            )

    if paint.line_color is not None:
        rgba = evaluate_color_expr(paint.line_color, props, zoom)
        if rgba:
            style = VectorStyle(
                fill_color=rgba,
                stroke_color=rgba,
                stroke_width=style.stroke_width,
                point_size=style.point_size,
            )

    if paint.line_opacity is not None:
        opacity = evaluate_number_expr(paint.line_opacity, props, zoom)
        if opacity is not None:
            r, g, b, a = style.stroke_color
            style = VectorStyle(
                fill_color=style.fill_color,
                stroke_color=(r, g, b, a * opacity),
                stroke_width=style.stroke_width,
                point_size=style.point_size,
            )

    if paint.line_width is not None:
        width = evaluate_number_expr(paint.line_width, props, zoom)
        if width is not None:
            style = VectorStyle(
                fill_color=style.fill_color,
                stroke_color=style.stroke_color,
                stroke_width=width,
                point_size=style.point_size,
            )

    if paint.circle_color is not None:
        rgba = evaluate_color_expr(paint.circle_color, props, zoom)
        if rgba:
            style = VectorStyle(
                fill_color=rgba,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
                point_size=style.point_size,
            )

    if paint.circle_radius is not None:
        radius = evaluate_number_expr(paint.circle_radius, props, zoom)
        if radius is not None:
            style = VectorStyle(
                fill_color=style.fill_color,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
                point_size=radius * 2.0,
            )

    return style


def layout_to_label_style(
    layout: LayoutProps,
    paint: PaintProps,
    properties: Optional[dict[str, Any]] = None,
    zoom: float = 10.0
) -> LabelStyle:
    """Convert layout and paint properties to LabelStyle.

    Args:
        layout: Layout properties from style layer.
        paint: Paint properties from style layer.
        properties: Feature properties for data-driven styling.
        zoom: Current zoom level.

    Returns:
        LabelStyle with converted text styling.
    """
    props = properties or {}
    style = LabelStyle()

    if layout.text_size is not None:
        size = evaluate_number_expr(layout.text_size, props, zoom)
        if size is not None:
            style = LabelStyle(
                size=size,
                color=style.color,
                halo_color=style.halo_color,
                halo_width=style.halo_width,
                offset=style.offset,
            )

    if paint.text_color is not None:
        rgba = evaluate_color_expr(paint.text_color, props, zoom)
        if rgba:
            style = LabelStyle(
                size=style.size,
                color=rgba,
                halo_color=style.halo_color,
                halo_width=style.halo_width,
                offset=style.offset,
            )

    if paint.text_opacity is not None:
        opacity = evaluate_number_expr(paint.text_opacity, props, zoom)
        if opacity is not None:
            r, g, b, a = style.color
            style = LabelStyle(
                size=style.size,
                color=(r, g, b, a * opacity),
                halo_color=style.halo_color,
                halo_width=style.halo_width,
                offset=style.offset,
            )

    if paint.text_halo_color is not None:
        rgba = evaluate_color_expr(paint.text_halo_color, props, zoom)
        if rgba:
            style = LabelStyle(
                size=style.size,
                color=style.color,
                halo_color=rgba,
                halo_width=style.halo_width,
                offset=style.offset,
            )

    if paint.text_halo_width is not None:
        width = evaluate_number_expr(paint.text_halo_width, props, zoom)
        if width is not None:
            style = LabelStyle(
                size=style.size,
                color=style.color,
                halo_color=style.halo_color,
                halo_width=width,
                offset=style.offset,
            )

    if layout.text_offset is not None and isinstance(layout.text_offset, list) and len(layout.text_offset) >= 2:
        em_to_px = style.size
        style = LabelStyle(
            size=style.size,
            color=style.color,
            halo_color=style.halo_color,
            halo_width=style.halo_width,
            offset=(layout.text_offset[0] * em_to_px, layout.text_offset[1] * em_to_px),
        )

    return style


def layer_to_vector_style(layer: StyleLayer) -> VectorStyle:
    """Convert a style layer to VectorStyle."""
    return paint_to_vector_style(layer.paint)


def layer_to_label_style(layer: StyleLayer) -> LabelStyle:
    """Convert a style layer to LabelStyle."""
    return layout_to_label_style(layer.layout, layer.paint)


def apply_style(
    spec: StyleSpec,
    features: list[dict],
    source_layer: Optional[str] = None,
    zoom: float = 10.0,
) -> list[tuple[dict, VectorStyle]]:
    """Apply style spec to a list of GeoJSON features.

    Args:
        spec: Parsed style specification.
        features: List of GeoJSON feature dictionaries.
        source_layer: Filter to layers with this source-layer.
        zoom: Current zoom level for zoom-dependent styles.

    Returns:
        List of (feature, VectorStyle) tuples for features that match.
    """
    result = []

    # Get applicable layers
    if source_layer:
        layers = spec.layers_for_source_layer(source_layer)
    else:
        layers = spec.layers

    # Filter to visible fill/line layers at current zoom
    layers = [
        l for l in layers
        if l.is_visible()
        and l.in_zoom_range(zoom)
        and l.layer_type in ("fill", "line", "circle")
    ]

    for feature in features:
        props = feature.get("properties", {})
        
        # Find first matching layer
        for layer in layers:
            if layer.matches_filter(props):
                style = layer_to_vector_style(layer)
                result.append((feature, style))
                break
        else:
            # No matching layer, use default style
            result.append((feature, VectorStyle()))

    return result


def parse_color(color_str: str) -> Optional[tuple[float, float, float, float]]:
    """Parse a CSS color string to RGBA tuple (0-1 range).

    Supports:
    - Hex: #RGB, #RGBA, #RRGGBB, #RRGGBBAA
    - RGB: rgb(r, g, b), rgba(r, g, b, a)
    - HSL: hsl(h, s%, l%), hsla(h, s%, l%, a)
    - Named colors: black, white, red, green, blue, etc.

    Args:
        color_str: CSS color string.

    Returns:
        RGBA tuple (0-1 range) or None if parsing fails.
    """
    s = color_str.strip()

    # Hex colors
    if s.startswith("#"):
        return _parse_hex_color(s)

    # RGB/RGBA
    if s.startswith("rgb"):
        return _parse_rgb_color(s)

    # HSL/HSLA
    if s.startswith("hsl"):
        return _parse_hsl_color(s)

    # Named colors
    named = {
        "black": (0.0, 0.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 0.5, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5, 1.0),
        "grey": (0.5, 0.5, 0.5, 1.0),
        "orange": (1.0, 0.647, 0.0, 1.0),
        "transparent": (0.0, 0.0, 0.0, 0.0),
    }
    return named.get(s.lower())


def _parse_hex_color(s: str) -> Optional[tuple[float, float, float, float]]:
    """Parse hex color."""
    hex_str = s.lstrip("#")
    try:
        if len(hex_str) == 3:
            r = int(hex_str[0] * 2, 16) / 255.0
            g = int(hex_str[1] * 2, 16) / 255.0
            b = int(hex_str[2] * 2, 16) / 255.0
            return (r, g, b, 1.0)
        elif len(hex_str) == 4:
            r = int(hex_str[0] * 2, 16) / 255.0
            g = int(hex_str[1] * 2, 16) / 255.0
            b = int(hex_str[2] * 2, 16) / 255.0
            a = int(hex_str[3] * 2, 16) / 255.0
            return (r, g, b, a)
        elif len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return (r, g, b, 1.0)
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
            return (r, g, b, a)
    except ValueError:
        pass
    return None


def _parse_rgb_color(s: str) -> Optional[tuple[float, float, float, float]]:
    """Parse rgb/rgba color."""
    match = re.match(r"rgba?\s*\(\s*([^)]+)\s*\)", s)
    if not match:
        return None

    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) < 3:
        return None

    try:
        # Check for percentage values
        if "%" in parts[0]:
            r = float(parts[0].rstrip("%")) / 100.0
            g = float(parts[1].rstrip("%")) / 100.0
            b = float(parts[2].rstrip("%")) / 100.0
        else:
            r = float(parts[0]) / 255.0
            g = float(parts[1]) / 255.0
            b = float(parts[2]) / 255.0

        a = float(parts[3]) if len(parts) >= 4 else 1.0
        return (r, g, b, a)
    except ValueError:
        return None


def _parse_hsl_color(s: str) -> Optional[tuple[float, float, float, float]]:
    """Parse hsl/hsla color."""
    match = re.match(r"hsla?\s*\(\s*([^)]+)\s*\)", s)
    if not match:
        return None

    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) < 3:
        return None

    try:
        h = float(parts[0]) / 360.0
        s_val = float(parts[1].rstrip("%")) / 100.0
        l_val = float(parts[2].rstrip("%")) / 100.0
        a = float(parts[3]) if len(parts) >= 4 else 1.0

        r, g, b = _hsl_to_rgb(h, s_val, l_val)
        return (r, g, b, a)
    except ValueError:
        return None


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    """Convert HSL to RGB."""
    if s == 0:
        return (l, l, l)

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    r = _hue_to_rgb(p, q, h + 1/3)
    g = _hue_to_rgb(p, q, h)
    b = _hue_to_rgb(p, q, h - 1/3)

    return (r, g, b)


def _hue_to_rgb(p: float, q: float, t: float) -> float:
    """Helper for HSL to RGB conversion."""
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1/6:
        return p + (q - p) * 6 * t
    if t < 0.5:
        return q
    if t < 2/3:
        return p + (q - p) * (2/3 - t) * 6
    return p


def _evaluate_filter(expr: list, props: dict[str, Any]) -> bool:
    """Evaluate a Mapbox filter expression against feature properties."""
    if not expr:
        return True

    op = expr[0] if expr else None

    if op == "==" or op == "eq":
        if len(expr) != 3:
            return True
        key = expr[1]
        expected = expr[2]
        return props.get(key) == expected

    elif op == "!=" or op == "neq":
        if len(expr) != 3:
            return True
        key = expr[1]
        expected = expr[2]
        return props.get(key) != expected

    elif op == "all":
        return all(_evaluate_filter(sub, props) for sub in expr[1:] if isinstance(sub, list))

    elif op == "any":
        return any(_evaluate_filter(sub, props) for sub in expr[1:] if isinstance(sub, list))

    elif op == "has":
        if len(expr) != 2:
            return True
        return expr[1] in props

    elif op == "!" or op == "not":
        if len(expr) != 2:
            return True
        if isinstance(expr[1], list):
            return not _evaluate_filter(expr[1], props)
        return True

    elif op == "in":
        if len(expr) < 3:
            return True
        key = expr[1]
        val = props.get(key)
        return val in expr[2:]

    # Unknown operators pass through
    return True


# === SPRITE ATLAS API ===

@dataclass
class SpriteEntry:
    """A single sprite entry in an atlas."""
    name: str
    x: int
    y: int
    width: int
    height: int
    pixel_ratio: float = 1.0
    sdf: bool = False


@dataclass
class SpriteAtlas:
    """Sprite atlas for icons and patterns.
    
    Contains sprite metadata and image data for rendering icons
    defined in Mapbox GL Style Spec.
    """
    entries: dict[str, SpriteEntry]
    image_path: Optional[Path] = None
    image_data: Optional[bytes] = None
    width: int = 0
    height: int = 0
    
    def get_sprite(self, name: str) -> Optional[SpriteEntry]:
        """Get a sprite entry by name."""
        return self.entries.get(name)
    
    def sprite_names(self) -> list[str]:
        """Get list of all sprite names."""
        return list(self.entries.keys())


def load_sprite_atlas(
    json_path: Path | str,
    image_path: Optional[Path | str] = None,
) -> SpriteAtlas:
    """Load a sprite atlas from JSON and optional PNG.
    
    Args:
        json_path: Path to sprite JSON metadata file.
        image_path: Path to sprite PNG image (defaults to json_path with .png).
        
    Returns:
        SpriteAtlas with entries loaded.
        
    Example:
        >>> atlas = load_sprite_atlas("sprites@2x.json")
        >>> icon = atlas.get_sprite("park")
        >>> print(f"Park icon: {icon.width}x{icon.height}")
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)
    
    entries = {}
    for name, props in data.items():
        entries[name] = SpriteEntry(
            name=name,
            x=props.get("x", 0),
            y=props.get("y", 0),
            width=props.get("width", 0),
            height=props.get("height", 0),
            pixel_ratio=props.get("pixelRatio", 1.0),
            sdf=props.get("sdf", False),
        )
    
    # Determine image path
    if image_path is None:
        image_path = json_path.with_suffix(".png")
    else:
        image_path = Path(image_path)
    
    # Read image data if exists
    image_data = None
    width = height = 0
    if image_path.exists():
        image_data = image_path.read_bytes()
        # Try to get dimensions from PNG header
        try:
            # PNG header: 8 bytes magic, then IHDR chunk
            if image_data[:8] == b'\x89PNG\r\n\x1a\n':
                width = int.from_bytes(image_data[16:20], 'big')
                height = int.from_bytes(image_data[20:24], 'big')
        except Exception:
            pass
    
    return SpriteAtlas(
        entries=entries,
        image_path=image_path if image_path.exists() else None,
        image_data=image_data,
        width=width,
        height=height,
    )


# === GLYPH / FONT API ===

@dataclass
class GlyphRange:
    """A range of glyphs from a font.
    
    Glyph ranges are typically 256 characters each (e.g., 0-255, 256-511).
    """
    start: int
    end: int
    font_stack: str
    data: Optional[bytes] = None
    
    @property
    def range_id(self) -> str:
        """Get range identifier (e.g., '0-255')."""
        return f"{self.start}-{self.end}"


@dataclass
class FontStack:
    """Font stack for text rendering.
    
    A font stack is an ordered list of font names to try when rendering
    text. The first available font is used.
    """
    fonts: list[str]
    glyph_ranges: dict[str, GlyphRange] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        """Get font stack name (comma-joined font names)."""
        return ",".join(self.fonts)
    
    def has_range(self, start: int, end: int) -> bool:
        """Check if glyph range is loaded."""
        return f"{start}-{end}" in self.glyph_ranges
    
    def add_range(self, glyph_range: GlyphRange) -> None:
        """Add a glyph range to the font stack."""
        self.glyph_ranges[glyph_range.range_id] = glyph_range


def parse_glyph_url(
    template: str,
    font_stack: str,
    range_start: int,
    range_end: int,
) -> str:
    """Parse a glyph URL template.
    
    Args:
        template: URL template with {fontstack} and {range} placeholders.
        font_stack: Font stack name (comma-separated font names).
        range_start: Start of glyph range.
        range_end: End of glyph range.
        
    Returns:
        Resolved URL string.
        
    Example:
        >>> url = parse_glyph_url(
        ...     "https://example.com/fonts/{fontstack}/{range}.pbf",
        ...     "Open Sans Regular",
        ...     0, 255
        ... )
        >>> print(url)
        'https://example.com/fonts/Open%20Sans%20Regular/0-255.pbf'
    """
    import urllib.parse
    encoded_stack = urllib.parse.quote(font_stack, safe="")
    return (
        template
        .replace("{fontstack}", encoded_stack)
        .replace("{range}", f"{range_start}-{range_end}")
    )


def get_required_glyph_ranges(text: str) -> list[tuple[int, int]]:
    """Get glyph ranges needed to render text.
    
    Glyph ranges are 256 characters each.
    
    Args:
        text: Text to analyze.
        
    Returns:
        List of (start, end) tuples for required glyph ranges.
        
    Example:
        >>> ranges = get_required_glyph_ranges("Hello")
        >>> print(ranges)
        [(0, 255)]
    """
    ranges = set()
    for char in text:
        code = ord(char)
        start = (code // 256) * 256
        end = start + 255
        ranges.add((start, end))
    return sorted(ranges)

