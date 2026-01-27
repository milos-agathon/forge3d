# python/forge3d/export.py
"""Vector export functionality for SVG and PDF formats.

This module provides print-grade vector export capabilities for terrain
overlays, including polygons, polylines, and text labels.

Example usage:
    from forge3d.export import export_svg, export_pdf, VectorScene

    # Create a scene with vector data
    scene = VectorScene()
    scene.add_polygon(exterior=[(0, 0), (100, 0), (50, 100)],
                      fill_color=(1, 0, 0, 1))
    scene.add_polyline(path=[(0, 50), (100, 50)],
                       stroke_color=(0, 0, 1, 1), stroke_width=2.0)
    scene.add_label("Mountain Peak", position=(50, 80), font_size=14)

    # Export to SVG
    export_svg(scene, "output.svg")

    # Export to PDF (requires cairosvg)
    export_pdf(scene, "output.pdf")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import xml.etree.ElementTree as ET


@dataclass
class VectorStyle:
    """Style configuration for vector primitives."""

    fill_color: Tuple[float, float, float, float] = (0.2, 0.4, 0.8, 1.0)
    """RGBA fill color (0-1 range)."""

    stroke_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """RGBA stroke color (0-1 range)."""

    stroke_width: float = 1.0
    """Stroke width in pixels/units."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "fill_color": list(self.fill_color),
            "stroke_color": list(self.stroke_color),
            "stroke_width": self.stroke_width,
        }


@dataclass
class LabelStyle:
    """Style configuration for text labels."""

    font_size: float = 14.0
    """Font size in pixels."""

    color: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
    """RGBA text color (0-1 range)."""

    halo_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8)
    """RGBA halo/outline color (0-1 range)."""

    halo_width: float = 1.5
    """Halo width in pixels (0 = no halo)."""

    font_family: str = "sans-serif"
    """CSS font family."""

    font_weight: str = "normal"
    """CSS font weight (normal, bold, etc.)."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "font_size": self.font_size,
            "color": list(self.color),
            "halo_color": list(self.halo_color),
            "halo_width": self.halo_width,
            "font_family": self.font_family,
            "font_weight": self.font_weight,
        }


@dataclass
class Polygon:
    """Polygon definition with exterior ring and optional holes."""

    exterior: List[Tuple[float, float]]
    """Exterior ring vertices as (x, y) tuples (CCW winding)."""

    holes: List[List[Tuple[float, float]]] = field(default_factory=list)
    """Interior rings (holes) as lists of (x, y) tuples (CW winding)."""

    style: VectorStyle = field(default_factory=VectorStyle)
    """Polygon style (fill and stroke)."""


@dataclass
class Polyline:
    """Polyline definition with path vertices."""

    path: List[Tuple[float, float]]
    """Path vertices as (x, y) tuples."""

    style: VectorStyle = field(default_factory=VectorStyle)
    """Polyline style (stroke only)."""


@dataclass
class Label:
    """Text label definition."""

    text: str
    """Label text content."""

    position: Tuple[float, float]
    """Label position as (x, y) tuple."""

    style: LabelStyle = field(default_factory=LabelStyle)
    """Label style configuration."""


@dataclass
class Bounds:
    """2D axis-aligned bounding box."""

    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 100.0
    max_y: float = 100.0

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
        )

    def expand_to_include(self, x: float, y: float) -> None:
        """Expand bounds to include a point."""
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def with_padding(self, padding: float) -> "Bounds":
        """Return new bounds with added padding."""
        return Bounds(
            min_x=self.min_x - padding,
            min_y=self.min_y - padding,
            max_x=self.max_x + padding,
            max_y=self.max_y + padding,
        )

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> Optional["Bounds"]:
        """Create bounds from a list of points."""
        if not points:
            return None

        bounds = cls(
            min_x=points[0][0],
            min_y=points[0][1],
            max_x=points[0][0],
            max_y=points[0][1],
        )
        for x, y in points[1:]:
            bounds.expand_to_include(x, y)
        return bounds


class VectorScene:
    """Container for vector geometry to be exported.

    Collects polygons, polylines, and labels for export to SVG or PDF format.
    """

    def __init__(self):
        self.polygons: List[Polygon] = []
        self.polylines: List[Polyline] = []
        self.labels: List[Label] = []
        self._bounds: Optional[Bounds] = None

    def add_polygon(
        self,
        exterior: List[Tuple[float, float]],
        holes: Optional[List[List[Tuple[float, float]]]] = None,
        fill_color: Tuple[float, float, float, float] = (0.2, 0.4, 0.8, 1.0),
        stroke_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        stroke_width: float = 1.0,
    ) -> None:
        """Add a polygon to the scene.

        Args:
            exterior: Exterior ring vertices (CCW winding).
            holes: Optional list of hole rings (CW winding).
            fill_color: RGBA fill color (0-1 range).
            stroke_color: RGBA stroke color (0-1 range).
            stroke_width: Stroke width in pixels.
        """
        style = VectorStyle(
            fill_color=fill_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        self.polygons.append(Polygon(
            exterior=exterior,
            holes=holes or [],
            style=style,
        ))
        self._bounds = None  # Invalidate cached bounds

    def add_polyline(
        self,
        path: List[Tuple[float, float]],
        stroke_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        stroke_width: float = 1.0,
    ) -> None:
        """Add a polyline to the scene.

        Args:
            path: Path vertices as (x, y) tuples.
            stroke_color: RGBA stroke color (0-1 range).
            stroke_width: Stroke width in pixels.
        """
        style = VectorStyle(
            fill_color=(0, 0, 0, 0),  # No fill for polylines
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        self.polylines.append(Polyline(path=path, style=style))
        self._bounds = None

    def add_label(
        self,
        text: str,
        position: Tuple[float, float],
        font_size: float = 14.0,
        color: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0),
        halo_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8),
        halo_width: float = 1.5,
        font_family: str = "sans-serif",
    ) -> None:
        """Add a text label to the scene.

        Args:
            text: Label text content.
            position: Label position as (x, y) tuple.
            font_size: Font size in pixels.
            color: RGBA text color (0-1 range).
            halo_color: RGBA halo color (0-1 range).
            halo_width: Halo width in pixels (0 = no halo).
            font_family: CSS font family.
        """
        style = LabelStyle(
            font_size=font_size,
            color=color,
            halo_color=halo_color,
            halo_width=halo_width,
            font_family=font_family,
        )
        self.labels.append(Label(text=text, position=position, style=style))
        self._bounds = None

    def compute_bounds(self, padding: float = 0.0) -> Bounds:
        """Compute bounding box of all geometry.

        Args:
            padding: Optional padding to add around bounds.

        Returns:
            Bounds object enclosing all geometry.
        """
        if self._bounds is not None and padding == 0.0:
            return self._bounds

        all_points: List[Tuple[float, float]] = []

        for polygon in self.polygons:
            all_points.extend(polygon.exterior)
            for hole in polygon.holes:
                all_points.extend(hole)

        for polyline in self.polylines:
            all_points.extend(polyline.path)

        for label in self.labels:
            all_points.append(label.position)

        if not all_points:
            return Bounds()

        bounds = Bounds.from_points(all_points)
        if bounds is None:
            return Bounds()

        self._bounds = bounds

        if padding > 0:
            return bounds.with_padding(padding)
        return bounds

    def clear(self) -> None:
        """Remove all geometry from the scene."""
        self.polygons.clear()
        self.polylines.clear()
        self.labels.clear()
        self._bounds = None


def _color_to_css(c: Tuple[float, float, float, float]) -> str:
    """Convert RGBA (0-1) to CSS color string."""
    r = int(max(0, min(255, c[0] * 255)))
    g = int(max(0, min(255, c[1] * 255)))
    b = int(max(0, min(255, c[2] * 255)))
    a = max(0.0, min(1.0, c[3]))

    if abs(a - 1.0) < 0.001:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"rgba({r},{g},{b},{a:.2f})"


def _project_to_screen(
    x: float, y: float,
    bounds: Bounds,
    width: int, height: int,
) -> Tuple[float, float]:
    """Project 2D coordinates to screen space."""
    range_x = bounds.width if bounds.width > 1e-10 else 1.0
    range_y = bounds.height if bounds.height > 1e-10 else 1.0

    norm_x = (x - bounds.min_x) / range_x
    norm_y = (y - bounds.min_y) / range_y

    # Y is flipped for screen coordinates
    screen_x = norm_x * width
    screen_y = (1.0 - norm_y) * height

    return (screen_x, screen_y)


def _escape_xml(text: str) -> str:
    """Escape text for XML."""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;"))


def generate_svg(
    scene: VectorScene,
    width: int = 800,
    height: int = 600,
    bounds: Optional[Bounds] = None,
    background: Optional[Tuple[float, float, float, float]] = None,
    precision: int = 2,
    include_labels: bool = True,
) -> str:
    """Generate SVG string from a vector scene.

    Args:
        scene: VectorScene containing geometry to export.
        width: SVG width in pixels.
        height: SVG height in pixels.
        bounds: Optional explicit bounds (auto-computed if None).
        background: Optional RGBA background color.
        precision: Coordinate decimal precision.
        include_labels: Whether to include text labels.

    Returns:
        Complete SVG document as a string.
    """
    if bounds is None:
        bounds = scene.compute_bounds(padding=10.0)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
    ]

    # Background
    if background is not None:
        bg_color = _color_to_css(background)
        lines.append(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>')

    # Polygons
    for polygon in scene.polygons:
        style = polygon.style
        fill = _color_to_css(style.fill_color) if style.fill_color[3] > 0.001 else "none"
        stroke = _color_to_css(style.stroke_color) if style.stroke_color[3] > 0.001 else "none"
        stroke_w = f"{style.stroke_width:.{precision}f}"

        if not polygon.holes:
            # Simple polygon
            points = " ".join(
                f"{_project_to_screen(x, y, bounds, width, height)[0]:.{precision}f},"
                f"{_project_to_screen(x, y, bounds, width, height)[1]:.{precision}f}"
                for x, y in polygon.exterior
            )
            lines.append(
                f'  <polygon points="{points}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_w}" stroke-linejoin="round"/>'
            )
        else:
            # Polygon with holes - use path
            d_parts = []

            # Exterior
            for i, (x, y) in enumerate(polygon.exterior):
                sx, sy = _project_to_screen(x, y, bounds, width, height)
                if i == 0:
                    d_parts.append(f"M{sx:.{precision}f},{sy:.{precision}f}")
                else:
                    d_parts.append(f"L{sx:.{precision}f},{sy:.{precision}f}")
            d_parts.append("Z")

            # Holes
            for hole in polygon.holes:
                for i, (x, y) in enumerate(hole):
                    sx, sy = _project_to_screen(x, y, bounds, width, height)
                    if i == 0:
                        d_parts.append(f"M{sx:.{precision}f},{sy:.{precision}f}")
                    else:
                        d_parts.append(f"L{sx:.{precision}f},{sy:.{precision}f}")
                d_parts.append("Z")

            d = " ".join(d_parts)
            lines.append(
                f'  <path d="{d}" fill-rule="evenodd" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_w}" stroke-linejoin="round"/>'
            )

    # Polylines
    for polyline in scene.polylines:
        style = polyline.style
        stroke = _color_to_css(style.stroke_color) if style.stroke_color[3] > 0.001 else "#000000"
        stroke_w = f"{style.stroke_width:.{precision}f}"

        points = " ".join(
            f"{_project_to_screen(x, y, bounds, width, height)[0]:.{precision}f},"
            f"{_project_to_screen(x, y, bounds, width, height)[1]:.{precision}f}"
            for x, y in polyline.path
        )
        lines.append(
            f'  <polyline points="{points}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # Labels
    if include_labels:
        for label in scene.labels:
            style = label.style
            sx, sy = _project_to_screen(
                label.position[0], label.position[1],
                bounds, width, height
            )

            text = _escape_xml(label.text)
            font_size = f"{style.font_size:.{precision}f}"

            common_attrs = (
                f'x="{sx:.{precision}f}" y="{sy:.{precision}f}" '
                f'font-family="{style.font_family}" font-size="{font_size}" '
                f'font-weight="{style.font_weight}" text-anchor="middle" '
                f'dominant-baseline="middle"'
            )

            # Halo (stroke behind text)
            if style.halo_width > 0 and style.halo_color[3] > 0.001:
                halo_color = _color_to_css(style.halo_color)
                halo_w = f"{style.halo_width * 2:.{precision}f}"
                lines.append(
                    f'  <text {common_attrs} fill="none" stroke="{halo_color}" '
                    f'stroke-width="{halo_w}" stroke-linejoin="round">{text}</text>'
                )

            # Main text
            text_color = _color_to_css(style.color)
            lines.append(f'  <text {common_attrs} fill="{text_color}">{text}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def export_svg(
    scene: VectorScene,
    path: Union[str, Path],
    width: int = 800,
    height: int = 600,
    bounds: Optional[Bounds] = None,
    background: Optional[Tuple[float, float, float, float]] = None,
    precision: int = 2,
    include_labels: bool = True,
) -> None:
    """Export vector scene to SVG file.

    Args:
        scene: VectorScene containing geometry to export.
        path: Output file path.
        width: SVG width in pixels.
        height: SVG height in pixels.
        bounds: Optional explicit bounds (auto-computed if None).
        background: Optional RGBA background color.
        precision: Coordinate decimal precision.
        include_labels: Whether to include text labels.
    """
    svg_content = generate_svg(
        scene=scene,
        width=width,
        height=height,
        bounds=bounds,
        background=background,
        precision=precision,
        include_labels=include_labels,
    )

    path = Path(path)
    path.write_text(svg_content, encoding='utf-8')


def export_pdf(
    scene: VectorScene,
    path: Union[str, Path],
    width: int = 800,
    height: int = 600,
    dpi: int = 300,
    bounds: Optional[Bounds] = None,
    background: Optional[Tuple[float, float, float, float]] = None,
    include_labels: bool = True,
) -> None:
    """Export vector scene to PDF file.

    Requires cairosvg package for PDF generation.

    Args:
        scene: VectorScene containing geometry to export.
        path: Output PDF file path.
        width: Output width in pixels.
        height: Output height in pixels.
        dpi: Output resolution (default 300 for print).
        bounds: Optional explicit bounds (auto-computed if None).
        background: Optional RGBA background color.
        include_labels: Whether to include text labels.

    Raises:
        ImportError: If cairosvg is not installed.
    """
    try:
        import cairosvg
    except ImportError:
        raise ImportError(
            "cairosvg is required for PDF export. "
            "Install with: pip install cairosvg"
        )

    # Generate SVG first
    svg_content = generate_svg(
        scene=scene,
        width=width,
        height=height,
        bounds=bounds,
        background=background,
        include_labels=include_labels,
    )

    # Convert to PDF
    path = Path(path)
    cairosvg.svg2pdf(
        bytestring=svg_content.encode('utf-8'),
        write_to=str(path),
        dpi=dpi,
    )


def validate_svg(svg_content: str) -> bool:
    """Validate SVG structure by parsing it.

    Args:
        svg_content: SVG document string.

    Returns:
        True if valid XML/SVG structure.

    Raises:
        xml.etree.ElementTree.ParseError: If XML is malformed.
    """
    root = ET.fromstring(svg_content)

    # Check root element
    if not root.tag.endswith('svg'):
        return False

    # Check for viewBox or width/height
    has_viewbox = 'viewBox' in root.attrib
    has_dimensions = 'width' in root.attrib and 'height' in root.attrib

    return has_viewbox or has_dimensions


# Convenience re-exports
__all__ = [
    'VectorScene',
    'VectorStyle',
    'LabelStyle',
    'Polygon',
    'Polyline',
    'Label',
    'Bounds',
    'generate_svg',
    'export_svg',
    'export_pdf',
    'validate_svg',
]
