"""Tests for SVG export functionality (P5-export).

Tests:
- SVG generation from VectorScene
- Polygon rendering (simple and with holes)
- Polyline rendering
- Label rendering with halos
- Bounds calculation
- XML validity
"""

import pytest
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

from forge3d.export import (
    VectorScene,
    VectorStyle,
    LabelStyle,
    Polygon,
    Polyline,
    Label,
    Bounds,
    generate_svg,
    export_svg,
    export_pdf,
    validate_svg,
)


class TestBounds:
    """Test Bounds class."""

    def test_bounds_creation(self):
        """Test basic bounds creation."""
        bounds = Bounds(min_x=0, min_y=0, max_x=100, max_y=100)
        assert bounds.width == 100
        assert bounds.height == 100
        assert bounds.center == (50, 50)

    def test_bounds_from_points(self):
        """Test bounds from point list."""
        points = [(10, 20), (30, 40), (5, 35)]
        bounds = Bounds.from_points(points)
        assert bounds is not None
        assert bounds.min_x == 5
        assert bounds.min_y == 20
        assert bounds.max_x == 30
        assert bounds.max_y == 40

    def test_bounds_empty_points(self):
        """Test bounds from empty point list."""
        bounds = Bounds.from_points([])
        assert bounds is None

    def test_bounds_expand(self):
        """Test bounds expansion."""
        bounds = Bounds(min_x=0, min_y=0, max_x=10, max_y=10)
        bounds.expand_to_include(20, -5)
        assert bounds.min_x == 0
        assert bounds.min_y == -5
        assert bounds.max_x == 20
        assert bounds.max_y == 10

    def test_bounds_padding(self):
        """Test bounds with padding."""
        bounds = Bounds(min_x=10, min_y=10, max_x=90, max_y=90)
        padded = bounds.with_padding(5)
        assert padded.min_x == 5
        assert padded.min_y == 5
        assert padded.max_x == 95
        assert padded.max_y == 95


class TestVectorScene:
    """Test VectorScene class."""

    def test_empty_scene(self):
        """Test empty scene."""
        scene = VectorScene()
        assert len(scene.polygons) == 0
        assert len(scene.polylines) == 0
        assert len(scene.labels) == 0

    def test_add_polygon(self):
        """Test adding a polygon."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(0, 0), (100, 0), (50, 100)],
            fill_color=(1, 0, 0, 1),
            stroke_color=(0, 0, 0, 1),
            stroke_width=2.0,
        )
        assert len(scene.polygons) == 1
        assert scene.polygons[0].style.fill_color == (1, 0, 0, 1)
        assert scene.polygons[0].style.stroke_width == 2.0

    def test_add_polygon_with_holes(self):
        """Test adding a polygon with holes."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(0, 0), (100, 0), (100, 100), (0, 100)],
            holes=[[(25, 25), (75, 25), (75, 75), (25, 75)]],
        )
        assert len(scene.polygons) == 1
        assert len(scene.polygons[0].holes) == 1

    def test_add_polyline(self):
        """Test adding a polyline."""
        scene = VectorScene()
        scene.add_polyline(
            path=[(0, 0), (50, 50), (100, 0)],
            stroke_color=(0, 0, 1, 1),
            stroke_width=1.5,
        )
        assert len(scene.polylines) == 1
        assert scene.polylines[0].style.stroke_color == (0, 0, 1, 1)

    def test_add_label(self):
        """Test adding a label."""
        scene = VectorScene()
        scene.add_label(
            text="Test Label",
            position=(50, 50),
            font_size=14,
            color=(0, 0, 0, 1),
            halo_width=2.0,
        )
        assert len(scene.labels) == 1
        assert scene.labels[0].text == "Test Label"
        assert scene.labels[0].style.halo_width == 2.0

    def test_compute_bounds(self):
        """Test bounds computation."""
        scene = VectorScene()
        scene.add_polygon(exterior=[(10, 10), (90, 10), (50, 90)])
        scene.add_polyline(path=[(0, 50), (100, 50)])

        bounds = scene.compute_bounds()
        assert bounds.min_x == 0
        assert bounds.max_x == 100
        assert bounds.min_y == 10
        assert bounds.max_y == 90

    def test_clear(self):
        """Test clearing scene."""
        scene = VectorScene()
        scene.add_polygon(exterior=[(0, 0), (1, 0), (0, 1)])
        scene.add_polyline(path=[(0, 0), (1, 1)])
        scene.add_label(text="Test", position=(0.5, 0.5))

        scene.clear()
        assert len(scene.polygons) == 0
        assert len(scene.polylines) == 0
        assert len(scene.labels) == 0


class TestSvgGeneration:
    """Test SVG generation."""

    def test_empty_svg(self):
        """Test SVG from empty scene."""
        scene = VectorScene()
        svg = generate_svg(scene)

        assert '<?xml version="1.0"' in svg
        assert '<svg' in svg
        assert '</svg>' in svg
        assert validate_svg(svg)

    def test_polygon_svg(self):
        """Test SVG with polygon."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(10, 10), (90, 10), (50, 90)],
            fill_color=(1, 0, 0, 1),
        )

        svg = generate_svg(scene, width=100, height=100)
        assert '<polygon' in svg
        assert 'fill="#ff0000"' in svg
        assert validate_svg(svg)

    def test_polygon_with_holes_svg(self):
        """Test SVG with polygon containing holes."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(0, 0), (100, 0), (100, 100), (0, 100)],
            holes=[[(25, 25), (75, 25), (75, 75), (25, 75)]],
        )

        svg = generate_svg(scene)
        assert '<path' in svg
        assert 'fill-rule="evenodd"' in svg
        assert validate_svg(svg)

    def test_polyline_svg(self):
        """Test SVG with polyline."""
        scene = VectorScene()
        scene.add_polyline(
            path=[(0, 50), (100, 50)],
            stroke_color=(0, 0, 1, 1),
            stroke_width=2.0,
        )

        svg = generate_svg(scene)
        assert '<polyline' in svg
        assert 'stroke="#0000ff"' in svg
        assert 'fill="none"' in svg
        assert validate_svg(svg)

    def test_label_svg(self):
        """Test SVG with labels."""
        scene = VectorScene()
        scene.add_label(
            text="Mountain Peak",
            position=(50, 50),
            font_size=14,
            halo_width=2.0,
        )

        svg = generate_svg(scene, include_labels=True)
        assert '<text' in svg
        assert 'Mountain Peak' in svg
        assert 'font-size="14' in svg
        assert validate_svg(svg)

    def test_label_halo_svg(self):
        """Test SVG label with halo."""
        scene = VectorScene()
        scene.add_label(
            text="Label",
            position=(50, 50),
            halo_width=2.0,
            halo_color=(1, 1, 1, 0.8),
        )

        svg = generate_svg(scene)
        # Should have two text elements (halo + main)
        assert svg.count('<text') == 2
        assert 'stroke=' in svg  # Halo uses stroke

    def test_labels_disabled(self):
        """Test SVG without labels."""
        scene = VectorScene()
        scene.add_label(text="Hidden", position=(50, 50))

        svg = generate_svg(scene, include_labels=False)
        assert '<text' not in svg
        assert 'Hidden' not in svg

    def test_background_color(self):
        """Test SVG with background color."""
        scene = VectorScene()
        svg = generate_svg(scene, background=(0.9, 0.9, 0.9, 1.0))
        assert '<rect' in svg
        assert validate_svg(svg)

    def test_viewbox(self):
        """Test SVG viewBox attribute."""
        scene = VectorScene()
        svg = generate_svg(scene, width=800, height=600)
        assert 'viewBox="0 0 800 600"' in svg

    def test_coordinate_precision(self):
        """Test coordinate precision."""
        scene = VectorScene()
        scene.add_polygon(exterior=[(0.12345, 0.12345), (1, 0), (0.5, 1)])

        svg = generate_svg(scene, precision=2)
        # Should not have more than 2 decimal places
        assert '.12345' not in svg

    def test_xml_escape(self):
        """Test XML escaping in labels."""
        scene = VectorScene()
        scene.add_label(text="<Test & Label>", position=(50, 50), halo_width=0)

        svg = generate_svg(scene)
        assert '&lt;Test' in svg
        assert '&amp;' in svg
        assert '&gt;' in svg
        assert validate_svg(svg)


class TestSvgExport:
    """Test SVG file export."""

    def test_export_svg_file(self):
        """Test exporting SVG to file."""
        scene = VectorScene()
        scene.add_polygon(exterior=[(0, 0), (100, 0), (50, 100)])
        scene.add_polyline(path=[(0, 50), (100, 50)])
        scene.add_label(text="Test", position=(50, 50))

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            path = Path(f.name)

        try:
            export_svg(scene, path)
            assert path.exists()

            content = path.read_text(encoding='utf-8')
            assert '<?xml version' in content
            assert '<svg' in content
            assert '<polygon' in content
            assert '<polyline' in content
            assert '<text' in content
            assert validate_svg(content)
        finally:
            path.unlink(missing_ok=True)

    def test_export_with_custom_bounds(self):
        """Test export with explicit bounds."""
        scene = VectorScene()
        scene.add_polygon(exterior=[(0, 0), (10, 0), (5, 10)])

        bounds = Bounds(min_x=-100, min_y=-100, max_x=200, max_y=200)

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            path = Path(f.name)

        try:
            export_svg(scene, path, bounds=bounds)
            content = path.read_text(encoding='utf-8')
            assert validate_svg(content)
        finally:
            path.unlink(missing_ok=True)


class TestPdfExport:
    """Test PDF export (requires cairosvg)."""

    def test_pdf_export_import_error(self):
        """Test PDF export raises ImportError if cairosvg not installed."""
        # This test assumes cairosvg may or may not be installed
        scene = VectorScene()
        scene.add_polygon(exterior=[(0, 0), (100, 0), (50, 100)])

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            path = Path(f.name)

        try:
            try:
                import cairosvg
                # cairosvg is installed, should work
                export_pdf(scene, path)
                assert path.exists()
                assert path.stat().st_size > 0
            except ImportError:
                # cairosvg not installed, should raise ImportError
                with pytest.raises(ImportError, match="cairosvg"):
                    export_pdf(scene, path)
        finally:
            path.unlink(missing_ok=True)


class TestSvgValidation:
    """Test SVG validation."""

    def test_valid_svg(self):
        """Test valid SVG passes validation."""
        svg = '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"></svg>'
        assert validate_svg(svg)

    def test_valid_svg_with_dimensions(self):
        """Test valid SVG with width/height."""
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        assert validate_svg(svg)

    def test_invalid_xml(self):
        """Test invalid XML fails validation."""
        svg = '<svg><unclosed>'
        with pytest.raises(ET.ParseError):
            validate_svg(svg)

    def test_missing_viewbox_and_dimensions(self):
        """Test SVG without viewBox or dimensions fails."""
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        assert not validate_svg(svg)


class TestColorConversion:
    """Test color conversion utilities."""

    def test_opaque_color(self):
        """Test opaque color conversion."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(0, 0), (1, 0), (0.5, 1)],
            fill_color=(1.0, 0.0, 0.0, 1.0),  # Pure red, fully opaque
        )
        svg = generate_svg(scene)
        assert '#ff0000' in svg

    def test_transparent_color(self):
        """Test semi-transparent color conversion."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(0, 0), (1, 0), (0.5, 1)],
            fill_color=(1.0, 0.0, 0.0, 0.5),  # 50% transparent red
        )
        svg = generate_svg(scene)
        assert 'rgba(255,0,0,0.50)' in svg

    def test_fully_transparent_no_fill(self):
        """Test fully transparent fill becomes 'none'."""
        scene = VectorScene()
        scene.add_polygon(
            exterior=[(0, 0), (1, 0), (0.5, 1)],
            fill_color=(1.0, 0.0, 0.0, 0.0),  # Fully transparent
        )
        svg = generate_svg(scene)
        assert 'fill="none"' in svg


class TestIntegration:
    """Integration tests with complex scenes."""

    def test_complex_scene(self):
        """Test scene with multiple elements."""
        scene = VectorScene()

        # Multiple polygons
        for i in range(5):
            scene.add_polygon(
                exterior=[(i*20, 0), (i*20+15, 0), (i*20+7.5, 15)],
                fill_color=(i/5, 0, 1-i/5, 0.8),
            )

        # Multiple polylines
        for i in range(3):
            scene.add_polyline(
                path=[(0, i*30+20), (100, i*30+20)],
                stroke_color=(0, 0, 0, 1),
                stroke_width=1+i,
            )

        # Multiple labels
        for i, name in enumerate(["A", "B", "C"]):
            scene.add_label(
                text=name,
                position=(25+i*25, 80),
                font_size=12+i*2,
            )

        svg = generate_svg(scene, width=800, height=600)

        # Verify all elements present
        assert svg.count('<polygon') == 5
        assert svg.count('<polyline') == 3
        # Labels have 2 text elements each (halo + main)
        assert svg.count('<text') == 6

        # Validate XML
        assert validate_svg(svg)

    def test_large_coordinate_values(self):
        """Test with large coordinate values (geographic-like)."""
        scene = VectorScene()

        # Simulate lat/lon coordinates
        scene.add_polygon(
            exterior=[
                (-122.4194, 37.7749),
                (-122.4000, 37.7749),
                (-122.4100, 37.7900),
            ]
        )

        bounds = Bounds(
            min_x=-122.5,
            min_y=37.7,
            max_x=-122.3,
            max_y=37.9,
        )

        svg = generate_svg(scene, bounds=bounds, precision=4)
        assert validate_svg(svg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
