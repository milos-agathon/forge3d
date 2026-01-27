#!/usr/bin/env python3
"""Vector Export Demo - Print-grade SVG/PDF export for terrain overlays.

Demonstrates the P5-export vector export functionality:
- Creating vector scenes with polygons, polylines, and labels
- Exporting to SVG format
- Optionally exporting to PDF (requires cairosvg)

Usage:
    python examples/vector_export_demo.py --output terrain_map.svg
    python examples/vector_export_demo.py --output terrain_map.svg --pdf
    python examples/vector_export_demo.py --demo-type contours --output contours.svg

Demo types:
    simple    - Simple triangle polygon with label (default)
    contours  - Simulated contour lines
    features  - Multiple polygons representing terrain features
    full      - Complete demo with all element types
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

from forge3d.export import (
    VectorScene,
    Bounds,
    generate_svg,
    export_svg,
    export_pdf,
    validate_svg,
)


def create_simple_demo() -> VectorScene:
    """Create a simple demo scene with basic elements."""
    scene = VectorScene()

    # Red triangle polygon
    scene.add_polygon(
        exterior=[(100, 100), (400, 100), (250, 350)],
        fill_color=(0.8, 0.2, 0.2, 0.6),
        stroke_color=(0.4, 0.0, 0.0, 1.0),
        stroke_width=2.0,
    )

    # Blue line
    scene.add_polyline(
        path=[(50, 200), (450, 200)],
        stroke_color=(0.0, 0.0, 0.8, 1.0),
        stroke_width=1.5,
    )

    # Label at the center
    scene.add_label(
        text="Demo Triangle",
        position=(250, 180),
        font_size=16,
        color=(0.1, 0.1, 0.1, 1.0),
        halo_color=(1.0, 1.0, 1.0, 0.9),
        halo_width=2.0,
    )

    return scene


def create_contour_demo(
    num_contours: int = 10,
    center: Tuple[float, float] = (250, 250),
    max_radius: float = 200,
) -> VectorScene:
    """Create a scene simulating topographic contour lines.

    Args:
        num_contours: Number of contour rings.
        center: Center point of contours.
        max_radius: Maximum radius of outermost contour.

    Returns:
        VectorScene with contour polylines.
    """
    scene = VectorScene()

    # Generate concentric contours
    for i in range(1, num_contours + 1):
        radius = (i / num_contours) * max_radius

        # Generate circle points
        num_points = 64
        path = []
        for j in range(num_points + 1):  # +1 to close the ring
            angle = (j / num_points) * 2 * math.pi
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            path.append((x, y))

        # Every 5th contour is bold (index contour)
        is_index = i % 5 == 0
        stroke_width = 1.5 if is_index else 0.5

        scene.add_polyline(
            path=path,
            stroke_color=(0.4, 0.3, 0.2, 1.0),
            stroke_width=stroke_width,
        )

        # Add elevation label on index contours
        if is_index:
            elevation = i * 50  # Meters
            label_x = center[0] + radius * 0.7
            label_y = center[1]
            scene.add_label(
                text=f"{elevation}m",
                position=(label_x, label_y),
                font_size=10,
                color=(0.3, 0.2, 0.1, 1.0),
                halo_color=(1.0, 1.0, 1.0, 0.8),
                halo_width=1.0,
            )

    # Peak marker
    scene.add_polygon(
        exterior=[
            (center[0] - 5, center[1] + 5),
            (center[0] + 5, center[1] + 5),
            (center[0], center[1] - 8),
        ],
        fill_color=(0.2, 0.2, 0.2, 1.0),
        stroke_color=(0.0, 0.0, 0.0, 1.0),
        stroke_width=1.0,
    )

    scene.add_label(
        text="Peak 500m",
        position=(center[0], center[1] - 20),
        font_size=12,
        color=(0.1, 0.1, 0.1, 1.0),
        halo_width=1.5,
    )

    return scene


def create_features_demo() -> VectorScene:
    """Create a scene with multiple terrain feature polygons."""
    scene = VectorScene()

    # Forest polygon (green)
    forest = [
        (50, 300), (150, 250), (200, 300), (180, 400),
        (100, 420), (30, 380),
    ]
    scene.add_polygon(
        exterior=forest,
        fill_color=(0.2, 0.5, 0.2, 0.6),
        stroke_color=(0.1, 0.3, 0.1, 1.0),
        stroke_width=1.0,
    )
    scene.add_label(
        text="Forest",
        position=(115, 350),
        font_size=11,
        color=(0.1, 0.3, 0.1, 1.0),
    )

    # Lake polygon (blue) with island hole
    lake = [
        (250, 280), (400, 250), (420, 350), (380, 420),
        (280, 400), (230, 340),
    ]
    island = [
        (320, 320), (350, 310), (360, 340), (340, 360), (310, 350),
    ]
    scene.add_polygon(
        exterior=lake,
        holes=[island],
        fill_color=(0.2, 0.4, 0.8, 0.7),
        stroke_color=(0.1, 0.2, 0.5, 1.0),
        stroke_width=1.5,
    )
    scene.add_label(
        text="Lake",
        position=(330, 380),
        font_size=11,
        color=(0.1, 0.2, 0.5, 1.0),
    )

    # Urban area polygon (gray)
    urban = [
        (100, 50), (250, 50), (280, 100), (260, 180),
        (150, 200), (80, 150), (70, 80),
    ]
    scene.add_polygon(
        exterior=urban,
        fill_color=(0.6, 0.6, 0.6, 0.5),
        stroke_color=(0.3, 0.3, 0.3, 1.0),
        stroke_width=1.0,
    )
    scene.add_label(
        text="Town",
        position=(170, 120),
        font_size=11,
        color=(0.2, 0.2, 0.2, 1.0),
    )

    # Road polyline
    road = [
        (0, 200), (100, 180), (200, 250), (350, 220), (500, 180),
    ]
    scene.add_polyline(
        path=road,
        stroke_color=(0.4, 0.3, 0.2, 1.0),
        stroke_width=3.0,
    )

    # River polyline
    river = [
        (450, 0), (420, 100), (350, 180), (330, 300), (340, 450),
    ]
    scene.add_polyline(
        path=river,
        stroke_color=(0.3, 0.5, 0.8, 1.0),
        stroke_width=2.0,
    )

    return scene


def create_full_demo() -> VectorScene:
    """Create a comprehensive demo with all element types."""
    scene = VectorScene()

    # Background reference frame
    for i in range(0, 501, 50):
        # Horizontal grid lines
        scene.add_polyline(
            path=[(0, i), (500, i)],
            stroke_color=(0.8, 0.8, 0.8, 0.5),
            stroke_width=0.5,
        )
        # Vertical grid lines
        scene.add_polyline(
            path=[(i, 0), (i, 500)],
            stroke_color=(0.8, 0.8, 0.8, 0.5),
            stroke_width=0.5,
        )

    # Mountain ridge polygon
    ridge = [
        (50, 400), (100, 300), (150, 350), (200, 250),
        (280, 280), (350, 200), (400, 250), (450, 350),
        (480, 400), (450, 450), (50, 450),
    ]
    scene.add_polygon(
        exterior=ridge,
        fill_color=(0.5, 0.4, 0.3, 0.4),
        stroke_color=(0.3, 0.2, 0.1, 1.0),
        stroke_width=1.5,
    )

    # Peak markers
    peaks = [
        ((200, 250), "Mt. Alpha", 2340),
        ((350, 200), "Mt. Beta", 2890),
    ]
    for (x, y), name, elevation in peaks:
        # Triangle marker
        scene.add_polygon(
            exterior=[(x-6, y+4), (x+6, y+4), (x, y-8)],
            fill_color=(0.2, 0.2, 0.2, 1.0),
            stroke_color=(0.0, 0.0, 0.0, 1.0),
            stroke_width=1.0,
        )
        scene.add_label(
            text=f"{name}\n{elevation}m",
            position=(x, y-20),
            font_size=10,
            color=(0.1, 0.1, 0.1, 1.0),
            halo_width=1.5,
        )

    # Trail polyline
    trail = [
        (50, 480), (100, 420), (150, 400), (200, 350),
        (250, 320), (300, 300), (350, 280), (400, 310),
        (450, 380),
    ]
    scene.add_polyline(
        path=trail,
        stroke_color=(0.6, 0.2, 0.1, 1.0),
        stroke_width=2.0,
    )

    # Title
    scene.add_label(
        text="Terrain Map Demo",
        position=(250, 30),
        font_size=20,
        color=(0.1, 0.1, 0.1, 1.0),
        halo_color=(1.0, 1.0, 1.0, 0.9),
        halo_width=2.5,
    )

    # Scale bar representation
    scene.add_polyline(
        path=[(50, 480), (150, 480)],
        stroke_color=(0.0, 0.0, 0.0, 1.0),
        stroke_width=2.0,
    )
    scene.add_label(
        text="1 km",
        position=(100, 495),
        font_size=9,
        halo_width=1.0,
    )

    return scene


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vector Export Demo - Generate print-grade SVG/PDF maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--demo-type",
        choices=["simple", "contours", "features", "full"],
        default="simple",
        help="Type of demo to generate (default: simple)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("vector_export_demo.svg"),
        help="Output file path (default: vector_export_demo.svg)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export to PDF (requires cairosvg)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Output width in pixels (default: 800)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Output height in pixels (default: 600)",
    )

    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Background color as hex (e.g., #f0f0f0)",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Coordinate decimal precision (default: 2)",
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Exclude text labels from output",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated SVG structure",
    )

    args = parser.parse_args()

    # Create scene based on demo type
    print(f"Creating {args.demo_type} demo scene...")

    if args.demo_type == "simple":
        scene = create_simple_demo()
    elif args.demo_type == "contours":
        scene = create_contour_demo()
    elif args.demo_type == "features":
        scene = create_features_demo()
    elif args.demo_type == "full":
        scene = create_full_demo()
    else:
        raise ValueError(f"Unknown demo type: {args.demo_type}")

    # Parse background color
    background = None
    if args.background:
        bg = args.background.lstrip('#')
        if len(bg) == 6:
            r = int(bg[0:2], 16) / 255.0
            g = int(bg[2:4], 16) / 255.0
            b = int(bg[4:6], 16) / 255.0
            background = (r, g, b, 1.0)

    # Print scene statistics
    bounds = scene.compute_bounds()
    print(f"Scene contains:")
    print(f"  - {len(scene.polygons)} polygon(s)")
    print(f"  - {len(scene.polylines)} polyline(s)")
    print(f"  - {len(scene.labels)} label(s)")
    print(f"Bounds: ({bounds.min_x:.1f}, {bounds.min_y:.1f}) to ({bounds.max_x:.1f}, {bounds.max_y:.1f})")

    # Export to SVG
    print(f"\nExporting to SVG: {args.output}")
    export_svg(
        scene,
        args.output,
        width=args.width,
        height=args.height,
        background=background,
        precision=args.precision,
        include_labels=not args.no_labels,
    )

    # Validate if requested
    if args.validate:
        print("Validating SVG structure...")
        content = args.output.read_text(encoding='utf-8')
        if validate_svg(content):
            print("  [OK] SVG is valid")
        else:
            print("  [FAIL] SVG validation failed")

    # Get file size
    svg_size = args.output.stat().st_size
    print(f"SVG file size: {svg_size:,} bytes")

    # Export to PDF if requested
    if args.pdf:
        pdf_path = args.output.with_suffix('.pdf')
        print(f"\nExporting to PDF: {pdf_path}")
        try:
            export_pdf(
                scene,
                pdf_path,
                width=args.width,
                height=args.height,
                background=background,
                include_labels=not args.no_labels,
            )
            pdf_size = pdf_path.stat().st_size
            print(f"PDF file size: {pdf_size:,} bytes")
        except ImportError as e:
            print(f"  [WARN] PDF export skipped: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
