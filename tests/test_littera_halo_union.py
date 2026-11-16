"""Independent geometry checks: shared CPU/SVG code cannot be its own oracle."""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pytest

from forge3d import text

ROOT = Path(__file__).resolve().parents[1]


def _quadrant_area(x, y, radius):
    x, y = min(max(x, 0.0), radius), min(max(y, 0.0), radius)
    switch = math.sqrt(max(0.0, radius * radius - y * y))
    if x <= switch:
        return x * y
    def primitive(t):
        return 0.5 * (t * math.sqrt(max(0.0, radius * radius - t * t))
                      + radius * radius * math.asin(t / radius))
    return switch * y + primitive(x) - primitive(switch)


def _rounded_rectangle_pixel(x, y, bounds, radius):
    left, top, right, bottom = bounds
    def rectangle(a, b, c, d):
        return max(0.0, min(x + 1, c) - max(x, a)) * max(0.0, min(y + 1, d) - max(y, b))
    area = rectangle(left, top, right, bottom)
    area += rectangle(left - radius, top, left, bottom)
    area += rectangle(right, top, right + radius, bottom)
    area += rectangle(left, top - radius, right, top)
    area += rectangle(left, bottom, right, bottom + radius)
    for cx, sx in ((left, -1), (right, 1)):
        for cy, sy in ((top, -1), (bottom, 1)):
            a, b = sorted((sx * (x - cx), sx * (x + 1 - cx)))
            c, d = sorted((sy * (y - cy), sy * (y + 1 - cy)))
            area += (_quadrant_area(b, d, radius) - _quadrant_area(a, d, radius)
                     - _quadrant_area(b, c, radius) + _quadrant_area(a, c, radius))
    return area


@pytest.mark.parametrize("origin", [(0.0, 0.0), (127.25, 63.5)])
def test_round_halo_matches_exact_disk_dilation(origin):
    bounds = (3.25, 4.375, 9.5, 10.75)
    radius = 1.25
    path = "M3.25 4.375L9.5 4.375L9.5 10.75L3.25 10.75Z"
    width, height = math.ceil(origin[0] + 12), math.ceil(origin[1] + 13)
    actual = text.rasterize_svg_path(path, width, height, origin=origin, halo_width=radius)
    translated = tuple(v + origin[i % 2] for i, v in enumerate(bounds))
    expected = np.array([[_rounded_rectangle_pixel(x, y, translated, radius)
                          for x in range(width)] for y in range(height)])
    # Stroke subdivision follows half a UNORM8 output quantum, with the f32
    # coordinate ULP as its representability floor. The exact disk reference
    # remains independent; bound the displaced boundary's area geometrically.
    tolerance = max(0.5 / np.iinfo(np.uint8).max,
                    max(translated) * np.finfo(np.float32).eps)
    delta = math.sqrt(2) * tolerance
    perimeter = 2 * ((bounds[2] - bounds[0]) + (bounds[3] - bounds[1])) + 2 * math.pi * radius
    bound = perimeter * delta + math.pi * delta * delta + np.finfo(np.float32).eps
    error = float(np.max(np.abs(actual - expected)))
    print(f"exact disk dilation origin={origin} max_error={error:.9g} geometric_bound={bound:.9g}")
    assert error <= bound
    assert error <= 1 / np.iinfo(np.uint8).max


def _path_area_roundoff(path, radius, origin):
    commands = re.findall(r"([MLQCZ])([^MLQCZ]*)", path)
    perimeter, points = 0.0, []
    first = previous = None
    for command, coordinates in commands:
        values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", coordinates)]
        current = np.array(values).reshape(-1, 2)
        if command == "M":
            first = previous = current[0]
        elif command == "Z":
            perimeter += float(np.linalg.norm(previous - first))
            previous = first
        else:
            for point in current:
                perimeter += float(np.linalg.norm(point - previous))
                previous = point
        points.extend(current)
    # Control-polygon length bounds Bezier length. Fill and both stroke sides,
    # plus all possible round joins, bound the union's boundary-band area.
    scale = max(1.0, float(np.max(np.abs(np.array(points) + origin))))
    delta = math.sqrt(2) * scale * np.finfo(np.float32).eps
    boundary = 3 * perimeter + 2 * math.pi * radius * len(commands)
    return boundary * delta + math.pi * delta * delta + np.finfo(np.float32).eps


@pytest.mark.parametrize("font,character", [
    ("NotoSansLatin-subset.ttf", "f"),
    ("NotoSansArabic-subset.ttf", "س"),
    ("NotoSansDevanagari-subset.ttf", "क"),
])
def test_overlapping_glyph_halo_obeys_union_bounds(font, character):
    shaped = text.shape(character, [str(ROOT / "assets/fonts" / font)], 32.0)
    path = shaped.svg_path(precision=9)
    origin = (24.0, 64.0)
    first = text.rasterize_svg_path(path, 128, 96, origin=origin, halo_width=1.0)
    # Translate the second copy in path coordinates, not via a shared rasterizer.
    shift = (1.125, 0.375)
    def translated(match):
        command, coordinates = match.groups()
        if command == "Z":
            return command
        values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", coordinates)]
        return command + " ".join(format(v + shift[i % 2], ".9g") for i, v in enumerate(values))
    moved = re.sub(r"([MLQCZ])([^MLQCZ]*)", translated, path)
    second = text.rasterize_svg_path(moved, 128, 96, origin=origin, halo_width=1.0)
    union = text.rasterize_svg_path(path + moved, 128, 96, origin=origin, halo_width=1.0)
    bound = _path_area_roundoff(path + moved, 1.0, origin)
    loss = float(np.max(np.maximum(first, second) - union))
    excess = float(np.max(union - np.minimum(1.0, first + second)))
    print(f"{character!r} halo union loss={loss:.9g} excess={excess:.9g} geometric_bound={bound:.9g}")
    assert loss <= bound
    assert excess <= bound
    duplicated = text.rasterize_svg_path(path + path, 128, 96, origin=origin, halo_width=1.0)
    np.testing.assert_allclose(duplicated, first, rtol=0, atol=np.finfo(np.float32).eps)
