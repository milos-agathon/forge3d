"""Curved authority rotations must match screen tangents at glyph centres."""
import math

import numpy as np
import pytest
import forge3d as f3d


@pytest.mark.parametrize("angle", (0, 30, -45, 90, "arc"))
def test_curved_glyph_rotation_matches_screen_tangent_at_center(angle):
    if angle == "arc":
        # Dense circle samples keep the polyline approximation below the
        # required 1e-4 radians. The oracle uses only the circle equation.
        theta = np.linspace(-math.pi / 3, math.pi / 3, 257)
        path = [(float(200 + 180 * math.sin(t)), float(200 - 180 * math.cos(t)), 0)
                for t in theta]
    else:
        theta = math.radians(angle)
        path = [(200, 200, 0), (200 + 300 * math.cos(theta), 200 + 300 * math.sin(theta), 0)]
    glyphs = [{"glyph_id": index + 1, "font_index": 0, "advance": [0.6, 0],
               "has_outline": True} for index in range(len("River"))]
    result = f3d.layout_label_candidates(kind="curved", label_id="river", text="River",
        screen_path=path, positioned_glyphs=glyphs, font_size=24)
    assert result is not None and result["candidates"]
    for candidate in result["candidates"]:
        emitted = candidate["positioned_glyphs"]
        assert len(emitted) == len(glyphs)
        for glyph in emitted:
            # `origin` is the pen (left end of the baseline); the glyph centre
            # is half an advance further along its rotation.
            pen = np.asarray(candidate["anchor"][:2]) + np.asarray(glyph["origin"]) * 24
            direction = np.array([math.cos(glyph["rotation"]), math.sin(glyph["rotation"])])
            center = pen + direction * glyph["advance"][0] * 24 * 0.5
            expected = (math.atan2(center[0] - 200, 200 - center[1])
                        if angle == "arc" else math.radians(angle))
            error = math.atan2(math.sin(glyph["rotation"] - expected),
                               math.cos(glyph["rotation"] - expected))
            assert abs(error) <= 1e-4, (angle, center, glyph["rotation"], expected)


@pytest.mark.parametrize("kind", ("curved", "line"))
@pytest.mark.parametrize("angle", (0, 30, -60))
def test_glyph_pens_advance_by_each_glyphs_own_advance(kind, angle):
    # Regression: the producer emitted glyph centres as pens, so each pen step
    # was the mean of two neighbouring advances and a wide glyph after a narrow
    # one ran into its neighbour ("Rim Trail" drew as "Ri mTrail").
    advances = [0.6, 0.25, 0.9, 0.28, 0.6, 0.4, 0.55, 0.25, 0.25]
    font_size, tracking = 20.0, 0.05
    theta = math.radians(angle)
    path = [(100, 300, 0), (100 + 600 * math.cos(theta), 300 + 600 * math.sin(theta), 0)]
    glyphs = [{"glyph_id": index + 1, "font_index": 0, "advance": [advance, 0],
               "has_outline": True} for index, advance in enumerate(advances)]
    result = f3d.layout_label_candidates(kind=kind, label_id="rim", text="Rim Trail",
        screen_path=path, positioned_glyphs=glyphs, font_size=font_size,
        arc_fractions=[0.5], tracking=tracking)
    assert result is not None
    (candidate,) = result["candidates"]
    emitted = candidate["positioned_glyphs"]
    direction = np.array([math.cos(theta), math.sin(theta)])
    pens = [np.asarray(glyph["origin"]) for glyph in emitted]
    for index in range(len(pens) - 1):
        step = float(np.dot(pens[index + 1] - pens[index], direction))
        assert step == pytest.approx(advances[index] + tracking, abs=1e-4), index
    # The text block is centred on the anchor (arc fraction 0.5).
    start = float(np.dot(pens[0], direction))
    end = float(np.dot(pens[-1], direction)) + advances[-1]
    assert start == pytest.approx(-end, abs=1e-4)
    # The reserved bounds cover every glyph from its pen to its far end.
    x0, y0, x1, y1 = candidate["bounds"]
    anchor = np.asarray(candidate["anchor"][:2])
    for pen, advance in zip(pens, advances):
        for point in (anchor + pen * font_size,
                      anchor + (pen + direction * advance) * font_size):
            assert x0 <= point[0] <= x1 and y0 <= point[1] <= y1
