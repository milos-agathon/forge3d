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
            center = np.asarray(candidate["anchor"][:2]) + np.asarray(glyph["origin"]) * 24
            expected = (math.atan2(center[0] - 200, 200 - center[1])
                        if angle == "arc" else math.radians(angle))
            error = math.atan2(math.sin(glyph["rotation"] - expected),
                               math.cos(glyph["rotation"] - expected))
            assert abs(error) <= 1e-4, (angle, center, glyph["rotation"], expected)
