"""Screen framing stays stable across the fullscreen and shared-mesh routes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import forge3d.map_scene as ms
from _concordia_scene import render
from _concordia_screen_fixtures import plateau_bounds, screen_framing_scene


def test_screen_framing_preserves_feature_location(tmp_path):
    """Keep each feature-bound edge within the owner-approved one output pixel."""
    bounds = {}
    frames = {}
    for effects in (False, True):
        model = screen_framing_scene(tmp_path, effects=effects)
        assert ms._mapscene_effective_camera_mode(model.recipe) == "screen"
        result = render(model)  # A missing physical frame fails, rather than skips.
        bounds[str(effects)] = plateau_bounds(result.rgba)
        frames[effects] = result.rgba
    directory = os.environ.get("FORGE3D_CONCORDIA_ARTIFACT_DIR")
    if directory:
        from PIL import Image
        target = Path(directory)/"screen-framing"
        target.mkdir(parents=True, exist_ok=True)
        for effects, frame in frames.items():
            Image.fromarray(frame).save(target/f"effects-{effects}.png")
        (target/"metrics.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    # The owner approved one output pixel per edge after physical fullscreen
    # and mesh rasterization differed at the left colour-threshold sample.
    assert all(abs(a - b) <= 1 for a, b in zip(bounds["False"], bounds["True"])), (
        "The same plateau moves by more than one output pixel per edge when "
        f"zero-intensity SSR selects interactive rendering: {bounds}")
