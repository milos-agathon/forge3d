"""The two label-compositor crashes Milos found, pinned.

1. `_composite_native_label_batches` called `_needs_native_label_composite(
   recipe, plans)` although neither name is a parameter — every label example
   (and every Studio map with labels) raised NameError. The guard belongs in
   `_composite_native_label_layers`, which has both.
2. `composite_label_layers` assumed an RGBA frame. Studio passes HxWx3 and the
   native compositor read it as RGBA, producing vertical stripes. The public
   entry now pads RGB for the draw and returns the caller's channel count.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge3d import map_scene


def _frame(channels: int = 3, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.integers(0, 255, size=(size, size, channels), dtype=np.uint8)


def test_label_batches_compositor_does_not_reference_recipe_or_plans():
    # The function takes only (base, batches); the guard that needs a recipe
    # must not live here. With no batches it returns the base untouched.
    base = _frame(4)
    out, drawn = map_scene._composite_native_label_batches(base, [])
    assert drawn is False
    assert out is base


def test_native_label_layers_guard_uses_its_own_parameters():
    recipe = map_scene.SceneRecipe(
        terrain=map_scene.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:4326",
            metadata={"width": 8, "height": 8},
        ),
        camera=map_scene.OrbitCamera(),
        lighting=map_scene.LightingPreset(),
        output=map_scene.OutputSpec(width=32, height=32),
    )
    base = _frame(4)
    # No label layers -> no native composite. Before the fix this path was
    # unreachable because the batches function raised first.
    out, drawn = map_scene._composite_native_label_layers(base, recipe, {})
    assert (out is base) and drawn is False


def test_public_compositor_accepts_rgb_and_returns_three_channels():
    base = _frame(3)
    # No layers/plans: the compositor returns the frame unchanged in shape.
    out, drawn = map_scene.composite_label_layers(base, [], {})
    assert drawn is False
    assert out.shape == base.shape


def test_public_compositor_rejects_an_unexpected_channel_count():
    with pytest.raises(ValueError, match="HxWx3 or HxWx4"):
        map_scene.composite_label_layers(_frame(2), [], {})
