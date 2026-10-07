"""Physical NVIDIA Vulkan curved-label pixel regressions."""
import hashlib

import numpy as np
import pytest
import forge3d as f3d
from forge3d.helpers.offscreen import save_png_deterministic
from _terrain_runtime import terrain_rendering_available
from _curved_label_fixture import curved_label_scene, render_curved_label


pytestmark = pytest.mark.skipif(not terrain_rendering_available(),
                              reason="needs a terrain-capable GPU adapter")


def _nvidia(record_property):
    probe = f3d.device_probe("vulkan")
    assert probe["status"] == "ok" and probe["backend"] == "Vulkan", probe
    assert probe["vendor"] == 0x10DE and not probe["software_fallback"], probe
    record_property("adapter", probe)


def test_horizontal_curved_label_is_byte_identical_to_main_9f14e648(tmp_path, record_property):
    _nvidia(record_property)
    pixels, base, _ = render_curved_label(0)
    assert np.any(pixels != base)
    path = tmp_path / "horizontal-curved.png"
    save_png_deterministic(path, pixels)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Measured from the same fixture on main 9f14e648, RTX 3070 Vulkan.
    assert digest == "9e0d2484a397744e5ecece348e4f18c6b0d6cf64ecca4a719f467edfd530fe54"
    record_property("png_sha256", digest)


def test_30_degree_curved_label_pixels_fit_candidate_bounds_plus_halo(record_property):
    _nvidia(record_property)
    pixels, base, candidate = render_curved_label(30)
    ys, xs = np.nonzero(np.any(pixels != base, axis=2))
    assert len(xs), "native curved glyph pixels must be observable"
    left, top, right, bottom = candidate.bounds
    halo = 1.0  # The fixture's declared halo_width_px.
    assert np.all((xs + 0.5 >= left - halo) & (xs + 0.5 <= right + halo)
                  & (ys + 0.5 >= top - halo) & (ys + 0.5 <= bottom + halo))
    glyphs = candidate.details["positioned_glyphs"]
    assert glyphs and all(abs(g["rotation"] - np.pi / 6) <= 1e-4 for g in glyphs)
    record_property("candidate_bounds", candidate.bounds)
    record_property("glyph_pixel_bounds", [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])


@pytest.mark.parametrize("angle", (0, 30))
def test_curved_label_scene_render_matches_native_compositor(
        tmp_path, record_property, angle):
    """Exercise compile_plan, terrain rendering and the native compositor together."""
    from PIL import Image
    from forge3d import map_scene

    _nvidia(record_property)
    scene = curved_label_scene(angle)
    candidate = scene.compile_plan().label_plans["river"].accepted[0].candidate
    labeled_path = tmp_path / "labeled.png"
    scene.render(str(labeled_path))
    assert scene.last_render_backend == "gpu_terrain"
    base_scene = curved_label_scene(angle)
    base_scene.recipe.layers = []
    base_path = tmp_path / "terrain.png"
    base_scene.render(str(base_path))
    assert base_scene.last_render_backend == "gpu_terrain"
    pixels = np.asarray(Image.open(labeled_path).convert("RGBA"))
    base = np.asarray(Image.open(base_path).convert("RGBA"))
    ys, xs = np.nonzero(np.any(pixels != base, axis=2))
    assert len(xs), "a complete scene render must contain the curved label"
    if angle == 30:
        left, top, right, bottom = candidate.bounds
        halo = 1.0
        assert np.all((xs + 0.5 >= left - halo) & (xs + 0.5 <= right + halo)
                      & (ys + 0.5 >= top - halo) & (ys + 0.5 <= bottom + halo))
    expected, _ = map_scene._composite_native_label_layers(
        base, scene.recipe, scene.compiled_label_plans)
    np.testing.assert_array_equal(pixels, expected)
    record_property("scene_png_sha256", hashlib.sha256(labeled_path.read_bytes()).hexdigest())
