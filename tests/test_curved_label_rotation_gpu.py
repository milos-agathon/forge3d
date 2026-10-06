"""Physical NVIDIA Vulkan curved-label pixel regressions."""
import hashlib

import numpy as np
import pytest
import forge3d as f3d
from forge3d.helpers.offscreen import save_png_deterministic
from _terrain_runtime import terrain_rendering_available
from _curved_label_fixture import render_curved_label


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
