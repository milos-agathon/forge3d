import os
import numpy as np
import pytest

try:
    import vulkan_forge._vulkan_forge as vf
except ImportError:
    try:
        import _vulkan_forge as vf
    except ImportError:
        pytest.skip("vulkan_forge module not available", allow_module_level=True)

def test_t31_uniform_lanes_layout():
    # Small offscreen to exercise pipeline creation
    spike = vf.TerrainSpike(256, 192, grid=64, colormap="viridis")
    # The constructor seeds uniforms; fetch raw 44-float view/proj + vec4 lanes
    u = spike.debug_uniforms_f32()
    assert isinstance(u, np.ndarray) and u.dtype == np.float32 and u.shape == (44,)

    # Layout indices:
    # [0..15]=view, [16..31]=proj, [32..35]=sun_exposure, [36..39]=spacing/h_range/exag/0, [40..43]=pad
    spacing, h_range, exag, zero = float(u[36]), float(u[37]), float(u[38]), float(u[39])

    # Defaults from Globals::default() are 1.0 for spacing, 1.0 for (h_max-h_min), 1.0 for exaggeration, pad lane = 0
    assert abs(spacing - 1.0) < 1e-6
    assert abs(h_range - 1.0) < 1e-6
    assert abs(exag   - 1.0) < 1e-6
    assert abs(zero)        < 1e-6

def test_t31_render_png_smoke(tmp_path):
    spike = vf.TerrainSpike(320, 240, grid=64, colormap="viridis")
    out = tmp_path / "terrain_smoke.png"
    spike.render_png(str(out))
    # File should exist and be non-trivial in size
    assert out.exists()
    assert out.stat().st_size > 4096