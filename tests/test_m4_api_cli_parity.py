#!/usr/bin/env python3
"""
Milestone 4: Python API & CLI parity tests

- API exposes `mode` parameter: full|ndf|g|dfg|spec|roughness
- Calling with mode overrides individual boolean toggles.
- Basic validations ensure shape, dtype, and non-uniformity for `mode='g'`.
- Invalid mode raises ValueError.
"""
import numpy as np
import pytest

try:
    import forge3d as f3d
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False

try:
    import forge3d._forge3d as f3d_native
    NATIVE_AVAILABLE = hasattr(f3d_native, 'render_brdf_tile') if FORGE3D_AVAILABLE else False
except (ImportError, AttributeError):
    NATIVE_AVAILABLE = False

skip_if_no_forge3d = pytest.mark.skipif(
    not FORGE3D_AVAILABLE,
    reason="forge3d not available (build with: maturin develop --release)",
)
skip_if_no_native = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="Native module with GPU support not available",
)


@skip_if_no_forge3d
@skip_if_no_native
class TestM4ApiMode:
    def test_mode_g_non_uniform(self):
        width = height = 128
        tile = f3d.render_brdf_tile("ggx", 0.5, width, height, mode="g")
        assert isinstance(tile, np.ndarray)
        assert tile.shape == (height, width, 4)
        assert tile.dtype == np.uint8

        # Check non-uniform grayscale — variance in center ROI should be > 0
        g = tile[:, :, 0].astype(np.float32) / 255.0
        h, w = g.shape
        roi = g[h//4:3*h//4, w//4:3*w//4]
        var = float(np.var(roi))
        assert var > 1e-5, f"G-only image appears uniform (variance={var:.6g})"

    def test_mode_variants_produce_different_outputs(self):
        width = height = 128
        modes = ["ndf", "g", "dfg", "spec"]
        tiles = {
            m: f3d.render_brdf_tile("ggx", 0.5, width, height, mode=m) for m in modes
        }
        # Ensure pairwise differences from full BRDF
        full = f3d.render_brdf_tile("ggx", 0.5, width, height, mode="full")
        for m, tile in tiles.items():
            assert not np.array_equal(tile, full), f"Mode '{m}' unexpectedly equals full BRDF output"

    def test_invalid_mode_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            _ = f3d.render_brdf_tile("ggx", 0.5, 64, 64, mode="invalid_mode")
