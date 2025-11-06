#!/usr/bin/env python3
"""
Milestone 3: sRGB encoding and looks acceptance tests

- Validate that mid-gray linear 0.18 encodes to ~0.46 sRGB (±0.02)
  by using roughness_visualize mode to output linear grayscale.
- Validate that SPEC-only images are not near-black by sampling a center ROI.
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


def _center_roi(img: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    ch = int(h * fraction)
    cw = int(w * fraction)
    y0 = h // 2 - ch // 2
    x0 = w // 2 - cw // 2
    return img[y0:y0 + ch, x0:x0 + cw]


@skip_if_no_forge3d
@skip_if_no_native
class TestM3sRGBEncoding:
    def test_midgray_linear_encodes_to_srgb(self):
        """
        Render with roughness_visualize=True and roughness=0.18 so shader outputs
        vec3(0.18) linearly. With an sRGB render target, the read-back should reflect
        sRGB-encoded value ~0.46 within ±0.02 tolerance.
        """
        width = height = 64
        roughness = 0.18
        tile = f3d.render_brdf_tile(
            "ggx", roughness, width, height,
            ndf_only=False, g_only=False, dfg_only=False,
            spec_only=False, roughness_visualize=True,
            exposure=1.0, light_intensity=0.8,
        )
        # Expect grayscale
        r = tile[:, :, 0].astype(np.float32) / 255.0
        g = tile[:, :, 1].astype(np.float32) / 255.0
        b = tile[:, :, 2].astype(np.float32) / 255.0
        assert np.allclose(r, g, atol=1/255 + 1e-3)
        assert np.allclose(g, b, atol=1/255 + 1e-3)
        roi = _center_roi(r, 0.5)
        mean = float(np.mean(roi))
        # sRGB encoding of 0.18 linear ~ 0.461 (within ±0.02)
        assert abs(mean - 0.46) <= 0.02, f"sRGB mid-gray mismatch: {mean:.3f}"


@skip_if_no_forge3d
@skip_if_no_native
class TestM3Looks:
    def test_spec_only_not_near_black(self):
        """
        SPEC-only images should not be near-black. Sample a center ROI and assert
        mean luminance above a conservative threshold.
        """
        width = height = 256
        tile = f3d.render_brdf_tile(
            "ggx", 0.5, width, height,
            ndf_only=False, g_only=False, dfg_only=False,
            spec_only=True, roughness_visualize=False,
            exposure=1.0, light_intensity=0.8,
        )
        # Compute luminance in [0,1]
        lum = (0.299 * tile[:, :, 0] + 0.587 * tile[:, :, 1] + 0.114 * tile[:, :, 2]).astype(np.float32) / 255.0
        roi = _center_roi(lum, 0.25)
        mean = float(np.mean(roi))
        # Threshold chosen to ensure visible highlight while being stable across GPUs
        assert mean > 0.10, f"SPEC-only center ROI too dark: mean={mean:.3f}"
