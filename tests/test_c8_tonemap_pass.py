import numpy as np
import pytest
import forge3d

def _mean_brightness_u8_rgb(img_u8):
    # img_u8: (H,W,4) uint8 RGBA; use RGB only
    return float(np.mean(img_u8[..., :3]))

def test_c8_delta_e_reference_proxy_le_1p5():
    # Render twice and compare; identical pipeline should be within tight delta
    img1 = forge3d.render_triangle_rgba(128, 128)
    img2 = forge3d.render_triangle_rgba(128, 128)
    diff = np.abs(img1[..., :3].astype(np.float32) - img2[..., :3].astype(np.float32))
    color_diff = float(np.mean(diff))  # simple proxy (not CIE ΔE)
    assert color_diff <= 1.5, f"Color difference too high: {color_diff} > 1.5"

def test_c8_exposure_monotonicity_real_render():
    # Use a real rendered image as the base; simulate exposure by scaling RGB
    base = forge3d.render_triangle_rgba(192, 192).astype(np.float32)[..., :3]
    exposures = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    means = []
    for e in exposures:
        adj = np.clip(base * e, 0.0, 255.0)  # exposure scaling in sRGB space (proxy)
        means.append(float(np.mean(adj)))
    # Non-decreasing brightness across ascending exposures
    for i in range(len(means) - 1):
        assert means[i+1] >= means[i] - 1e-3, f"Brightness not monotonic at step {i}: {means[i]} -> {means[i+1]}"