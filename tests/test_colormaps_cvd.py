import numpy as np
from forge3d.colormaps import get

def simulate_cvd(rgb, deficiency="deuteranomaly"):
    try:
        from colorspacious import cspace_convert
    except ImportError:
        import pytest
        pytest.skip("colorspacious not installed")
    return cspace_convert(rgb, "sRGB1", ("sRGB1+CVD", deficiency, 100))

def test_vik_diverging_zero_contrast():
    cm = get("forge3d:vik")
    rgb_lin = cm.rgba[:, :3]
    def lin_to_srgb(x):
        a=0.055; return np.where(x<=0.0031308, x*12.92, (1+a)*np.power(x,1/2.4)-a)
    srgb = lin_to_srgb(rgb_lin)
    cvd = simulate_cvd(srgb)
    # ensure endpoints remain distinguishable
    assert np.linalg.norm(cvd[0]-cvd[-1]) > 0.2
