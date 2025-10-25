import numpy as np
from forge3d.colormaps import get, available

def test_core_available_min():
    keys = available()
    assert any(k.endswith("forge3d:viridis") for k in keys) or ("forge3d:viridis" in keys)

def test_monotonic_lightness():
    try:
        from colorspacious import cspace_convert
    except ImportError:
        import pytest
        pytest.skip("colorspacious not installed")

    cm = get("forge3d:viridis")
    rgb = cm.rgba[:, :3]
    # convert linear sRGB to sRGB before LAB conversion (colorspacious expects sRGB)
    def lin_to_srgb(x):
        a=0.055; return np.where(x<=0.0031308, x*12.92, (1+a)*np.power(x,1/2.4)-a)
    srgb = lin_to_srgb(rgb)
    lab = cspace_convert(srgb, "sRGB1", "CIELab")
    L = lab[:,0]
    # allow small numerical noise
    diffs = np.diff(L)
    assert (diffs >= -1e-2).all()

def test_hash_determinism():
    cm1 = get("forge3d:viridis")
    cm2 = get("forge3d:viridis")
    assert np.allclose(cm1.rgba, cm2.rgba)
