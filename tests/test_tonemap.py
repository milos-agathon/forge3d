# T22-BEGIN:test
import math
import numpy as np
import pytest
import vulkan_forge as vf

def reinhard(x):
    return x / (1.0 + x)

def gamma_correct(x, gamma=2.2):
    return np.maximum(x, 0.0) ** (1.0/gamma)

def tonemap_cpu(rgb, exposure=1.0):
    return gamma_correct(reinhard(rgb * exposure), 2.2)

def test_tonemap_cpu_vector():
    rgb = np.array([0.0, 0.18, 4.0], dtype=np.float32) # black, mid-gray, bright
    out = tonemap_cpu(rgb, 1.0)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    # Known spot checks
    assert np.isclose(out[0], 0.0, atol=1e-6)
    assert 0.4 < out[1] < 0.6
    assert out[2] < 1.0

def test_set_sun_and_exposure():
    r = vf.Renderer(16, 16)
    # Should not throw
    r.set_sun(45.0, 30.0)
    with pytest.raises(ValueError): r.set_exposure(0.0)
    r.set_exposure(1.25)
# T22-END:test