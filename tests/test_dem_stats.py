# T02-BEGIN:tests
import numpy as np
import pytest
import vulkan_forge as vf

@pytest.mark.skipif(not hasattr(vf.Renderer, "add_terrain"), reason="T0.1 not merged")
def test_height_stats_and_override():
    r = vf.Renderer(32, 32)
    h = np.linspace(-10.0, 50.0, 32*32, dtype=np.float32).reshape(32, 32)
    r.add_terrain(h, (1.0, 1.0), 1.0, "viridis")
    # Override OK
    r.set_height_range(-5.0, 40.0)
    # Invalid overrides raise
    with pytest.raises(ValueError):
        r.set_height_range(1.0, 1.0)   # equal
    with pytest.raises(ValueError):
        r.set_height_range(2.0, -3.0)  # min > max
# T02-END:tests