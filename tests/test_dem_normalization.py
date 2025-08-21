# T02-BEGIN:tests-dem
import numpy as np
import forge3d as f3d

def _make_plane(h, w):
    x = np.linspace(-1, 1, w, dtype=np.float32)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    return 0.25*np.sin(1.3*X) + 0.25*np.cos(1.1*Y)

def test_dem_stats_minmax():
    Z = _make_plane(32, 48)
    mn, mx, mean, std = f3d.dem_stats(Z)
    assert mx > mn
    assert std > 0.0

def test_dem_normalize_minmax_shape_dtype():
    Z = _make_plane(16, 20)
    N = f3d.dem_normalize(Z, mode="minmax", out_range=(0.0, 1.0))
    assert N.shape == Z.shape and N.dtype == np.float32
    assert N.min() >= -1e-5 and N.max() <= 1.0 + 1e-5

def test_renderer_terrain_stats_and_normalize():
    r = f3d.Renderer(64, 64)
    Z = _make_plane(64, 64)
    r.add_terrain(Z, spacing=(1.0,1.0), exaggeration=1.0, colormap="viridis")
    mn, mx, mean, std = r.terrain_stats()
    assert mx > mn and std > 0.0
    r.normalize_terrain("minmax", range=(0.0,1.0))
    mn2, mx2, _, _ = r.terrain_stats()
    assert mn2 >= -1e-5 and mx2 <= 1.0 + 1e-5
# T02-END:tests-dem