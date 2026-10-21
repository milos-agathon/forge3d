import numpy as np
from forge3d.tiles3d import BoundingVolume, Tiles3dRenderer

def test_equatorial_region_matches_cartesian_sse():
    region = BoundingVolume("region", [-0.001, -0.001, 0.001, 0.001, 0, 0])
    np.testing.assert_allclose(region.center(), [6378137, 0, 0], atol=1e-8)
    assert region.radius() > 6000
    box = BoundingVolume("box", [6378137., 0., 0., 0., 6378.137, 0., 0., 0., 6378.137, 0., 0., 0.])
    renderer = Tiles3dRenderer()
    camera = (6400000, 0, 0)
    np.testing.assert_allclose(renderer.compute_sse(100, region, camera), renderer.compute_sse(100, box, camera), rtol=1e-12)

def test_dateline_region():
    region = BoundingVolume("region", [np.deg2rad(179), 0, np.deg2rad(-179), 0, 0, 0])
    np.testing.assert_allclose(region.center(), [-6378137, 0, 0], atol=1e-8)
    np.testing.assert_allclose(region.radius(), np.deg2rad(1) * 6378137)
