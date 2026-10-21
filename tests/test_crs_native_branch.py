import numpy as np
from forge3d import crs


def test_native_branch_without_pyproj(monkeypatch):
    monkeypatch.setattr(crs, "HAS_NATIVE_PROJ", True)
    monkeypatch.setattr(crs, "HAS_PYPROJ", False)
    monkeypatch.setattr(crs, "HAS_PYPROJ_LEGACY", False)
    xy = crs.transform_coords([[10., 0.]], "EPSG:4326", "EPSG:3857")
    np.testing.assert_allclose(xy, [[6378137 * np.deg2rad(10), 0]], atol=1e-8)
    np.testing.assert_allclose(crs.transform_coords([[0., 10.]], "EPSG:4326", "EPSG:3857", always_xy=False), xy)
