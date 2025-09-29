import numpy as np
import pytest

from forge3d.io import import_osm_buildings_extrude


@pytest.mark.geometry
@pytest.mark.io
def test_osm_buildings_extrude_merges_meshes_and_respects_heights():
    # Two simple rectangles at different heights
    rect1 = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float32)
    rect2 = np.array([[3, 0], [4, 0], [4, 2], [3, 2]], dtype=np.float32)

    features = [
        {"coords": rect1, "height": 10.0},
        {"coords": rect2, "height": 15.0},
    ]

    mesh = import_osm_buildings_extrude(features, default_height=8.0)

    assert mesh.positions.shape[0] > 0
    assert mesh.indices.shape[0] > 0

    # Height spans from ~0 to ~max(height); tolerate either Z-up or Y-up
    zmin = float(mesh.positions[:, 2].min())
    zmax = float(mesh.positions[:, 2].max())
    ymin = float(mesh.positions[:, 1].min())
    ymax = float(mesh.positions[:, 1].max())
    hmin = min(zmin, ymin)
    hmax = max(zmax, ymax)
    assert hmin >= -1e-4
    assert abs(hmax - 15.0) < 1e-3

    # Normals and UVs are vertex-aligned
    assert mesh.normals.shape[0] in (0, mesh.positions.shape[0])
    assert mesh.uvs.shape[0] in (0, mesh.positions.shape[0])
