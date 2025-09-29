import numpy as np
import pytest

from forge3d.io import import_osm_buildings_from_geojson


@pytest.mark.geometry
@pytest.mark.io
def test_osm_buildings_from_geojson_featurecollection():
    # Minimal FeatureCollection with two polygons and explicit heights
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"height": 10},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [2.0, 0.0],
                            [2.0, 1.0],
                            [0.0, 1.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"height": 20},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [3.0, 0.0],
                            [4.0, 0.0],
                            [4.0, 2.0],
                            [3.0, 2.0],
                            [3.0, 0.0],
                        ]
                    ],
                },
            },
        ],
    }

    import json

    mesh = import_osm_buildings_from_geojson(json.dumps(gj), default_height=8.0)

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
    assert abs(hmax - 20.0) < 1e-3
