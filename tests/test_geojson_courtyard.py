import json
import numpy as np
import pytest
from forge3d import _forge3d as native


def mesh(kind, coordinates):
    return native.import_osm_buildings_from_geojson_py(json.dumps({"type":"FeatureCollection", "features":[{"type":"Feature", "properties":{}, "geometry":{"type":kind,"coordinates":coordinates}}]}), 3.)


def test_holes_and_multipolygon():
    rings = [[[0,0],[10,0],[10,10],[0,10],[0,0]], [[3,3],[3,7],[7,7],[7,3],[3,3]]]
    for kind, coordinates, area in [("Polygon", rings,84), ("MultiPolygon", [rings, [[[x+20,y] for x,y in ring] for ring in rings]],168)]:
        result = mesh(kind, coordinates)
        vertices=np.asarray(result["positions"]).reshape(-1,3)
        triangles=vertices[np.asarray(result["indices"]).reshape(-1,3)]
        roof=triangles[np.all(triangles[:,:,1]==3,axis=1)]
        actual=np.linalg.norm(np.cross(roof[:,1]-roof[:,0],roof[:,2]-roof[:,0]),axis=1).sum()/2
        assert actual == pytest.approx(area)


def test_malformed_coordinate_is_error():
    with pytest.raises(ValueError):
        mesh("Polygon", [[[0,0],[10,"bad"],[0,10],[0,0]]])
