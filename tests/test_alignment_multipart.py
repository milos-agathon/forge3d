import pytest
from forge3d.alignment import transform_geometry

@pytest.mark.parametrize("kind,coordinates,part", [("MultiPoint",[[10,45],[11,46]],"Point"),("MultiLineString",[[[10,45],[11,46]]],"LineString"),("MultiPolygon",[[[[10,45],[11,45],[10,46],[10,45]]]],"Polygon")])
def test_multipart_matches_individual_parts(kind,coordinates,part):
    args=("EPSG:4326","EPSG:3857")
    result=transform_geometry({"type":kind,"coordinates":coordinates},*args)
    assert result["coordinates"] == [transform_geometry({"type":part,"coordinates":c},*args)["coordinates"] for c in coordinates]
    collection=transform_geometry({"type":"GeometryCollection","geometries":[{"type":kind,"coordinates":coordinates}]},*args)
    assert collection["geometries"] == [result]

@pytest.mark.parametrize("target", ["EPSG:4326","EPSG:3857"])
def test_unknown_geometry_rejected(target):
    with pytest.raises(ValueError,match="Unsupported geometry"):
        transform_geometry({"type":"Unknown"},"EPSG:4326",target)
