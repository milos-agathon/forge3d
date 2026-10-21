import pytest
from forge3d.scale_bar import ScaleBar
from forge3d.map_plate import BBox

def test_projected_metres_and_geographic_degrees():
    assert ScaleBar.compute_meters_per_pixel(BBox(0,0,1000,1000,crs="EPSG:3857"),1000,geodesic=False) == 1
    assert ScaleBar.compute_meters_per_pixel(BBox(0,-1,1,1),1000,geodesic=False) == pytest.approx(111.32)
