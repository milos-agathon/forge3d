import pytest
from forge3d.scale_bar import ScaleBar, ScaleBarConfig
from forge3d.map_plate import BBox

def test_projected_metres_and_geographic_degrees():
    assert ScaleBar.compute_meters_per_pixel(BBox(0,0,1000,1000,crs="EPSG:3857"),1000,geodesic=False) == 1
    assert ScaleBar.compute_meters_per_pixel(BBox(0,-1,1,1),1000,geodesic=False) == pytest.approx(111.32)


@pytest.mark.parametrize("mpp", [0.06, 0.3, 1.0, 10.0, 100.0])
def test_rendered_scale_label_preserves_short_distances(mpp):
    import numpy as np
    bar = ScaleBar(mpp, ScaleBarConfig())
    image, label, _ = bar.render_geometry()
    border = np.flatnonzero(np.all(image[bar.config.padding, :, :3] == 0, axis=1))
    number, unit = label.split()
    metres = float(number) * {"m": 1, "km": 1000}[unit]
    assert metres > 0
    assert abs(metres - (border[-1] - border[0] + 1) * mpp) <= mpp
