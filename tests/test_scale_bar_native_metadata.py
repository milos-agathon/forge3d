import sys
import pytest
from forge3d.scale_bar import ScaleBar
from forge3d.map_plate import BBox
from forge3d import gis

@pytest.mark.parametrize("crs,expected", [("EPSG:3857", 1.), ("EPSG:32633", 1.), ("EPSG:4326", 111.32)])
def test_scale_bar_without_pyproj(monkeypatch, crs, expected):
    monkeypatch.setitem(sys.modules, "pyproj", None)
    width = 1 if crs == "EPSG:4326" else 1000
    assert ScaleBar.compute_meters_per_pixel(BBox(0, -1, width, 1, crs=crs), 1000, geodesic=False) == pytest.approx(expected)
    info = gis._crs_metadata(crs)
    assert info["axis_order"] == ("yx" if crs == "EPSG:4326" else "xy")


def test_unknown_native_authority_order_is_rejected():
    with pytest.raises((ValueError, RuntimeError), match="authority axis order"):
        gis.create_crs_transformer("EPSG:4269", "EPSG:4269", always_xy=False)


def test_native_factory_authority_axis_order():
    result = gis.create_crs_transformer("EPSG:4326", "EPSG:3857", always_xy=False).transform_point(0, 10)
    assert result == pytest.approx((1113194.9079327357, 0), abs=1e-8)


@pytest.mark.parametrize("geodesic", [True, False])
def test_unknown_scale_bar_crs_fails_without_pyproj(monkeypatch, geodesic):
    monkeypatch.setitem(sys.modules, "pyproj", None)
    with pytest.raises(ValueError, match="Unknown CRS"):
        ScaleBar.compute_meters_per_pixel(BBox(0, 0, 1, 1, crs="EPSG:4269"), 100, geodesic=geodesic)
