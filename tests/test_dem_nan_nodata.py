import numpy as np
import pytest
from forge3d.io import calculate_dem_stats, fill_nodata, load_dem_from_array

def test_nan_nodata_fill_statistics_and_domain():
    data = np.array([[1,2],[3,np.nan]], dtype=np.float32)
    assert calculate_dem_stats(data, np.nan)["count"] == 3
    assert load_dem_from_array(data, nodata=np.nan).domain == (1,3)
    filled = fill_nodata(data, np.nan)
    assert np.isfinite(filled).all()
    np.testing.assert_array_equal(filled[np.isfinite(data)], data[np.isfinite(data)])
    invalid = np.full((2,2), np.nan)
    filled = fill_nodata(invalid, np.nan)
    assert filled is not invalid
    np.testing.assert_array_equal(filled, invalid)


def test_all_nodata_file_loads_unchanged(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from forge3d.io import load_dem
    path = tmp_path / "empty.tif"
    data = np.full((2, 2), np.nan, np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="float32", nodata=np.nan, transform=from_origin(0,2,1,1)) as dst:
        dst.write(data, 1)
    np.testing.assert_array_equal(load_dem(path).data, data)
