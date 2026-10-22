import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from forge3d.io import load_dem

@pytest.mark.parametrize("options", [{"normalize":True},{"target_domain":(10,20)},{"normalize":True,"target_domain":(10,20)}])
def test_transforms_preserve_nodata(tmp_path, options):
    path = tmp_path / "dem.tif"
    with rasterio.open(path,"w",driver="GTiff",width=2,height=2,count=1,dtype="float32",nodata=-9999,crs="EPSG:3857",transform=from_origin(0,2,1,1)) as dst:
        dst.write(np.array([[1,2],[-9999,3]],np.float32),1)
    dem = load_dem(path,fill_nodata_values=False,**options)
    assert dem.data[1,0] == dem.nodata_value == -9999
    assert dem.stats["count"] == 3


def test_integer_normalization_keeps_fractions(tmp_path):
    path = tmp_path / "integer.tif"
    with rasterio.open(path, "w", driver="GTiff", width=3, height=1, count=1,
                       dtype="int16", crs="EPSG:3857", transform=from_origin(0,1,1,1)) as dst:
        dst.write(np.array([[1, 2, 3]], np.int16), 1)
    result = load_dem(path, dtype="int16", normalize=True)
    assert result.data.dtype == np.float64
    np.testing.assert_array_equal(result.data, [[0., 0.5, 1.]])
