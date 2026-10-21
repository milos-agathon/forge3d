import numpy as np
import rasterio
from rasterio.transform import from_origin
from forge3d.cog import CogDatasetFallback

def test_edge_tile_is_padded_without_stretching(tmp_path):
    path=tmp_path/"edge.tif"
    data=np.broadcast_to(np.arange(300,dtype=np.float32),(300,300))
    with rasterio.open(path,"w",driver="GTiff",width=300,height=300,count=1,dtype="float32",nodata=-9999,transform=from_origin(0,300,1,1)) as dst: dst.write(data,1)
    with CogDatasetFallback(str(path)) as dataset:
        edge=dataset.read_tile(1,0)
        np.testing.assert_array_equal(edge[:,:44],data[:256,256:])
        assert np.all(edge[:,44:] == -9999)
        np.testing.assert_array_equal(dataset.read_tile(0,0),data[:256,:256])
