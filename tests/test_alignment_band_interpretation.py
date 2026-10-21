import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from forge3d.alignment import resample_raster_to_grid, reproject_dem_to_target

@pytest.mark.parametrize("scale,offset", [(0.1,100),(1,0)])
@pytest.mark.parametrize("operation", ["resample","reproject"])
def test_preserve_band_scale_offset(tmp_path,scale,offset,operation):
    source, output = tmp_path/"source.tif", tmp_path/"aligned.tif"
    transform = from_origin(0,2,1,1)
    samples = np.array([[10,20],[-9999,30]],np.float32)
    with rasterio.open(source,"w",driver="GTiff",width=2,height=2,count=1,dtype="float32",nodata=-9999,crs="EPSG:3857",transform=transform) as dst:
        dst.write(samples,1); dst.scales=(scale,); dst.offsets=(offset,)
    if operation == "resample":
        result=resample_raster_to_grid(source,{"crs":"EPSG:3857","transform":transform,"width":2,"height":2},output_path=output)
    else:
        result=reproject_dem_to_target(source,"EPSG:3857",output_path=output,resampling="nearest")
    assert result["metadata"]["scales"] == [scale]
    assert result["metadata"]["offsets"] == [offset]
    with rasterio.open(output) as dst:
        assert dst.scales == (scale,) and dst.offsets == (offset,)
        np.testing.assert_array_equal(dst.read(1),samples)
        assert dst.read_masks(1)[1,0] == 0


def test_shifted_grid_preserves_physical_values(tmp_path):
    source, output = tmp_path/"shift-source.tif", tmp_path/"shifted.tif"
    samples = np.array([[10,20,30],[40,-9999,60]],np.float32)
    with rasterio.open(source,"w",driver="GTiff",width=3,height=2,count=1,dtype="float32",nodata=-9999,crs="EPSG:3857",transform=from_origin(0,2,1,1)) as dst:
        dst.write(samples,1)
        dst.scales=(0.1,)
        dst.offsets=(100.,)
    resample_raster_to_grid(source,{"crs":"EPSG:3857","transform":from_origin(1,2,1,1),"width":2,"height":2},output_path=output,resampling="nearest")
    with rasterio.open(output) as dst:
        raw=dst.read(1)
        valid=dst.read_masks(1)>0
        expected=samples[:,1:]
        np.testing.assert_array_equal(valid,expected!=-9999)
        np.testing.assert_allclose((raw*dst.scales[0]+dst.offsets[0])[valid],(expected*0.1+100)[valid])


@pytest.mark.parametrize("scale,offset",[(0.1,100.),(1.,0.)])
def test_reprojected_grid_preserves_physical_values(tmp_path,scale,offset):
    from rasterio.warp import transform as transform_points
    source,output=tmp_path/"geographic.tif",tmp_path/"projected.tif"
    samples=np.array([[10,20],[30,-9999]],np.float32)
    with rasterio.open(source,"w",driver="GTiff",width=2,height=2,count=1,dtype="float32",nodata=-9999,crs="EPSG:4326",transform=from_origin(0,2,1,1)) as dst:
        dst.write(samples,1)
        dst.scales=(scale,)
        dst.offsets=(offset,)
    reproject_dem_to_target(source,"EPSG:3857",output_path=output,resampling="nearest")
    with rasterio.open(source) as src,rasterio.open(output) as dst:
        assert dst.crs != src.crs
        values=dst.read(1)
        for row in range(dst.height):
            for col in range(dst.width):
                x,y=dst.xy(row,col)
                xs,ys=transform_points(dst.crs,src.crs,[x],[y])
                source_row,source_col=src.index(xs[0],ys[0])
                expected=samples[source_row,source_col]
                if expected == -9999:
                    assert dst.read_masks(1)[row,col] == 0
                else:
                    assert values[row,col]*dst.scales[0]+dst.offsets[0] == pytest.approx(expected*scale+offset)
