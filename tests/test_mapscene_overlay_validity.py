import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from forge3d.map_scene import RasterOverlay, _load_native_raster_overlay

@pytest.mark.parametrize("bands", [1,3,4])
@pytest.mark.parametrize("mask_kind", ["nodata","internal"])
def test_validity_and_explicit_alpha(tmp_path,bands,mask_kind):
    path=tmp_path/"overlay.tif"
    data=np.full((bands,2,2),100,dtype=np.uint8)
    if bands == 4: data[3]=128
    if mask_kind == "nodata": data[:,0,0]=0
    transform=from_origin(0,2,1,1)
    with rasterio.open(path,"w",driver="GTiff",width=2,height=2,count=bands,dtype="uint8",crs="EPSG:3857",transform=transform,nodata=0 if mask_kind=="nodata" else None) as dst:
        dst.write(data)
        if mask_kind=="internal": dst.write_mask(np.array([[0,255],[255,255]],np.uint8))
    layer=RasterOverlay("overlay",path=path)
    for grid in (None,{"crs":"EPSG:3857","transform":transform,"width":3,"height":2}):
        rgba=_load_native_raster_overlay(layer,target_grid=grid)
        assert rgba[0,0,3]==0
        assert rgba[1,1,3]==(128 if bands==4 else 255)
        np.testing.assert_array_equal(rgba[1,1,:3],[100,100,100])
        if grid is not None: assert np.all(rgba[:,2,3]==0)


def test_production_drape_ignores_masked_nan(tmp_path):
    from forge3d.map_scene import TerrainSource, SceneRecipe, OrbitCamera, LightingPreset
    from forge3d._map_scene_rasters import _overlay_in_terrain_uv

    path = tmp_path / "nan-overlay.tif"
    transform = from_origin(0, 2, 1, 1)
    samples = np.array([[np.nan, 100], [100, 100]], dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="float32", crs="EPSG:3857", transform=transform, nodata=np.nan) as dst:
        dst.write(samples, 1)
    terrain = TerrainSource(data=np.zeros((2, 2), np.float32), crs="EPSG:3857",
        metadata={"geotransform": list(transform.to_gdal()), "width": 2, "height": 2})
    recipe = SceneRecipe(terrain=terrain, camera=OrbitCamera(), lighting=LightingPreset(), target_crs="EPSG:3857")
    rgba = _overlay_in_terrain_uv(RasterOverlay("overlay", path=path), recipe, (2, 2))
    assert rgba[0, 0, 3] == 0
    np.testing.assert_array_equal(rgba[1, 1], [100, 100, 100, 255])
