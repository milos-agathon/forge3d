"""The mapmaker's complete shared-recipe entry point, with synthetic inputs."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

import forge3d as f3d


@pytest.mark.parametrize("samples,hdr,denoiser", ((1, False, "none"), (4, True, "atrous")),
                         ids=("preview", "final-hdr"))
def test_shared_recipe_script_outputs_scene_still_frames_and_meaning(tmp_path, samples, hdr, denoiser):
    transform = rasterio.transform.from_origin(500000, 4600000, 2, 2)
    profile = {"driver":"GTiff", "width":16, "height":16, "count":1,
               "dtype":"float32", "crs":"EPSG:32632", "transform":transform}
    for name, data in (("dem",np.zeros((16,16),np.float32)), ("classes",np.ones((16,16),np.float32))):
        with rasterio.open(tmp_path / f"{name}.tif", "w", **profile) as dst:
            dst.write(data,1)
    features = [{"id":"fixture", "type":"Feature", "properties":{"class_id":1,"height":8},
        "geometry":{"type":"Polygon", "coordinates":[[[500005,4599995],[500015,4599995],
        [500015,4599985],[500005,4599985],[500005,4599995]]]}}]
    for name in ("vectors","buildings"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"type":"FeatureCollection","features":features}))
    palette = f3d.CategoricalRasterStyle({1:(80,140,50,255)}, {1:"Class one"})
    material = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer("base",(170,125,80)),), categorical=palette)
    material.save(tmp_path / "recipe.json")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("material_workflow",root / "examples/terrain_material_recipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(dem=str(tmp_path/"dem.tif"),classes=str(tmp_path/"classes.tif"),
        recipe=str(tmp_path/"recipe.json"),source_credit="synthetic fixtures",window=None,output=str(tmp_path/"output"),
        size=[384,256],frames=2,fps=24,azimuth=[140,160],elevation=55,fov=45,camera_distance=1.2,
        exaggeration=1,movie=False, vectors=str(tmp_path/"vectors.json"),buildings=str(tmp_path/"buildings.json"),
        feature_crs="EPSG:32632",feature_recipe=None,class_field="class_id",feature_field=None,
        samples=samples,hdr=hdr,denoiser=denoiser,elevation_units="m",vertical_datum=None,
        sun_direction=(.8, .6, 0),title=None)
    with pytest.raises(ValueError, match="DEM elevation units"):
        module.run(argparse.Namespace(**{**vars(args), "elevation_units": None}))
    record = module.run(args)
    assert record["backend"] == "gpu_terrain"
    assert record["used_quality"]["samples_used"] == samples
    assert record["used_quality"]["denoiser_used"] == denoiser
    assert record["camera"]["native_polar_angle"] == 35
    assert record["camera"]["mode"] == "mesh:zup:north"
    assert set(record["feature_bindings"]) == {"material-vectors","material-buildings"}
    output = tmp_path/"output"
    for name in ("material.png","material.png.json","recipe.json","still.png","evidence.json","scene.forge3d/manifest.json"):
        assert (output/name).is_file()
    assert len(list((output/"frames").glob("frame-*.png"))) == 2
    assert json.loads((output/"material.png.json").read_text())["legend"] == palette.legend()
