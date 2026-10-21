import numpy as np
import forge3d as f3d
from forge3d.determinism import _canonical_params_config, canonical_heightmap, write_canonical_hdr


def resources(tmp_path):
    hdr=tmp_path/"env.hdr"
    write_canonical_hdr(str(hdr))
    renderer=f3d.TerrainRenderer(f3d.Session(window=False))
    return renderer, f3d.IBL.from_hdr(str(hdr),intensity=1.), f3d.TerrainRenderParams(_canonical_params_config()(64,64)), np.ascontiguousarray(canonical_heightmap(),dtype=np.float32)


def test_warm_cache_recomputes_depth_and_preserves_pixels(tmp_path):
    renderer,ibl,params,height=resources(tmp_path)
    material=f3d.MaterialSet.terrain_default()
    images=[]
    reports=[]
    for cache in (None,tmp_path/"cache",tmp_path/"cache"):
        images.append(renderer.render_terrain_pbr_pom(material,ibl,params,height,cache=cache).to_numpy().copy())
        reports.append(dict(renderer.last_anamnesis_cache_report))
    np.testing.assert_array_equal(images[0],images[1])
    np.testing.assert_array_equal(images[0],images[2])
    assert "terrain.shadow" in reports[2]["misses"]
    assert "terrain.forward" in reports[2]["hits"]
