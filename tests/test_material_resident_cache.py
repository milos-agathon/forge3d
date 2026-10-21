import numpy as np
import forge3d as f3d
from forge3d.determinism import _canonical_params_config, canonical_heightmap, write_canonical_hdr


def resources(tmp_path):
    hdr=tmp_path/"env.hdr"
    write_canonical_hdr(str(hdr))
    renderer=f3d.TerrainRenderer(f3d.Session(window=False))
    return renderer, f3d.IBL.from_hdr(str(hdr),intensity=1.), f3d.TerrainRenderParams(_canonical_params_config()(64,64)), np.ascontiguousarray(canonical_heightmap(),dtype=np.float32)


def test_cache_tracks_resident_material_bytes_and_dimensions(tmp_path,monkeypatch):
    from PIL import Image
    textures=tmp_path/"textures"
    textures.mkdir()
    monkeypatch.setenv("FORGE3D_MATERIAL_DIR",str(textures))
    def write(color,size):
        for name in ("rock","grass","dirt","snow"):
            Image.new("RGBA",(size,size),color).save(textures/(name+"_albedo.png"))
    write((255,0,0,255),4)
    renderer,ibl,params,height=resources(tmp_path)
    old=f3d.MaterialSet.terrain_default()
    cache=tmp_path/"cache"
    def render(material, cache_path):
        frame=renderer.render_terrain_pbr_pom(material,ibl,params,height,cache=cache_path)
        return frame.to_numpy().copy(),dict(renderer.last_anamnesis_cache_report)
    original,_=render(old,cache)
    write((0,0,255,255),4)
    reused,report=render(old,cache)
    np.testing.assert_array_equal(reused,original)
    assert "terrain.forward" in report["hits"]
    new=f3d.MaterialSet.terrain_default()
    changed,report=render(new,cache)
    assert "terrain.forward" in report["misses"]
    assert not np.array_equal(changed,original)
    uncached,_=render(new,None)
    np.testing.assert_array_equal(changed,uncached)
    write((0,0,255,255),8)
    _,report=render(f3d.MaterialSet.terrain_default(),cache)
    assert "terrain.forward" in report["misses"]
