import numpy as np
import forge3d as f3d
from forge3d.determinism import canonical_heightmap


def test_physical_terrain_mutation_cannot_cross_hit(tmp_path):
    heights=np.ascontiguousarray(canonical_heightmap(),dtype=np.float32)
    scene=f3d.MapScene(
        terrain=f3d.TerrainSource(data=heights,crs="EPSG:32610",metadata={"source_id":"terrain-mutation-regression"}),
        camera=f3d.OrbitCamera(target=(0.,0.,0.),distance=900.),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64,height=64,format="png",path=str(tmp_path/"frame.png")),
    )
    cache=tmp_path/"cache"
    scene.render(cache=cache)
    first=(tmp_path/"frame.png").read_bytes()
    assert scene.last_render_metadata["cache_hit"] is False
    heights[:]=0
    scene.render(cache=cache)
    changed=(tmp_path/"frame.png").read_bytes()
    assert scene.last_render_metadata["cache_hit"] is False
    assert first != changed
    scene.render(cache=cache)
    assert scene.last_render_metadata["cache_hit"] is True
    assert (tmp_path/"frame.png").read_bytes() == changed
