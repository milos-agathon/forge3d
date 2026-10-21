import numpy as np
import forge3d as f3d
from forge3d.determinism import _canonical_params_config, canonical_heightmap, write_canonical_hdr


def resources(tmp_path):
    hdr=tmp_path/"env.hdr"
    write_canonical_hdr(str(hdr))
    renderer=f3d.TerrainRenderer(f3d.Session(window=False))
    return renderer, f3d.IBL.from_hdr(str(hdr),intensity=1.), f3d.TerrainRenderParams(_canonical_params_config()(64,64)), np.ascontiguousarray(canonical_heightmap(),dtype=np.float32)


def test_hdr_frame_after_fallback_initialization(tmp_path):
    renderer,ibl,params,height=resources(tmp_path)
    renderer.begin_offline_accumulation(material_set=f3d.MaterialSet.terrain_default(),env_maps=ibl,params=params,heightmap=height,water_mask=None)
    try:
        assert renderer.accumulate_batch(1)["total_samples"] == 1
        hdr,_=renderer.resolve_offline_hdr()
        pixels=hdr.to_numpy_f32()
        assert np.isfinite(pixels).all()
        assert np.any(pixels[...,:3]>0)
    finally:
        renderer.end_offline_accumulation()


def test_physical_hdr_cold_and_warm_sequence_cache(tmp_path):
    from forge3d.anamnesis import render_sequence
    renderer, ibl, params, height = resources(tmp_path)
    executions = []
    def render_hdr(recipe, frame):
        executions.append(frame)
        renderer.begin_offline_accumulation(material_set=f3d.MaterialSet.terrain_default(), env_maps=ibl,
            params=params, heightmap=height, water_mask=None)
        try:
            renderer.accumulate_batch(1)
            hdr, _ = renderer.resolve_offline_hdr()
            pixels = hdr.to_numpy_f32()
            assert np.isfinite(pixels).all()
            assert np.any(pixels[..., :3] > 0)
            return pixels.tobytes()
        finally:
            renderer.end_offline_accumulation()
    kwargs = dict(frames=[0], cache=tmp_path / "hdr-cache", render_frame=render_hdr,
                  render_frame_fingerprint=b"physical-offline-hdr-v1", render_frame_context=height.tobytes())
    cold = render_sequence({}, **kwargs)
    warm = render_sequence({}, **kwargs)
    assert executions == [0]
    assert warm.frame_blobs == cold.frame_blobs
    assert warm.cache_report.hits
    pixels = np.frombuffer(warm.frame_blobs[0], dtype=np.float32)
    assert np.isfinite(pixels).all() and np.any(pixels > 0)
