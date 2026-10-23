from anamnesis_packed_store import metadata_entries, remove_entry
import json
import shutil

import numpy as np
import pytest

import forge3d as f3d
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)


def resources(tmp_path):
    hdr = tmp_path / "env.hdr"
    write_canonical_hdr(str(hdr))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    return (
        renderer,
        f3d.IBL.from_hdr(str(hdr), intensity=1.0),
        f3d.TerrainRenderParams(_canonical_params_config()(64, 64)),
        np.ascontiguousarray(canonical_heightmap(), dtype=np.float32),
    )


@pytest.mark.parametrize("evicted", [
    ("terrain.forward",),
    ("terrain.resolve",),
    ("terrain.forward", "terrain.resolve"),
])
def test_warm_cache_materializes_depth_for_live_consumers_and_preserves_pixels(tmp_path, evicted):
    renderer, ibl, params, height = resources(tmp_path)
    material = f3d.MaterialSet.terrain_default()
    cache = tmp_path / "cache"

    def render(cache_path):
        frame = renderer.render_terrain_pbr_pom(material, ibl, params, height, cache=cache_path)
        return frame.to_numpy().copy(), dict(renderer.last_anamnesis_cache_report)

    fresh, _ = render(None)
    cold, report = render(cache)
    np.testing.assert_array_equal(fresh, cold)
    assert {"terrain.shadow", "terrain.forward", "terrain.resolve"} <= set(report["misses"])

    warm, report = render(cache)
    np.testing.assert_array_equal(fresh, warm)
    assert not report["misses"]
    assert {"terrain.shadow", "terrain.forward", "terrain.resolve"} <= set(report["hits"])

    # Force real execution with the same shadow key. Dropping forward and
    # resolve together also exposes the newly shaded pixels to comparison.
    entries = {meta["pass_label"]: key for key, meta in metadata_entries(cache).items()}
    assert "terrain.shadow" not in entries  # Depth is never stored or restored.
    for label in evicted:
        remove_entry(cache, entries[label])
    partial, report = render(cache)
    np.testing.assert_array_equal(fresh, partial)
    assert set(evicted) <= set(report["misses"])
    if "terrain.forward" in evicted:
        # The forward consumer needs live depth even if resolve can restore.
        assert "terrain.shadow" in report["misses"]
        assert "terrain.shadow" not in report["hits"]
    else:
        # Resolve reads the restored beauty texture, not the shadow map.
        assert "terrain.forward" in report["hits"]
        assert "terrain.shadow" in report["hits"]
        assert "terrain.shadow" not in report["misses"]
