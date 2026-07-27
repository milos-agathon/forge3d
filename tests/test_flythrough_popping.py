from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _deltae import delta_e_2000, srgb_to_lab
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.diagnostics import seam_stats
from test_terrain_clipmap_streaming import _make_params, _render_rgba, _steep_dem


@pytest.mark.gpu_lane
@pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)
def test_600_frame_streaming_flythrough_has_no_pop_or_crack():
    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        dem = _steep_dem(256)
        overview = dem[::4, ::4].copy()
        renderer.enable_height_streaming(
            terrain_extent_m=100_000.0,
            ring_count=4,
            ring_resolution=32,
            lod=2,
            tile_resolution=64,
            max_in_flight=16,
            pool_size=2,
            dem=dem,
            coarse_prefill=True,
            max_resident_bytes=8 * 1024 * 1024,
        )
        params = _make_params(size_px=(96, 54))
        previous = None
        max_delta_e = 0.0
        path = np.linspace(-35_000.0, 35_000.0, 600)
        for x in path:
            renderer.stream_height_tiles((float(x), 500.0, 0.0), max_uploads=8)
            current = _render_rgba(renderer, params, overview, ibl)
            if previous is not None:
                delta = delta_e_2000(
                    srgb_to_lab(previous[..., :3]), srgb_to_lab(current[..., :3])
                )
                max_delta_e = max(max_delta_e, float(delta.max()))
            previous = current

    seams = seam_stats()
    assert max_delta_e < 1.0, max_delta_e
    assert seams["crack_count"] == 0, seams
    assert seams["seams_valid"] is True, seams
