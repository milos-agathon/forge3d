# tests/test_aovs_cpu_equiv.py
# CPU/GPU determinism test: compares simple statistics when GPU is available.
# This file exists to ensure CPU/GPU parity or skip when no adapter is present.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_aovs_gpu.py,docs/api/aovs.md

import pytest
import numpy as np

import forge3d.path_tracing as pt

try:
    from forge3d import enumerate_adapters
except Exception:  # extension not loaded
    def enumerate_adapters():
        return []


def make_test_scene():
    scene = [
        {"center": (0.0, 0.0, 0.0), "radius": 0.5, "albedo": (0.7, 0.6, 0.2)},
    ]
    cam = {"pos": (0.0, 0.0, 1.5)}
    return scene, cam


@pytest.mark.skipif((enumerate_adapters() or []) == [], reason="No GPU adapter; skipping CPU/GPU parity test")
def test_cpu_gpu_stats_match_or_skip():
    tr_scene, cam = make_test_scene()
    a_gpu = pt.render_aovs(32, 32, tr_scene, cam, aovs=("albedo", "depth"), seed=7, frames=1, use_gpu=True)
    a_cpu = pt.render_aovs(32, 32, tr_scene, cam, aovs=("albedo", "depth"), seed=7, frames=1, use_gpu=False)
    for k in a_gpu:
        g, c = a_gpu[k], a_cpu[k]
        if g.ndim == 3:
            axes = (0, 1)
        else:
            axes = None
        gm = np.nanmean(g, axis=axes)
        cm = np.nanmean(c, axis=axes)
        gv = np.nanvar(g, axis=axes)
        cv = np.nanvar(c, axis=axes)
        assert np.allclose(gm, cm, rtol=5e-2, atol=5e-2)
        assert np.allclose(gv, cv, rtol=1e-1, atol=1e-1)

