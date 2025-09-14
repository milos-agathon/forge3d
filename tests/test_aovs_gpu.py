# tests/test_aovs_gpu.py
# GPU AOVs smoke test: ensures keys, shapes, and dtypes when GPU path is available.
# This file exists to validate the AOV API surface and GPU skip behavior without being brittle.
# RELEVANT FILES:python/forge3d/path_tracing.py,docs/api/aovs.md,tests/test_aovs_cpu_equiv.py

import pytest
import numpy as np

import forge3d.path_tracing as pt
from forge3d import enumerate_adapters


def make_test_scene():
    # Simple sphere scene
    scene = [
        {"center": (0.0, 0.0, 0.0), "radius": 0.5, "albedo": (0.8, 0.3, 0.2)},
        {"center": (0.8, 0.1, -0.1), "radius": 0.3, "albedo": (0.2, 0.8, 0.3)},
    ]
    cam = {"pos": (0.0, 0.0, 1.5)}
    return scene, cam


@pytest.mark.skipif(
    (lambda: (enumerate_adapters() or []) == [])(),
    reason="No compatible GPU adapter available",
)
def test_gpu_aovs_shapes_and_dtypes():
    scene, cam = make_test_scene()
    out = pt.render_aovs(32, 32, scene, cam, aovs=(
        "albedo", "normal", "depth", "direct", "indirect", "emission", "visibility"
    ), seed=7, frames=1, use_gpu=True)

    assert isinstance(out, dict)
    for k in ("albedo", "normal", "direct", "indirect", "emission"):
        assert k in out
        assert out[k].dtype in (np.float32, np.float64)
        assert out[k].ndim == 3 and out[k].shape[0] == 32 and out[k].shape[1] == 32

    assert "depth" in out and out["depth"].ndim == 2 and out["depth"].shape == (32, 32)
    assert out["depth"].dtype in (np.float32, np.float64)

    assert "visibility" in out and out["visibility"].shape == (32, 32)
    assert out["visibility"].dtype == np.uint8

