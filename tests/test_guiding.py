# tests/test_guiding.py
# Minimal contract tests for A13 path guiding utilities (Python-side).
# This exists to validate deterministic PDFs and basic accumulation behavior.
# RELEVANT FILES:python/forge3d/guiding.py,python/forge3d/__init__.py,src/path_tracing/guiding.rs,docs/api/guiding.md

import numpy as np
import pytest

from forge3d import OnlineGuidingGrid


def test_guiding_dims_and_uniform_pdf():
    g = OnlineGuidingGrid(8, 6, 8)
    assert g.dims() == (8, 6, 8)
    p = g.pdf(3, 2)
    assert p.shape == (8,)
    assert p.dtype == np.float32
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(), 1.0, atol=1e-6)
    assert np.allclose(p, np.full((8,), 1.0 / 8.0, dtype=np.float32))


def test_guiding_updates_bias_pdf():
    g = OnlineGuidingGrid(2, 2, 4)
    for _ in range(16):
        g.update(0, 0, 1, 1.0)
    p = g.pdf(0, 0)
    assert p[1] > 0.5
    assert np.allclose(p.sum(), 1.0, atol=1e-6)

