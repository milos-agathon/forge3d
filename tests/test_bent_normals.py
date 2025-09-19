#!/usr/bin/env python3
"""P5: Bent normals validation for offline AO/G-buffer path.

Validates shape, normalization, and correlation with AO (higher AO -> more bending magnitude).
"""
from __future__ import annotations

import numpy as np

def test_bent_normals_shape_and_norm():
    from forge3d.path_tracing import render_aovs
    w, h = 128, 96
    aovs = render_aovs(w, h, scene=None, camera=None, aovs=("ao", "bent"), seed=7)
    assert "bent" in aovs and "ao" in aovs
    bent = aovs["bent"]
    ao = aovs["ao"]
    assert bent.shape == (h, w, 3)
    assert ao.shape == (h, w)
    # Norm near 1 for most pixels
    n = np.linalg.norm(bent.reshape(-1, 3), axis=1)
    assert np.percentile(n, 5) > 0.8
    assert np.percentile(n, 95) <= 1.1

def test_bent_normals_correlate_with_ao():
    from forge3d.path_tracing import render_aovs
    w, h = 128, 96
    aovs = render_aovs(w, h, scene=None, camera=None, aovs=("ao", "bent"), seed=13)
    bent = aovs["bent"].reshape(-1, 3)
    ao = aovs["ao"].reshape(-1)
    # Higher AO -> greater deviation from Z axis nominal direction
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cos_to_z = (bent @ z)
    # Split into bins by AO and compare mean cos_to_z
    lo = cos_to_z[ao < 0.5].mean()
    hi = cos_to_z[ao >= 0.5].mean()
    # Expect more bend (smaller cosine) for higher AO
    assert hi <= lo + 1e-3
