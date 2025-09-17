#!/usr/bin/env python3
"""A23: Tests for Hair BSDF (Kajiya-Kay) and curve primitives

Covers:
- Highlight/tilt behavior under changes in view/light relative to the hair tangent
- Basic sanity (finite outputs)
"""

import numpy as np

from forge3d.hair import HairBSDF


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / (n + 1e-8)).astype(np.float32)


def test_highlight_tilt_behavior():
    bsdf = HairBSDF(longitudinal_roughness=0.3, azimuthal_roughness=0.25)

    t = _norm([1.0, 0.0, 0.0])  # hair tangent along +X
    v = _norm([0.0, 0.0, 1.0])  # view toward +Z

    # Light directions: one aligned more with tangent, one less aligned
    l_aligned = _norm([1.0, 0.2, 1.0])
    l_offaxis = _norm([0.0, 0.5, 1.0])

    rgb_aligned = bsdf.evaluate_kajiya_kay(l_aligned, v, t)
    rgb_offaxis = bsdf.evaluate_kajiya_kay(l_offaxis, v, t)

    # In the simplified evaluator, off-axis (smaller t·l) yields larger sin term and
    # higher specular; ensure a monotonic difference rather than a specific direction.
    assert float(rgb_offaxis.mean()) >= float(rgb_aligned.mean())


def test_outputs_are_finite():
    bsdf = HairBSDF()
    t = _norm([1.0, 0.0, 0.0])
    v = _norm([0.0, 0.0, 1.0])
    l = _norm([0.2, 0.7, 0.7])
    rgb = bsdf.evaluate_kajiya_kay(l, v, t)
    assert np.all(np.isfinite(rgb))
