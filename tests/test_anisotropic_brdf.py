#!/usr/bin/env python3
"""A24: Tests for Anisotropic Microfacet BRDF

Covers:
- Isotropic reduction (ax == ay) invariance to tangent/bitangent rotation
- Anisotropy symmetry (swap ax/ay corresponds to swapping tangent/bitangent)
"""

import numpy as np

from forge3d.pbr import AnisotropicBRDF


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / (n + 1e-8)).astype(np.float32)


def test_isotropic_flag_and_rotation_invariance():
    brdf = AnisotropicBRDF(alpha_x=0.3, alpha_y=0.3)
    assert brdf.is_isotropic() is True

    # Basis
    n = _norm([0.0, 0.0, 1.0])
    v = _norm([0.0, 0.0, 1.0])
    l = _norm([0.6, 0.0, 0.8])
    t = _norm([1.0, 0.0, 0.0])
    b = _norm([0.0, 1.0, 0.0])

    rgb_tb = brdf.evaluate_ggx(n, v, l, t, b)
    rgb_bt = brdf.evaluate_ggx(n, v, l, b, t)

    # With ax==ay the response should be invariant to swapping tangent and bitangent
    assert np.allclose(rgb_tb, rgb_bt, rtol=1e-5, atol=1e-6)


def test_anisotropy_symmetry_under_axis_swap():
    # If we swap alpha_x/alpha_y and simultaneously swap tangent/bitangent,
    # the BRDF response should remain close (up to numerical tolerance)
    n = _norm([0.0, 0.0, 1.0])
    v = _norm([0.0, 0.0, 1.0])
    l = _norm([0.5, 0.2, 0.84])
    t = _norm([1.0, 0.0, 0.0])
    b = _norm([0.0, 1.0, 0.0])

    brdf_xy = AnisotropicBRDF(alpha_x=0.2, alpha_y=0.5)
    brdf_yx = AnisotropicBRDF(alpha_x=0.5, alpha_y=0.2)

    rgb_xy = brdf_xy.evaluate_ggx(n, v, l, t, b)
    rgb_yx = brdf_yx.evaluate_ggx(n, v, l, b, t)  # swap axis basis

    # Expect symmetry within a small tolerance (shading model approximations)
    assert np.allclose(rgb_xy, rgb_yx, rtol=1e-3, atol=1e-5)
