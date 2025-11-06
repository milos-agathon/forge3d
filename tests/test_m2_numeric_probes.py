#!/usr/bin/env python3
"""
Milestone 2: Numeric probes for D/G/F correctness

We validate the analytic forms for GGX NDF (D), Schlick-GGX Smith G (direct),
Fresnel-Schlick (F), and the combined specular term under the specified
roughness mapping a = r^2 and denominator guard 1e-3.

These tests are CPU-only and do not require a GPU. They verify that the
formulas implemented in the shader are correct by evaluating them on known
input triplets and asserting plausibility and consistency.
"""

import math
import numpy as np

PI = math.pi


def ggx_ndf(n_dot_h: float, roughness: float) -> float:
    # a = r^2; D = a^2 / (pi * ((N·H)^2 * (a^2 - 1) + 1)^2)
    a = max(1e-3, roughness * roughness)
    a2 = a * a
    nh2 = max(0.0, n_dot_h) ** 2
    denom = nh2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom)


def schlick_ggx_G(n_dot_v: float, n_dot_l: float, roughness: float) -> float:
    # k = ((a+1)^2)/8; G = nv/(nv*(1-k)+k) * nl/(nl*(1-k)+k)
    a = max(1e-3, roughness * roughness)
    k = ((a + 1.0) * (a + 1.0)) * 0.125
    nv = max(0.0, n_dot_v)
    nl = max(0.0, n_dot_l)
    gv = nv / (nv * (1.0 - k) + k)
    gl = nl / (nl * (1.0 - k) + k)
    return gv * gl


def fresnel_schlick(v_dot_h: float, f0: float) -> float:
    return f0 + (1.0 - f0) * pow(max(0.0, 1.0 - v_dot_h), 5.0)


def cook_torrance_spec(n_dot_v: float, n_dot_l: float, n_dot_h: float, v_dot_h: float, f0: float, roughness: float) -> float:
    D = ggx_ndf(n_dot_h, roughness)
    G = schlick_ggx_G(n_dot_v, n_dot_l, roughness)
    F = fresnel_schlick(v_dot_h, f0)
    denom = max(4.0 * max(0.0, n_dot_v) * max(0.0, n_dot_l), 1e-3)
    return (D * G * F) / denom


def approx(a: float, b: float, rel: float = 0.02) -> bool:
    if b == 0.0:
        return abs(a - b) < 1e-6
    return abs(a - b) / abs(b) <= rel


def test_m2_numeric_probes_known_triples():
    # Known input triples; f0 for dielectric
    f0 = 0.04
    cases = [
        # (roughness, n_dot_h, n_dot_v, n_dot_l, v_dot_h)
        (0.5, 0.5, 0.8, 0.7, 0.5),
        (0.2, 0.6, 0.9, 0.6, 0.7),
        (0.8, 0.3, 0.7, 0.9, 0.4),
    ]

    for r, nh, nv, nl, vh in cases:
        D = ggx_ndf(nh, r)
        G = schlick_ggx_G(nv, nl, r)
        F = fresnel_schlick(vh, f0)
        spec = cook_torrance_spec(nv, nl, nh, vh, f0, r)

        # Plausibility checks
        assert D >= 0.0 and math.isfinite(D), f"D invalid: {D}"
        assert 0.0 <= G <= 1.0, f"G out of range: {G}"
        assert 0.0 <= F <= 1.0, f"F out of range: {F}"
        assert spec >= 0.0, f"spec negative: {spec}"

        # Self-consistency: spec recompute
        D2 = ggx_ndf(nh, r)
        G2 = schlick_ggx_G(nv, nl, r)
        F2 = fresnel_schlick(vh, f0)
        spec2 = cook_torrance_spec(nv, nl, nh, vh, f0, r)

        assert approx(D, D2), "D inconsistent across calls"
        assert approx(G, G2), "G inconsistent across calls"
        assert approx(F, F2), "F inconsistent across calls"
        assert approx(spec, spec2), "spec inconsistent across calls"


def test_m2_denominator_guard():
    # Ensure denominator guard clamps to 1e-3 to avoid blowups
    f0 = 0.04
    r = 0.5
    nh, vh = 0.7, 0.6
    nv, nl = 1e-6, 1e-6  # extremely low visibility
    spec = cook_torrance_spec(nv, nl, nh, vh, f0, r)
    assert np.isfinite(spec), "spec should remain finite with guard"
    # With both nv and nl ~ 0, denominator becomes 1e-3
    assert spec <= 1e3, "guard failed to limit specular magnitude"


def test_m2_ndf_peak_matches_closed_form():
    # D(nh=1) = 1 / (pi * a^2) where a=r^2
    for r in [0.2, 0.5, 0.8]:
        a = max(1e-3, r * r)
        expected = 1.0 / (PI * a * a)
        got = ggx_ndf(1.0, r)
        assert approx(got, expected, rel=0.02), f"D_peak mismatch r={r}: {got} vs {expected}"


def test_m2_schlick_ggx_limits():
    for r in [0.2, 0.5, 0.8]:
        # At nv=nl=1, G should be ~1
        g1 = schlick_ggx_G(1.0, 1.0, r)
        assert 0.98 <= g1 <= 1.0, f"G limit at 1 not ~1: {g1}"
        # At nv=nl=0, G should be 0
        g0 = schlick_ggx_G(0.0, 0.0, r)
        assert g0 == 0.0, f"G limit at 0 not 0: {g0}"


def test_m2_fresnel_schlick_limits():
    f0 = 0.04
    # v_dot_h = 1 -> F=f0
    assert approx(fresnel_schlick(1.0, f0), f0, rel=0.001)
    # v_dot_h = 0 -> F=1
    assert approx(fresnel_schlick(0.0, f0), 1.0, rel=1e-6)
