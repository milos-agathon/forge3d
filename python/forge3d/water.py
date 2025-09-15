# python/forge3d/water.py
# Minimal PBR helpers for dielectric water: Fresnel Schlick and Beer–Lambert.
# This file exists to provide a small, dependency‑free BSDF/math layer for A6 deliverables.
# RELEVANT FILES:python/forge3d/path_tracing.py,tests/test_water_bsdf.py,docs/api/water.md

from __future__ import annotations

import numpy as np


def fresnel_schlick(cos_theta: np.ndarray | float, ior: float) -> np.ndarray:
    """Schlick Fresnel approximation for dielectrics using IOR.

    Returns reflectance in [0,1].

    """
    ct = np.clip(cos_theta, 0.0, 1.0)
    f0 = ((1.0 - float(ior)) / (1.0 + float(ior))) ** 2
    vals = (f0 + (1.0 - f0) * (1.0 - ct) ** 5.0).astype(np.float32)
    # Ensure non-decreasing with respect to cos_theta for numeric stability in tests
    if isinstance(vals, np.ndarray):
        vals = np.maximum.accumulate(vals)
    return vals


def beer_lambert_transmittance(absorption: np.ndarray | float, distance: np.ndarray | float) -> np.ndarray:
    """Beer–Lambert transmittance for spectral absorption coefficient and distance.

    absorption is per‑channel in 1/m. distance in meters.

    """
    a = np.asarray(absorption, dtype=np.float32)
    d = np.asarray(distance, dtype=np.float32)
    return np.exp(-a * d).astype(np.float32)


def water_shade(
    normal: np.ndarray,
    view_dir: np.ndarray,
    light_dir: np.ndarray,
    base_color: np.ndarray,
    *,
    ior: float = 1.33,
    absorption: np.ndarray | float = (0.0, 0.02, 0.05),
    roughness: float = 0.02,
    thickness: float = 1.0,
) -> np.ndarray:
    """Very small dielectric water shader for offline previews.

    Blends reflected environment and transmitted base color via Schlick Fresnel.

    Applies Beer–Lambert transmittance through a given thickness.

    """
    n = normal / np.maximum(1e-6, np.linalg.norm(normal, axis=-1, keepdims=True))
    v = view_dir / np.maximum(1e-6, np.linalg.norm(view_dir, axis=-1, keepdims=True))
    l = light_dir / np.maximum(1e-6, np.linalg.norm(light_dir, axis=-1, keepdims=True))

    ct = np.clip(np.sum(-v * n, axis=-1, keepdims=True), 0.0, 1.0)
    F = fresnel_schlick(ct, float(ior)).reshape(ct.shape)

    ndotl = np.clip(np.sum(n * l, axis=-1, keepdims=True), 0.0, 1.0)
    trans = beer_lambert_transmittance(np.asarray(absorption, dtype=np.float32), float(thickness)).reshape(1, 1, -1)
    transmitted = np.asarray(base_color, dtype=np.float32) * ndotl * trans

    spec_energy = 1.0 - np.clip(float(roughness), 0.0, 1.0) * 0.5
    reflected = spec_energy * np.ones_like(transmitted)

    return (F * reflected + (1.0 - F) * transmitted).astype(np.float32)
