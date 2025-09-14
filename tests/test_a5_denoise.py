# tests/test_a5_denoise.py
# Minimal SSIM-based check that A-trous denoiser improves noisy image quality.
# Ensures deterministic output and guides via albedo/normal/depth features.
# RELEVANT FILES:python/forge3d/denoise.py,tests/_ssim.py,docs/api/denoise.md

import numpy as np
from tests._ssim import ssim

from python.forge3d.denoise import atrous_denoise


def _make_synthetic_scene(h=64, w=64):
    # Clean image: two colored squares and a diagonal edge
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[: h // 2, : w // 2, 0] = 0.9  # red block
    img[: h // 2, w // 2 :, 1] = 0.8  # green block
    for i in range(h):
        j = int((i / max(1, h - 1)) * (w - 1))
        img[i, : j, 2] = 0.6  # blue gradient left of diagonal

    # Albedo ~ base color without shading
    albedo = img.copy()

    # Normal: point up for top half, front for bottom half (edge guidance)
    normal = np.zeros_like(img)
    normal[: h // 2, :, :] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    normal[h // 2 :, :, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Depth: shallow on left, deeper on right
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    depth = np.tile(xx[None, :], (h, 1))

    return img, albedo, normal, depth


def test_atrous_denoise_improves_ssim():
    clean, albedo, normal, depth = _make_synthetic_scene()

    rng = np.random.default_rng(1234)
    noise = rng.normal(0.0, 0.05, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0).astype(np.float32)

    den = atrous_denoise(
        noisy,
        albedo=albedo,
        normal=normal,
        depth=depth,
        iterations=5,
        sigma_color=0.30,
        sigma_albedo=0.30,
        sigma_normal=0.60,
        sigma_depth=0.80,
    )

    # SSIM against clean reference
    ssim_noisy = ssim((noisy * 255).astype(np.uint8), (clean * 255).astype(np.uint8), data_range=255.0)
    ssim_den = ssim((den * 255).astype(np.uint8), (clean * 255).astype(np.uint8), data_range=255.0)

    assert ssim_den > ssim_noisy + 0.03  # improvement


def test_atrous_denoise_deterministic():
    clean, albedo, normal, depth = _make_synthetic_scene()
    rng = np.random.default_rng(42)
    noisy = np.clip(clean + rng.normal(0.0, 0.08, size=clean.shape).astype(np.float32), 0.0, 1.0)

    kwargs = dict(
        albedo=albedo,
        normal=normal,
        depth=depth,
        iterations=2,
        sigma_color=0.12,
        sigma_albedo=0.20,
        sigma_normal=0.35,
        sigma_depth=0.45,
    )

    den1 = atrous_denoise(noisy, **kwargs)
    den2 = atrous_denoise(noisy, **kwargs)
    assert np.allclose(den1, den2, atol=1e-6)
