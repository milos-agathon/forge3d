"""A normal-free pixel is sky, not a black denoising sample."""
import numpy as np
from forge3d.denoise import atrous_denoise


def test_atrous_preserves_every_unguided_sky_pixel():
    rows, cols = np.mgrid[:8, :12]
    color = np.stack((rows / 8, cols / 12, (rows + cols) / 20), axis=-1).astype(np.float32)
    normal = np.zeros_like(color)
    normal[4:, :, 2] = 1
    result = atrous_denoise(color, normal=normal)
    np.testing.assert_array_equal(result[:4], color[:4])
    assert np.isfinite(result).all()


def test_atrous_all_sky_is_an_exact_passthrough():
    color = np.full((8, 12, 3), (.2, .4, .7), np.float32)
    np.testing.assert_array_equal(atrous_denoise(color, normal=np.zeros_like(color)), color)
