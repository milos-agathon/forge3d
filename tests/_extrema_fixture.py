"""Deterministic analytic, band-limited fractal, and damaged versions of one grid."""

import numpy as np
from affine import Affine

TRANSFORM = Affine(3, 1, 500, 0.5, -4, 1200)


def surfaces():
    y, x = np.indices((96, 96))
    smooth = 120 + 0.3 * x + 0.17 * y + 0.002 * x * y + 0.0003 * y * y
    rng = np.random.default_rng(82)
    terrain = smooth.copy()
    for octave in (1, 2, 4, 8, 16):
        # Fractal synthesis with seeded orientations and octave-decaying amplitude.
        for angle, phase in rng.uniform(0, 2 * np.pi, (4, 2)):
            distance = x * np.cos(angle) + y * np.sin(angle)
            terrain += 2 / octave * np.sin(distance * octave * 2 * np.pi / 96 + phase)
    damaged = np.round(terrain * 4) / 4
    damaged += np.array([0, 2, 0, -2])[y % 4]
    damaged += np.array([1, -1])[x % 2]
    damaged[:, 64:] += 9
    damaged[46, 39] += 72
    damaged[54, 26] -= 54
    damaged[12:16, 9:14] = np.nan
    damaged[70:78, 30:38] = 123.125
    return smooth, terrain, damaged
