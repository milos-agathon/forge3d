"""Deterministic checks for the independent CPU path estimator."""
import unittest
import numpy as np
from nephele_multiple_scattering_cpu import uniform, phase, local_direction, trace


class PathChecks(unittest.TestCase):
    def test_sample_identity_matches_rust_integer_contract(self):
        mask = (1 << 64) - 1
        rot = lambda x, n: ((x << n) | (x >> (64 - n))) & mask
        for frame, pixel, sample, bounce, dimension in ((1313165384, 4095, 639, 3, 40), (0, 0, 0, 0, 0), (mask, mask, mask, 127, 2000)):
            value = frame ^ rot(pixel, 13) ^ rot(sample, 29) ^ rot(bounce, 47) ^ ((dimension * 0x9e3779b97f4a7c15) & mask)
            value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
            value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
            value ^= value >> 31
            expected = ((value >> 41) + .5) / (1 << 23)
            self.assertEqual(uniform(np.uint64(frame), np.uint64(pixel), np.uint64(sample), bounce, dimension), expected)

    def test_phase_and_local_basis(self):
        self.assertEqual(phase(.5, 0.), 1 / (4 * np.pi))
        for axis in np.eye(3):
            direction = local_direction(axis, .45, np.pi)
            np.testing.assert_allclose(np.dot(direction, axis), .45)
            np.testing.assert_allclose(np.linalg.norm(direction), 1.)

    def test_vacuum_and_open_surface_from_below(self):
        grid = np.zeros((2, 2, 2))
        dem = np.zeros((2, 2))
        sigma = np.zeros(3)
        for y, expected in ((1., .25), (-1., .6)):
            # Unit albedo and a direct upward sun at pi radiance: hitting the
            # surface from below still reflects upward, as in the reference.
            full, first, _ = trace(np.array([0., y, 0.]), np.array([0., 1., 0.]), 0, 0, 0, True,
                                   grid, -np.ones(3), np.ones(3), dem, np.ones(2), sigma, sigma,
                                   np.array([0., 1., 0.]), np.ones(3) * np.pi, .25 if y > 0 else 0., np.ones(3) * .6, 0., 0.)
            np.testing.assert_allclose(full, expected)
            np.testing.assert_array_equal(first, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
