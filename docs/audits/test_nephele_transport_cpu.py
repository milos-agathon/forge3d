"""Analytic CPU checks for the audit geometry and ablation source domain."""
from pathlib import Path
import unittest

import numpy as np

from nephele_transport_cpu import Scene, density_integral, height_one, terrain_hit


class TransportChecks(unittest.TestCase):
    def test_constant_density_exact_clipping(self):
        value = density_integral(np.array([-1., 1., 1.]), np.array([1., 0., 0.]), np.inf, np.ones((3, 3, 3)), np.zeros(3), np.full(3, 2.))
        self.assertEqual(value, 2.)

    def test_cubic_density_integral(self):
        z, y, x = np.indices((2, 2, 2))
        grid = ((1 + x) * (1 + 2 * y) * (1 + 3 * z)).astype(float)
        value = density_integral(np.full(3, .5), np.ones(3), 1., grid, np.zeros(3), np.full(3, 2.))
        # Integral of (1+t)(1+2t)(1+3t), 0<=t<=1.
        np.testing.assert_array_max_ulp(value, 55 / 6)

    def test_secondary_hit_from_below(self):
        flat, spacing = np.zeros((2, 2)), np.full(2, 4.)
        self.assertEqual(terrain_hit(np.array([0., -1., 0.]), np.array([0., 1., 0.]), flat, spacing, 1e-3), 1.)
        self.assertEqual(terrain_hit(np.array([0., 1., 0.]), np.array([0., -1., 0.]), flat, spacing, 1e-3), 1.)
        self.assertTrue(np.isinf(terrain_hit(np.array([3., 1., 0.]), np.array([0., -1., 0.]), flat, spacing, 1e-3)))

    def test_excluded_contact_keeps_later_bilinear_root(self):
        saddle = np.array([[1., -1.], [-1., 1.]])
        self.assertEqual(terrain_hit(np.zeros(3), np.array([1., .5, 1.]), saddle, np.full(2, 2.), 1e-3), .5)

    def test_valid_air_is_independent_of_visibility(self):
        scene = Scene(Path(__file__).resolve().parents[2])
        p = np.zeros(3)
        p[1] = height_one(p, scene.dem, scene.spacing) - .5
        solar, environment, _, valid = scene.source(p[None], np.array([[0., 0., 1.]]))
        self.assertFalse(valid[0])
        self.assertTrue(np.array_equal(solar, np.zeros_like(solar)))
        self.assertTrue(np.array_equal(environment, np.zeros_like(environment)))

    def test_reference_open_surface_retains_medium_on_both_sides(self):
        scene = Scene(Path(__file__).resolve().parents[2])
        p = np.zeros(3)
        p[1] = height_one(p, scene.dem, scene.spacing) - .5
        solar, environment, rho, above_surface = scene.source(p[None], np.array([[0., 0., 1.]]), reject_interior=False)
        self.assertFalse(above_surface[0])
        self.assertGreater(rho[0], 0.)
        self.assertTrue(np.all(solar[0, 0] > 0))
        self.assertTrue(np.all(environment[0, 0] > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
