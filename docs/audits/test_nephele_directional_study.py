"""Analytic checks for the new audit's ray operator, independent of captures."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("directional_study", HERE / "nephele-directional-study-2026-10-22.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class DirectionalChecks(unittest.TestCase):
    def operator(self, enabled, density=1., sigma=None):
        # Start below an open horizontal surface and look upward. The medium
        # continues on BOTH sides; only the enabled ray ends on the surface.
        point = np.array([[0., .5, 0.]])
        direction = np.array([[0., 1., 0.]])
        grid = np.full((3, 3, 3), density)
        lo, hi = np.array([-1., 0., -1.]), np.array([1., 2., 1.])
        dem, spacing = np.ones((2, 2)), np.full(2, 2.)
        sigma = np.array([.1, .2, .3]) if sigma is None else sigma
        op = study.ray_operator(point, direction, enabled, grid, lo, hi, dem, spacing, sigma,
                                np.array([-2., 0., 0.]), np.array([1., 0., 0.]),
                                np.array([0., 0., 1.]), np.array([0., 1., 0.]), 1., np.zeros(6400, np.int32), dem.shape)
        return op, sigma

    def test_exact_constant_density_characteristic(self):
        for enabled, length in ((True, .5), (False, 1.5)):
            op, sigma = self.operator(enabled)
            np.testing.assert_allclose(op[2][0, 0], np.exp(-sigma * length), rtol=np.finfo(np.float32).eps, atol=0)
            expected = -np.expm1(-sigma * length) / sigma
            np.testing.assert_allclose(op[1][0, 0].astype(float).sum(0), expected, rtol=np.finfo(np.float32).eps, atol=0)

    def test_open_surface_switch_changes_visibility_not_source_domain(self):
        enabled, _ = self.operator(True)
        disabled, _ = self.operator(False)
        self.assertTrue(np.all(enabled[0] >= 0))
        self.assertTrue(np.all(disabled[0] >= 0))
        self.assertTrue(np.all(enabled[3] >= 0))
        self.assertTrue(np.all(disabled[3] == -1))

    def test_vacuum_preserves_environment(self):
        op, _ = self.operator(False, density=0.)
        source, surface = np.zeros((1, 1, 3), np.float32), np.zeros((4, 3), np.float32)
        value = study.apply_operator(*op, source, surface, .25)
        np.testing.assert_array_equal(value, np.full((1, 1, 3), .25))

    def test_zero_extinction_uses_the_density_integral_limit(self):
        op, _ = self.operator(False, sigma=np.zeros(3))
        np.testing.assert_array_equal(op[2], np.ones((1, 1, 3)))
        np.testing.assert_allclose(op[1][0, 0].astype(float).sum(0), 1.5, rtol=np.finfo(np.float32).eps, atol=0)

    def test_hg_continuation_retains_direction(self):
        g = .45
        forward, sideways, backward = study.hg(np.array([1., 0., -1.]), g)
        self.assertGreater(forward, sideways)
        self.assertGreater(sideways, backward)
        np.testing.assert_allclose(forward / backward, ((1 + g) / (1 - g)) ** 3,
                                   rtol=np.finfo(float).eps * 3, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
