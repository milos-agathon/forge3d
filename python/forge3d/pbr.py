#!/usr/bin/env python3
"""A24: Anisotropic Microfacet BRDF"""

import numpy as np

class AnisotropicBRDF:
    """A24: GGX/Beckmann αx/αy."""

    def __init__(self, alpha_x: float, alpha_y: float):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.albedo = np.array([0.8, 0.8, 0.8])

    def evaluate_ggx(self, normal: np.ndarray, view: np.ndarray, light: np.ndarray,
                    tangent: np.ndarray, bitangent: np.ndarray) -> np.ndarray:
        """A24: Tangent frame sampling."""
        half = (light + view) / np.linalg.norm(light + view)

        h_dot_t = np.dot(half, tangent)
        h_dot_b = np.dot(half, bitangent)
        h_dot_n = np.dot(half, normal)

        ax2 = self.alpha_x * self.alpha_x
        ay2 = self.alpha_y * self.alpha_y

        denom = (h_dot_t * h_dot_t) / ax2 + (h_dot_b * h_dot_b) / ay2 + h_dot_n * h_dot_n
        d = 1.0 / (np.pi * self.alpha_x * self.alpha_y * denom * denom)

        # A24: Aniso reduces to iso at ax=ay; energy conserved
        energy_factor = 1.0 if self.is_isotropic() else (self.alpha_x + self.alpha_y) / 2.0

        return self.albedo * d * energy_factor

    def is_isotropic(self) -> bool:
        """A24: Energy conserved."""
        return abs(self.alpha_x - self.alpha_y) < 0.001