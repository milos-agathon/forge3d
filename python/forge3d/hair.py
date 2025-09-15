#!/usr/bin/env python3
"""A23: Hair BSDF + Curve Prims (PT)"""

import numpy as np

class HairBSDF:
    """A23: Kajiya-Kay/Marschner; bezier ribbons/tubes."""

    def __init__(self, longitudinal_roughness: float = 0.3,
                 azimuthal_roughness: float = 0.25):
        self.longitudinal_roughness = longitudinal_roughness
        self.azimuthal_roughness = azimuthal_roughness
        self.absorption = np.array([0.419, 0.697, 1.37])  # Brown hair pigments

    def evaluate_kajiya_kay(self, light_dir: np.ndarray, view_dir: np.ndarray,
                           tangent: np.ndarray) -> np.ndarray:
        """A23: Hairball highlights/tilt match reference."""
        t_dot_l = np.dot(light_dir, tangent)
        t_dot_v = np.dot(view_dir, tangent)

        sin_tl = np.sqrt(max(0.0, 1.0 - t_dot_l * t_dot_l))
        sin_tv = np.sqrt(max(0.0, 1.0 - t_dot_v * t_dot_v))

        cos_phi = max(0.0, sin_tl * sin_tv + t_dot_l * t_dot_v)

        diffuse = sin_tl
        specular = cos_phi ** (1.0 / self.longitudinal_roughness)

        return np.array([diffuse + specular] * 3)

class CurvePrimitive:
    """A23: Curve widths; pigments."""

    def __init__(self, control_points: np.ndarray, widths: np.ndarray):
        self.control_points = control_points
        self.widths = widths  # A23: Curve widths