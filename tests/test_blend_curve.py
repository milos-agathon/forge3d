"""
Tests for Milestone D1: Blend curve verification.

Verifies the bandlimit fade curve is:
- Monotonic decreasing
- Correctly bounded at lod_lo and lod_hi
- Correctly valued at midpoint

RELEVANT FILES: docs/flake_debug_contract.md
"""
from __future__ import annotations

import numpy as np
import pytest

# Blend curve parameters (must match shader)
LOD_LO = 1.0
LOD_HI = 4.0


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """GLSL/WGSL smoothstep implementation."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def blend(lod: float) -> float:
    """Compute blend factor for given LOD."""
    return 1.0 - smoothstep(LOD_LO, LOD_HI, lod)


class TestBlendCurve:
    """Milestone D1: Blend curve unit tests."""
    
    def test_monotonic_decreasing(self):
        """Blend must be monotonically decreasing with LOD."""
        lod_samples = np.linspace(0, 5, 100)
        blend_values = [blend(l) for l in lod_samples]
        
        for i in range(len(blend_values) - 1):
            assert blend_values[i] >= blend_values[i + 1], (
                f"Monotonicity violated: blend({lod_samples[i]:.2f})={blend_values[i]:.4f} < "
                f"blend({lod_samples[i+1]:.2f})={blend_values[i+1]:.4f}"
            )
    
    def test_clamped_zero_to_one(self):
        """Blend must be in [0, 1] range."""
        lod_samples = np.linspace(-1, 10, 200)
        for l in lod_samples:
            b = blend(l)
            assert 0.0 <= b <= 1.0, f"blend({l})={b} out of [0,1] range"
    
    def test_below_lod_lo_equals_one(self):
        """Blend should be 1.0 for LOD <= lod_lo."""
        for lod in [0.0, 0.5, LOD_LO - 0.1, LOD_LO]:
            b = blend(lod)
            # At exactly lod_lo, smoothstep returns 0, so blend = 1
            if lod < LOD_LO:
                assert b == 1.0, f"blend({lod}) should be 1.0 below lod_lo, got {b}"
            else:
                # At boundary, allow small tolerance
                assert abs(b - 1.0) < 0.01, f"blend({lod}) should be ~1.0 at lod_lo, got {b}"
    
    def test_above_lod_hi_equals_zero(self):
        """Blend should be 0.0 for LOD >= lod_hi."""
        for lod in [LOD_HI, LOD_HI + 0.1, 5.0, 10.0]:
            b = blend(lod)
            if lod > LOD_HI:
                assert b == 0.0, f"blend({lod}) should be 0.0 above lod_hi, got {b}"
            else:
                # At boundary, allow small tolerance
                assert abs(b - 0.0) < 0.01, f"blend({lod}) should be ~0.0 at lod_hi, got {b}"
    
    def test_midpoint_value(self):
        """Blend at midpoint (lod=2.5) should be 0.5 for symmetric range."""
        midpoint = (LOD_LO + LOD_HI) / 2.0
        expected = 0.5
        actual = blend(midpoint)
        
        assert abs(actual - expected) < 0.01, (
            f"blend({midpoint}) should be {expected} at midpoint, got {actual}"
        )
    
    def test_exact_values_at_key_points(self):
        """Verify exact blend values at specific LOD points."""
        test_cases = [
            (0.0, 1.0),      # Below lod_lo
            (0.5, 1.0),      # Below lod_lo
            (1.0, 1.0),      # At lod_lo
            (2.5, 0.5),      # Midpoint
            (4.0, 0.0),      # At lod_hi
            (5.0, 0.0),      # Above lod_hi
        ]
        
        for lod, expected in test_cases:
            actual = blend(lod)
            assert abs(actual - expected) < 0.02, (
                f"blend({lod}) expected {expected}, got {actual}"
            )
    
    def test_smoothstep_hermite_interpolation(self):
        """Verify smoothstep uses correct Hermite interpolation."""
        # Smoothstep derivative should be zero at boundaries
        eps = 0.001
        
        # At lod_lo (edge0), derivative should be 0
        d_lo = (blend(LOD_LO - eps) - blend(LOD_LO + eps)) / (2 * eps)
        
        # At lod_hi (edge1), derivative should be 0
        d_hi = (blend(LOD_HI - eps) - blend(LOD_HI + eps)) / (2 * eps)
        
        # Derivatives at boundaries should be very small (approaching 0)
        assert abs(d_lo) < 0.1, f"Derivative at lod_lo should be ~0, got {d_lo}"
        assert abs(d_hi) < 0.1, f"Derivative at lod_hi should be ~0, got {d_hi}"
