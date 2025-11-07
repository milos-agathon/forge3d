#!/usr/bin/env python3
"""Unit tests for Milestone 5 tone mapping utilities."""

import numpy as np
import pytest

from forge3d import tonemap as tm


REFERENCE_ACES = np.array(
    [
        0.0,
        0.3743083140,
        0.6191154269,
        0.7374565521,
        0.8035617826,
        0.8450334392,
        0.8732640036,
        0.8936434627,
        0.9090137678,
        0.9210037708,
        0.9306099309,
        0.9384743803,
        0.9450288044,
        0.9505737612,
        0.9553247687,
        0.9594402922,
        0.9630393942,
        0.9662132303,
        0.9690327527,
        0.9715539999,
        0.9738218119,
        0.9758724882,
        0.9777357256,
        0.9794360512,
        0.9809938988,
        0.9824264274,
        0.9837481509,
        0.9849714284,
        0.9861068492,
        0.9871635381,
        0.9881494002,
        0.9890713177,
        0.9899353099,
    ],
    dtype=np.float32,
)


def _greyscale_stack(samples: np.ndarray) -> np.ndarray:
    return np.stack([samples, samples, samples], axis=1)


def test_apply_tonemap_invalid_mode() -> None:
    data = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        tm.apply_tonemap(data, "invalid-mode")


def test_linear_tonemap_identity() -> None:
    color = np.array([[0.2, 0.5, 1.2]], dtype=np.float32)
    expected = np.clip(color, 0.0, None)
    result = tm.apply_tonemap(color, tm.TONEMAP_LINEAR)
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_aces_matches_reference_curve() -> None:
    samples = np.linspace(0.0, 16.0, 33, dtype=np.float32)
    rgb = _greyscale_stack(samples)
    mapped = tm.tonemap_aces(rgb)
    np.testing.assert_allclose(mapped[:, 0], REFERENCE_ACES, atol=1e-3)


def test_tone_curves_are_monotonic() -> None:
    samples = np.linspace(0.0, 32.0, 257, dtype=np.float32)
    rgb = _greyscale_stack(samples)
    for func in (tm.tonemap_reinhard, tm.tonemap_aces):
        mapped = func(rgb)[:, 0]
        diffs = np.diff(mapped)
        # Allow tiny negative noise due to float precision.
        assert np.all(diffs >= -1e-6), f"{func.__name__} is not monotonic"
